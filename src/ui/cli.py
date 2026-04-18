"""Rich CLI — interactive agent terminal with streaming output."""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import uuid
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import patch_stdout
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from src.core.agent_loop import agent_loop
from src.core.config import load_config
from src.core.env_injector import EnvironmentInjector
from src.core.iteration_budget import IterationBudget
from src.core.plan_mode import PlanMode
from src.core.streaming_executor import StreamingExecutor
from src.core.types import (
    BudgetExhaustedEvent, ErrorEvent, FinalAnswerEvent,
    PermissionDecision, ToolCall, ToolCallStartEvent,
    ToolContext, ToolResultEvent, TokenEvent,
)
from src.providers.resolver import resolve_provider
import src.tools  # trigger self-registration
from src.tools.registry import get_registry

logger = logging.getLogger(__name__)
console = Console()


async def _permission_prompt(tc: ToolCall) -> PermissionDecision:
    """Interactive permission prompt for write/execute tools."""
    console.print(f"\n[bold yellow]Permission required:[/] {tc.name}({tc.arguments})")
    while True:
        try:
            resp = input("  Allow? (y)es / (n)o / (a)lways: ").strip().lower()
        except EOFError:
            return PermissionDecision.BLOCK
        if resp in ("y", "yes"):
            return PermissionDecision.ALLOW
        if resp in ("n", "no"):
            return PermissionDecision.BLOCK
        if resp in ("a", "always"):
            return PermissionDecision.ALLOW


async def _user_callback(question: str) -> str:
    """Callback for ask_user tool."""
    console.print(f"\n[bold cyan]Agent asks:[/] {question}")
    try:
        return input("  Your answer: ").strip()
    except EOFError:
        return ""


def _show_tools(registry) -> None:
    for t in sorted(registry._tools.values(), key=lambda t: t.name):
        status = "active" if registry.is_activated(t.name) else "deferred"
        console.print(f"  [{status:8s}] {t.name:20s} {t.description[:60]}")


async def _run_conversation(config, provider, registry, budget, plan_mode,
                            env_injector, executor, ctx) -> None:
    """Main REPL loop."""
    session = PromptSession()
    conversation: list = []
    session_id = str(uuid.uuid4())[:8]
    ctx.session_id = session_id

    console.print(Panel(
        f"[bold]Yigent Agent[/] | provider: {config.provider.name} | "
        f"model: {config.provider.model} | budget: {budget.total}",
        title="Session",
    ))
    console.print("[dim]Commands: /quit /tools /budget /plan[/]\n")

    while True:
        try:
            with patch_stdout():
                user_input = await session.prompt_async("You> ")
        except (EOFError, KeyboardInterrupt):
            break

        user_input = user_input.strip()
        if not user_input:
            continue

        # Slash commands
        if user_input == "/quit":
            break
        if user_input == "/tools":
            _show_tools(registry)
            continue
        if user_input == "/budget":
            console.print(f"Budget: {budget.remaining}/{budget.total}")
            continue
        if user_input == "/plan":
            if plan_mode.is_active:
                console.print("[yellow]Already in plan mode.[/]")
            else:
                plan_mode.enter(session_id)
                console.print("[yellow]Plan mode activated.[/]")
            continue

        conversation.append({"role": "user", "content": user_input})

        # Run agent loop and consume events
        text_streamed = False
        try:
            async for event in agent_loop(
                conversation=conversation, tools=registry, budget=budget,
                provider=provider, executor=executor, env_injector=env_injector,
                plan_mode=plan_mode, config=config,
                permission_callback=_permission_prompt,
            ):
                if isinstance(event, TokenEvent):
                    console.print(event.token, end="")
                    text_streamed = True
                elif isinstance(event, ToolCallStartEvent):
                    tc = event.tool_call
                    console.print(f"\n[dim]▶ {tc.name}({tc.arguments})[/]")
                elif isinstance(event, ToolResultEvent):
                    r = event.result
                    style = "red" if r.is_error else "green"
                    content = r.content[:500] + ("..." if len(r.content) > 500 else "")
                    console.print(Panel(content, title=r.name, border_style=style))
                elif isinstance(event, FinalAnswerEvent):
                    if text_streamed:
                        console.print()  # newline after streaming
                    else:
                        console.print(Markdown(event.content))
                elif isinstance(event, BudgetExhaustedEvent):
                    console.print("[bold red]Budget exhausted.[/]")
                    return
                elif isinstance(event, ErrorEvent):
                    style = "yellow" if event.recoverable else "red"
                    console.print(f"[{style}]{event.error}[/]")
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted.[/]")
            continue

    console.print("[dim]Goodbye.[/]")


async def _smoke_test(config, provider, registry, budget, plan_mode,
                      env_injector, executor, ctx) -> None:
    """Send one message, verify response, exit."""
    conversation = [{"role": "user", "content": "What tools do you have? List them briefly."}]
    got_answer = False
    async for event in agent_loop(
        conversation=conversation, tools=registry, budget=budget,
        provider=provider, executor=executor, env_injector=env_injector,
        plan_mode=plan_mode, config=config,
    ):
        if isinstance(event, TokenEvent):
            console.print(event.token, end="")
            got_answer = True
        elif isinstance(event, FinalAnswerEvent):
            if not got_answer:
                console.print(event.content[:500])
            got_answer = True
        elif isinstance(event, ErrorEvent):
            console.print(f"[red]{event.error}[/]")
    console.print()
    if got_answer:
        console.print("\n[green]Smoke test PASSED[/]")
        sys.exit(0)
    else:
        console.print("[red]Smoke test FAILED: no answer[/]")
        sys.exit(1)


async def async_main() -> None:
    parser = argparse.ArgumentParser(description="Yigent Agent CLI")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    config = load_config()
    provider = resolve_provider(config)
    registry = get_registry()
    budget = IterationBudget(config.agent.max_iterations)
    plan_mode = PlanMode(save_dir=config.plan_mode.save_dir)
    env_injector = EnvironmentInjector()

    ctx = ToolContext(
        plan_mode=plan_mode, registry=registry,
        config=config, working_dir=Path.cwd(),
        user_callback=_user_callback,
    )
    executor = StreamingExecutor(registry, ctx)

    if args.smoke_test:
        await _smoke_test(config, provider, registry, budget, plan_mode,
                         env_injector, executor, ctx)
    else:
        await _run_conversation(config, provider, registry, budget, plan_mode,
                               env_injector, executor, ctx)


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
