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
from rich.status import Status

from src.context.assembler import ContextAssembler
from src.context.engine import CompressionEngine
from src.core.agent_loop import _SYSTEM_PROMPT, agent_loop
from src.core.capability_router import CapabilityRouter
from src.core.config import load_config
from src.core.env_injector import EnvironmentInjector
from src.core.iteration_budget import IterationBudget
from src.core.multi_agent import TaskBoard
from src.core.plan_mode import PlanMode
from src.core.streaming_executor import StreamingExecutor
from src.core.types import (
    BudgetExhaustedEvent, ErrorEvent, FinalAnswerEvent,
    PermissionDecision, PlanModeTriggeredEvent, ReasoningDeltaEvent,
    ToolCall, ToolCallStartEvent,
    ToolContext, ToolResultEvent, TokenEvent, TurnStartedEvent,
)
from src.memory.markdown_store import MarkdownMemoryStore
from src.memory.working import WorkingMemory
from src.providers.resolver import (
    resolve_auxiliary, resolve_provider, resolve_scenario_router,
)
from src.safety.hook_system import HookSystem, load_hooks_from_config
from src.safety.permission_gate import PermissionGate
import src.tools  # trigger self-registration
from src.tools.mcp_adapter import MCPClient
from src.tools.registry import get_registry
from src.tools.task_tools import make_task_tools
from src.ui.slash_commands import DispatchResult, SlashDispatcher

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
                            env_injector, executor, ctx, assembler, hooks,
                            memory_store, scenario_router=None,
                            capability_router=None) -> None:
    """Main REPL loop."""
    session = PromptSession()
    working = WorkingMemory()
    conversation = working.conversation
    session_id = str(uuid.uuid4())[:8]
    ctx.session_id = session_id

    # Slash commands — Aider-style cmd_* introspection via SlashDispatcher.
    # Unknown /commands are intercepted here, not sent to the LLM.
    class _Commands:
        def cmd_quit(self, args: str):
            """Exit the session."""
            return SlashDispatcher.QUIT_SENTINEL

        def cmd_tools(self, args: str) -> None:
            """List registered tools and their activation status."""
            _show_tools(registry)

        def cmd_budget(self, args: str) -> None:
            """Show remaining iteration budget."""
            console.print(f"Budget: {budget.remaining}/{budget.total}")

        def cmd_plan(self, args: str) -> None:
            """Enter plan mode — blocks write/execute tools until approved."""
            if plan_mode.is_active:
                console.print("[yellow]Already in plan mode.[/]")
            else:
                plan_mode.enter(session_id)
                console.print("[yellow]Plan mode activated.[/]")

        def cmd_remember(self, args: str) -> None:
            """Save a fact to long-term memory: /remember TOPIC: CONTENT"""
            if memory_store is None:
                console.print("[red]Memory store not configured.[/]")
                return
            if ":" not in args:
                console.print(
                    "[yellow]Usage:[/] /remember TOPIC: CONTENT"
                )
                return
            topic, _, content = args.partition(":")
            topic = topic.strip()
            content = content.strip()
            if not topic or not content:
                console.print(
                    "[yellow]Both TOPIC and CONTENT are required.[/]"
                )
                return
            t = memory_store.write_topic(topic, content, title=topic)
            memory_store.record_index_entry(
                t.slug, t.title, content[:80],
            )
            console.print(
                f"[green]Saved[/] memory topic '{t.slug}' "
                f"({len(t.body)} chars)."
            )

        def cmd_memory(self, args: str) -> None:
            """List saved memory topics, or show one: /memory [TOPIC]"""
            if memory_store is None:
                console.print("[red]Memory store not configured.[/]")
                return
            args = args.strip()
            if not args:
                slugs = memory_store.list_topics()
                if not slugs:
                    console.print("[dim](no memory topics)[/]")
                    return
                console.print("[bold]Memory topics:[/]")
                for s in slugs:
                    console.print(f"  [cyan]{s}[/]")
                return
            t = memory_store.read_topic(args)
            if t is None:
                console.print(f"[yellow]No topic named[/] '{args}'")
                return
            console.print(f"[bold]{t.title}[/] [dim](updated {t.updated})[/]\n")
            console.print(t.body)

        def cmd_help(self, args: str) -> None:
            """Show available slash commands."""
            console.print("[bold]Available commands:[/]")
            for name, doc in sorted(dispatcher.commands.items()):
                console.print(f"  [cyan]/{name:<10}[/] {doc}")

        def report_unknown_command(self, name: str, known: list[str]) -> None:
            console.print(f"[red]Unknown command:[/] /{name}")
            console.print("[dim]Type /help for a list of commands.[/]")

    dispatcher = SlashDispatcher(_Commands())

    console.print(Panel(
        f"[bold]Yigent Agent[/] | provider: {config.provider.name} | "
        f"model: {config.provider.model} | budget: {budget.total}",
        title="Session",
    ))
    command_hint = " ".join(f"/{c}" for c in sorted(dispatcher.commands))
    console.print(f"[dim]Commands: {command_hint}[/]\n")

    while True:
        try:
            with patch_stdout():
                user_input = await session.prompt_async("You> ")
        except (EOFError, KeyboardInterrupt):
            break

        user_input = user_input.strip()
        if not user_input:
            continue

        # Slash commands — intercept before the LLM sees anything.
        result = await dispatcher.dispatch(user_input)
        if result is DispatchResult.QUIT:
            break
        if result is DispatchResult.HANDLED:
            continue

        user_msg = {"role": "user", "content": user_input}
        conversation.append(user_msg)

        # Run agent loop and consume events
        text_streamed = False
        thinking: Status | None = None

        def _stop_thinking() -> None:
            nonlocal thinking
            if thinking is not None:
                thinking.stop()
                thinking = None

        try:
            async for event in agent_loop(
                conversation=conversation, tools=registry, budget=budget,
                provider=provider, executor=executor, env_injector=env_injector,
                plan_mode=plan_mode, config=config,
                permission_callback=_permission_prompt,
                assembler=assembler, hooks=hooks,
                scenario_router=scenario_router,
                capability_router=capability_router,
            ):
                if isinstance(event, TurnStartedEvent):
                    # Turn began — show an immediate spinner so the user
                    # knows work has started, even before any reasoning or
                    # tokens have streamed back.
                    if thinking is None and not text_streamed:
                        thinking = console.status(
                            "[dim italic]Preparing...[/]", spinner="dots",
                        )
                        thinking.start()
                    continue
                if isinstance(event, ReasoningDeltaEvent):
                    # Reasoning started → refine the spinner label. If no
                    # spinner is running (turn event was missed somehow),
                    # start one here.
                    if thinking is None and not text_streamed:
                        thinking = console.status(
                            "[dim italic]Thinking...[/]", spinner="dots",
                        )
                        thinking.start()
                    else:
                        thinking.update("[dim italic]Thinking...[/]")  # type: ignore[union-attr]
                    continue
                if isinstance(event, TokenEvent):
                    _stop_thinking()
                    console.print(event.token, end="")
                    text_streamed = True
                elif isinstance(event, ToolCallStartEvent):
                    _stop_thinking()
                    tc = event.tool_call
                    console.print(f"\n[dim]▶ {tc.name}({tc.arguments})[/]")
                elif isinstance(event, ToolResultEvent):
                    _stop_thinking()
                    r = event.result
                    style = "red" if r.is_error else "green"
                    content = r.content[:500] + ("..." if len(r.content) > 500 else "")
                    console.print(Panel(content, title=r.name, border_style=style))
                elif isinstance(event, FinalAnswerEvent):
                    _stop_thinking()
                    if text_streamed:
                        console.print()  # newline after streaming
                    else:
                        console.print(Markdown(event.content))
                elif isinstance(event, BudgetExhaustedEvent):
                    _stop_thinking()
                    console.print("[bold red]Budget exhausted.[/]")
                    return
                elif isinstance(event, PlanModeTriggeredEvent):
                    _stop_thinking()
                    console.print(
                        f"[bold yellow]Plan mode triggered:[/bold yellow] {event.reason}",
                    )
                elif isinstance(event, ErrorEvent):
                    _stop_thinking()
                    style = "yellow" if event.recoverable else "red"
                    console.print(f"[{style}]{event.error}[/]")
        except KeyboardInterrupt:
            _stop_thinking()
            console.print("\n[yellow]Interrupted.[/]")
            continue
        finally:
            _stop_thinking()

    console.print("[dim]Goodbye.[/]")


async def _smoke_test(config, provider, registry, budget, plan_mode,
                      env_injector, executor, ctx, assembler, hooks,
                      scenario_router=None, capability_router=None) -> None:
    """Send one message, verify response, exit."""
    conversation = [{"role": "user", "content": "What tools do you have? List them briefly."}]
    got_answer = False
    async for event in agent_loop(
        conversation=conversation, tools=registry, budget=budget,
        provider=provider, executor=executor, env_injector=env_injector,
        plan_mode=plan_mode, config=config,
        assembler=assembler, hooks=hooks,
        scenario_router=scenario_router,
        capability_router=capability_router,
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
    scenario_router = resolve_scenario_router(config, provider)
    registry = get_registry()
    budget = IterationBudget(config.agent.max_iterations)
    plan_mode = PlanMode(save_dir=config.plan_mode.save_dir)
    env_injector = EnvironmentInjector()

    # Phase 2b Unit 5: shared TaskBoard exposed as task tools. The board is
    # stand-alone for now — spawn_fork / spawn_subagent wiring into the agent
    # loop is a Phase 3 concern.
    task_board = TaskBoard()
    for tool in make_task_tools(task_board):
        registry.register(tool)

    ctx = ToolContext(
        plan_mode=plan_mode, registry=registry,
        config=config, working_dir=Path.cwd(),
        user_callback=_user_callback,
    )
    # Phase 2a Unit 3 (revised 2026-04-20): markdown memory store at
    # ~/.yigent/memory/<project_hash>/. Claude-Code-style MEMORY.md index
    # + per-topic .md files. Replaces the earlier SQLite+FTS5 implementation.
    memory_store = MarkdownMemoryStore()
    try:
        memory_store.ensure_root()
    except OSError as exc:
        logger.warning("Memory store init failed: %s", exc)
        memory_store = None  # degrade gracefully

    # Phase 2 Unit 2: hook system (loaded from config) + permission gate.
    # Both are optional in the constructors so other tests/setups still work.
    hooks = load_hooks_from_config(config.hooks or {})
    plan_mode.set_hook_system(hooks)
    permission_gate = PermissionGate(
        registry=registry, ctx=ctx, hooks=hooks,
        yolo_mode=config.permissions.yolo_mode,
    )
    executor = StreamingExecutor(registry, ctx, permission_gate=permission_gate)

    # Phase 2 Unit 1: assemble context with the 5-zone assembler + 5-layer
    # compression. Aux provider is optional; if missing, layers 3-4 short-circuit
    # via the breaker. Memory store is injected so Zone 3 can surface the
    # MEMORY.md index to the model automatically.
    aux_provider = resolve_auxiliary(config)
    compression = CompressionEngine(auxiliary_provider=aux_provider, hook_system=hooks)
    assembler = ContextAssembler(
        system_prompt=_SYSTEM_PROMPT,
        plan_mode=plan_mode,
        compression_engine=compression,
        memory_store=memory_store,
        output_reserve=config.context.output_reserve,
        safety_buffer=config.context.buffer,
    )

    # Phase 2b Unit 4: capability router. Reuses the same auxiliary LLM as
    # compression. With no aux provider, defaults to "direct" (pass-through).
    capability_router = CapabilityRouter(aux_provider=aux_provider)

    # Make memory store available to memory tools via ToolContext.
    ctx.memory_store = memory_store

    # Phase 2b Unit 3: bring up configured MCP servers. Each server's tools
    # are registered into the shared registry with `${server}__${tool}`
    # names. Connection failures are logged so one bad server doesn't kill
    # the whole session.
    mcp_clients: list[MCPClient] = []
    for mcp_cfg in config.mcp_servers:
        client = MCPClient(
            server_name=mcp_cfg.name,
            transport=mcp_cfg.transport,
            command=mcp_cfg.command,
            url=mcp_cfg.url,
            env=mcp_cfg.env,
            headers=mcp_cfg.headers,
            default_permission=mcp_cfg.default_permission,
        )
        try:
            await client.connect(registry)
            mcp_clients.append(client)
        except Exception as exc:
            logger.warning("Failed to connect MCP server %r: %s", mcp_cfg.name, exc)

    try:
        if args.smoke_test:
            await _smoke_test(config, provider, registry, budget, plan_mode,
                             env_injector, executor, ctx, assembler, hooks,
                             scenario_router=scenario_router,
                             capability_router=capability_router)
        else:
            await _run_conversation(config, provider, registry, budget, plan_mode,
                                   env_injector, executor, ctx, assembler, hooks,
                                   memory_store, scenario_router=scenario_router,
                                   capability_router=capability_router)
    finally:
        for client in mcp_clients:
            try:
                await client.close()
            except Exception as exc:
                logger.debug("Error closing MCP client: %s", exc)


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
