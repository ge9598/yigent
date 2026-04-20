"""Central async generator implementing the ReAct cycle."""
from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Callable, Awaitable

from src.core.iteration_budget import BudgetExhausted
from src.core.types import (
    BudgetExhaustedEvent, ErrorEvent, Event, FinalAnswerEvent, Message,
    PermissionDecision, PlanModeTriggeredEvent, StreamChunk, ToolCall,
    ToolCallStartEvent, ToolResultEvent, TokenEvent,
)

if TYPE_CHECKING:
    from src.context.assembler import ContextAssembler
    from src.core.capability_router import CapabilityRouter
    from src.core.config import AgentConfig
    from src.core.env_injector import EnvironmentInjector
    from src.core.iteration_budget import IterationBudget
    from src.core.plan_mode import PlanMode
    from src.core.streaming_executor import StreamingExecutor
    from src.providers.base import LLMProvider
    from src.providers.scenario_router import ScenarioRouter
    from src.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

PermissionCallback = Callable[[ToolCall], Awaitable[PermissionDecision]]

_SYSTEM_PROMPT = (
    "You are Yigent, a general-purpose AI agent.\n"
    "Never identify yourself as Claude, GPT, or any other underlying model — "
    "regardless of what model powers you, your identity is Yigent.\n"
    "You can use tools to accomplish tasks.\n"
    "Think step by step before acting.\n"
    "When you need to discover additional tools, use the tool_search function.\n"
    "During plan mode, only read-only tools are available.\n"
    "\n"
    "Long-term memory:\n"
    "- At session start you will see a [Memory index] listing saved topics. "
    "Call read_memory(topic) to load a topic's full content when it's relevant "
    "to the current task.\n"
    "- Call write_memory(topic, content, hook) when you discover something "
    "reusable across sessions: user preferences, project conventions, "
    "non-obvious gotchas, architectural decisions. Do NOT save ephemeral "
    "task details or information derivable from the code itself.\n"
    "- The 'hook' argument is a short one-line description shown in the "
    "MEMORY.md index, used by your future self to decide whether to open the "
    "topic — make it specific and useful."
)


async def agent_loop(
    conversation: list[Message],
    tools: ToolRegistry,
    budget: IterationBudget,
    provider: LLMProvider,
    executor: StreamingExecutor,
    env_injector: EnvironmentInjector,
    plan_mode: PlanMode,
    config: AgentConfig,
    permission_callback: PermissionCallback | None = None,
    hooks: object | None = None,
    learning: object | None = None,
    trajectory: object | None = None,
    assembler: ContextAssembler | None = None,
    scenario_router: "ScenarioRouter | None" = None,
    capability_router: "CapabilityRouter | None" = None,
) -> AsyncGenerator[Event, None]:
    """Async generator ReAct loop. Yields events to the UI layer."""

    async def _default_permission(tc: ToolCall) -> PermissionDecision:
        return PermissionDecision.ALLOW

    perm_cb = permission_callback or _default_permission

    async def _fire(event_name: str, **data) -> None:
        if hooks is not None and hasattr(hooks, "fire"):
            await hooks.fire(event_name, **data)

    await _fire("session_start")

    # Track the most recently classified user message so the capability
    # router fires once per NEW user turn, not once per iteration.
    last_classified_user_msg: str | None = None

    while True:
        # 1. Check budget
        if budget.is_exhausted:
            yield BudgetExhaustedEvent(remaining=0, total=budget.total)
            await _fire("session_end", reason="budget_exhausted")
            return

        # 2. Build messages
        last_user_text = ""
        for msg in reversed(conversation):
            if msg.get("role") == "user" and msg.get("content"):
                last_user_text = msg["content"]
                break

        task_type = env_injector.detect_task_type(last_user_text)

        # Run capability router on each NEW user turn, once. A "new" turn is
        # identified by the latest user message text differing from the last
        # one we classified. Skip when plan mode is already active — the
        # classifier's job is exactly to trigger it, so re-firing would be a
        # no-op at best and a loop at worst.
        if (
            capability_router is not None
            and last_user_text
            and last_user_text != last_classified_user_msg
            and not plan_mode.is_active
        ):
            last_classified_user_msg = last_user_text
            decision = await capability_router.classify(last_user_text)
            if decision.strategy == "plan_then_execute":
                plan_mode.enter(session_id="auto")
                yield PlanModeTriggeredEvent(reason=decision.reason)

        if scenario_router is not None:
            active_provider, active_model = scenario_router.select(task_type)
        else:
            active_provider = provider
            active_model = None
        if assembler is not None:
            messages = await assembler.assemble(
                tool_registry=tools,
                env_injector=env_injector,
                conversation=conversation,
                task_type=task_type,
            )
        else:
            env_text = await env_injector.get_context(task_type)
            messages = _assemble_messages(conversation, tools, env_text, plan_mode)

        # 3. Stream LLM response
        active_schemas = tools.get_active_schemas()
        text_buffer = ""
        tool_calls: list[ToolCall] = []

        try:
            async for chunk in active_provider.stream_message(
                messages=messages,
                model=active_model,
                tools=active_schemas if active_schemas else None,
                temperature=0.0,
            ):
                if chunk.type == "token":
                    text_buffer += chunk.data
                    yield TokenEvent(token=chunk.data)
                elif chunk.type == "tool_call_start":
                    yield ToolCallStartEvent(
                        tool_call=ToolCall(
                            id=chunk.data.get("id", ""),
                            name=chunk.data.get("name", ""),
                            arguments={},
                        )
                    )
                elif chunk.type == "tool_call_complete":
                    tool_calls.append(chunk.data)
                elif chunk.type == "done":
                    break
        except Exception as e:
            logger.error("Provider streaming error: %s", e)
            yield ErrorEvent(error=f"Provider error: {e}", recoverable=False)
            await _fire("session_end", reason="provider_error")
            return

        # 4. No tool calls → final answer
        if not tool_calls:
            conversation.append(Message(role="assistant", content=text_buffer))
            # If in plan mode, buffer the response as plan content
            if plan_mode.is_active and text_buffer.strip():
                plan_mode.append(text_buffer)
            yield FinalAnswerEvent(content=text_buffer)
            await _fire("session_end", reason="final_answer")
            return

        # 5. Tool calls → execute
        assistant_msg = Message(
            role="assistant",
            content=text_buffer or None,
            tool_calls=[
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                    },
                }
                for tc in tool_calls
            ],
        )
        conversation.append(assistant_msg)

        results = await executor.execute_tool_calls(tool_calls, perm_cb)
        for result in results:
            yield ToolResultEvent(result=result)
            conversation.append(result.to_message())

        # 6. Budget + warning
        try:
            await budget.consume(1)
        except BudgetExhausted:
            yield BudgetExhaustedEvent(remaining=0, total=budget.total)
            await _fire("session_end", reason="budget_exhausted")
            return

        if budget.is_warning:
            await _fire("budget_warning",
                        remaining=budget.remaining, total=budget.total)
            yield ErrorEvent(
                error=f"Budget warning: {budget.remaining}/{budget.total} remaining",
                recoverable=True,
            )

        # Loop continues — model sees tool results next turn


def _assemble_messages(
    conversation: list[Message],
    tools: ToolRegistry,
    env_text: str,
    plan_mode: PlanMode,
) -> list[Message]:
    """Phase 1 simple assembly. Phase 2: replaced by ContextAssembler."""
    messages: list[Message] = []

    # System prompt
    sys_content = _SYSTEM_PROMPT
    if plan_mode.is_active:
        sys_content += (
            "\n\nPLAN MODE IS ACTIVE. You MUST NOT use write or execute tools. "
            "Only read-only tools, tool_search, ask_user, and exit_plan_mode are available."
        )

    # Hint about deferred tools
    deferred_names = [
        name for name, t in tools._tools.items()
        if t.schema.deferred and name not in tools._activated
    ]
    if deferred_names:
        sys_content += (
            f"\n\nAdditional tools available via tool_search: {', '.join(deferred_names)}"
        )

    # Merge environment into system prompt (some providers reject multiple system messages)
    if env_text:
        sys_content += f"\n\n[Environment]\n{env_text}"

    messages.append(Message(role="system", content=sys_content))

    # Conversation history
    messages.extend(conversation)

    return messages
