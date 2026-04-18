"""Central async generator implementing the ReAct cycle."""
from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Callable, Awaitable

from src.core.iteration_budget import BudgetExhausted
from src.core.types import (
    BudgetExhaustedEvent, ErrorEvent, Event, FinalAnswerEvent, Message,
    PermissionDecision, StreamChunk, ToolCall, ToolCallStartEvent,
    ToolResultEvent, TokenEvent,
)

if TYPE_CHECKING:
    from src.core.config import AgentConfig
    from src.core.env_injector import EnvironmentInjector
    from src.core.iteration_budget import IterationBudget
    from src.core.plan_mode import PlanMode
    from src.core.streaming_executor import StreamingExecutor
    from src.providers.base import LLMProvider
    from src.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

PermissionCallback = Callable[[ToolCall], Awaitable[PermissionDecision]]

_SYSTEM_PROMPT = (
    "You are a general-purpose AI agent. You can use tools to accomplish tasks.\n"
    "Think step by step before acting.\n"
    "When you need to discover additional tools, use the tool_search function.\n"
    "During plan mode, only read-only tools are available."
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
) -> AsyncGenerator[Event, None]:
    """Async generator ReAct loop. Yields events to the UI layer."""

    async def _default_permission(tc: ToolCall) -> PermissionDecision:
        return PermissionDecision.ALLOW

    perm_cb = permission_callback or _default_permission

    while True:
        # 1. Check budget
        if budget.is_exhausted:
            yield BudgetExhaustedEvent(remaining=0, total=budget.total)
            return

        # 2. Build messages
        last_user_text = ""
        for msg in reversed(conversation):
            if msg.get("role") == "user" and msg.get("content"):
                last_user_text = msg["content"]
                break

        task_type = env_injector.detect_task_type(last_user_text)
        env_text = await env_injector.get_context(task_type)
        messages = _assemble_messages(conversation, tools, env_text, plan_mode)

        # 3. Stream LLM response
        active_schemas = tools.get_active_schemas()
        text_buffer = ""
        tool_calls: list[ToolCall] = []

        try:
            async for chunk in provider.stream_message(
                messages=messages,
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
            return

        # 4. No tool calls → final answer
        if not tool_calls:
            conversation.append(Message(role="assistant", content=text_buffer))
            # If in plan mode, buffer the response as plan content
            if plan_mode.is_active and text_buffer.strip():
                plan_mode.append(text_buffer)
            yield FinalAnswerEvent(content=text_buffer)
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
            return

        if budget.is_warning:
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

    messages.append(Message(role="system", content=sys_content))

    # Environment context
    if env_text:
        messages.append(Message(role="system", content=f"[Environment]\n{env_text}"))

    # Conversation history
    messages.extend(conversation)

    return messages
