"""Central async generator implementing the ReAct cycle."""
from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any, Callable, Awaitable

from src.core.iteration_budget import BudgetExhausted
from src.core.types import (
    BudgetExhaustedEvent, ErrorEvent, Event, FinalAnswerEvent, Message,
    PermissionDecision, PlanModeTriggeredEvent, ProviderFallbackEvent,
    ReasoningDeltaEvent, StreamChunk, ToolCall, ToolCallStartEvent,
    ToolResultEvent, TokenEvent, TruncatedEvent, TurnStartedEvent,
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


# Maps env_injector task types (coding/data_analysis/file_ops/research) to
# CCR scenario-router route keys (default/background/long_context/thinking).
# Keeps two separate vocabularies (injector picks by keyword heuristic,
# router uses CCR-standard names) in sync. ScenarioRouter.select() already
# falls back to "default" when the key is missing, so this map only needs
# to mention task types that should route away from default.
_TASK_TYPE_TO_ROUTE: dict[str, str] = {
    "coding": "default",
    "data_analysis": "long_context",  # data tasks may load large files
    "file_ops": "default",
    "research": "background",          # low-priority summarization/lookup
}

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
    # Track how many conversation messages the trajectory recorder has
    # already seen, so a user message introduced mid-session (e.g. a
    # multi-turn chat) is attached to exactly one TurnRecord.
    last_recorded_conversation_len: int = 0
    # Cumulative tool calls executed this session; drives the periodic nudge
    # trigger. Bucketed by floor(count / interval) so we fire at exactly
    # each interval crossing, even when a turn executes multiple tools at once.
    nudge_tool_call_count: int = 0
    last_nudge_bucket: int = 0

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

        # Immediately signal the UI that work has started. The capability
        # router and the first provider call can each block for seconds on
        # reasoning models; without this the user sees a blank terminal
        # with no feedback.
        yield TurnStartedEvent()

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
            # Unit 10 — pre-activate tools the classifier predicts we'll need.
            # Avoids a ToolSearch round-trip for the obvious cases.
            for capability in decision.capabilities:
                tools.activate_capability_group(capability)
            if decision.strategy == "plan_then_execute":
                plan_mode.enter(session_id="auto")
                yield PlanModeTriggeredEvent(reason=decision.reason)

        if scenario_router is not None:
            # Translate env_injector task types to CCR route keys. Unmapped
            # task types pass through unchanged; ScenarioRouter.select() then
            # falls back to the "default" route if the key isn't configured.
            route_key = _TASK_TYPE_TO_ROUTE.get(task_type, task_type)
            active_provider, active_model = scenario_router.select(route_key)
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

        # 3. Stream LLM response (with single-turn fallback retry on failure)
        active_schemas = tools.get_active_schemas()

        # Unit 7 — pending dispatched tool tasks, populated mid-stream.
        # Maps tool_call_id → asyncio.Task[ToolResult]. Awaited after the
        # stream's ``done`` chunk via executor.collect().
        pending_dispatched: dict[str, "asyncio.Task[Any]"] = {}

        async def _stream_one(prov: "LLMProvider", model_name: str | None):
            """Drain one stream attempt, returning collected state.

            On tool_call_complete the tool is dispatched IMMEDIATELY (Unit 7
            — real streaming tool execution) so it begins running while the
            model is still emitting other tokens / tool calls. The returned
            state's tool_calls list preserves emission order; the caller
            collects results via ``executor.collect(pending_dispatched, …)``.

            Returns ``(text, tool_calls, reasoning_text, reasoning_details,
            finish_reason, did_yield_tokens)``. The caller decides what to do
            with truncation / failure. Errors propagate so the caller can
            decide whether to fall back.
            """
            text = ""
            tcs: list[ToolCall] = []
            r_text = ""
            r_details: list[dict[str, Any]] | None = None
            finish: str | None = None
            yielded_anything = False
            async for ch in prov.stream_message(
                messages=messages,
                model=model_name,
                tools=active_schemas if active_schemas else None,
                temperature=0.0,
            ):
                if ch.type == "token":
                    text += ch.data
                    yielded_anything = True
                    yield ("event", TokenEvent(token=ch.data))
                elif ch.type == "tool_call_start":
                    yielded_anything = True
                    yield ("event", ToolCallStartEvent(
                        tool_call=ToolCall(
                            id=ch.data.get("id", ""),
                            name=ch.data.get("name", ""),
                            arguments={},
                        )
                    ))
                elif ch.type == "tool_call_complete":
                    tcs.append(ch.data)
                    # Streaming dispatch: kick off execution NOW instead of
                    # waiting for the stream to finish. Subsequent siblings
                    # (and continued model tokens) overlap with this tool
                    # actually doing work.
                    try:
                        task = await executor.dispatch(ch.data, perm_cb)
                        pending_dispatched[ch.data.id] = task
                    except Exception as exc:
                        logger.error(
                            "dispatch failed for %s: %s", ch.data.name, exc,
                        )
                elif ch.type == "reasoning_delta":
                    yield ("event", ReasoningDeltaEvent(fragment=ch.data))
                elif ch.type == "reasoning":
                    data = ch.data or {}
                    r_text = data.get("text", "") or r_text
                    details = data.get("details")
                    if details:
                        r_details = details
                elif ch.type == "done":
                    finish = ch.data if isinstance(ch.data, str) else None
                    break
            yield ("result", (text, tcs, r_text, r_details, finish, yielded_anything))

        async def _drive_with_fallback():
            """Try primary provider; on failure (and if fallback configured),
            retry once with the fallback. Each yielded value is ``("event", ev)``
            for UI events, or ``("result", state)`` for the final state tuple.
            On total failure yields ``("error", exc)``.
            """
            try:
                async for kind, payload in _stream_one(active_provider, active_model):
                    yield kind, payload
                return
            except Exception as exc:
                primary_failure = exc
                logger.warning("Primary provider failed mid-stream: %s", exc)
                # Cancel any tools the primary stream already dispatched —
                # we don't want their results polluting the fallback's run.
                for tid, t in list(pending_dispatched.items()):
                    if not t.done():
                        t.cancel()
                pending_dispatched.clear()

            # Decide whether to fall back. Conditions: a fallback exists AND
            # we haven't already yielded substantial content (don't replay
            # tokens the user already saw).
            from src.providers.resolver import _build_provider, _maybe_build_pool, _single_key
            section = config.provider
            if section.fallback is None:
                yield "error", primary_failure
                return
            try:
                fb = section.fallback
                fb_pool = _maybe_build_pool(fb)
                fallback_provider = _build_provider(
                    name=fb.name or section.name,
                    api_key=_single_key(fb) or _single_key(section),
                    base_url=fb.base_url or section.base_url,
                    model=fb.model or section.model,
                    debug=config.ui.debug,
                    credential_pool=fb_pool,
                )
            except Exception as build_exc:
                logger.error("Could not build fallback provider: %s", build_exc)
                yield "error", primary_failure
                return

            yield "event", ProviderFallbackEvent(
                primary=section.name,
                fallback=fb.name or section.name,
                reason=str(primary_failure),
            )
            try:
                async for kind, payload in _stream_one(
                    fallback_provider, fb.model or active_model,
                ):
                    yield kind, payload
            except Exception as fb_exc:
                logger.error("Fallback provider also failed: %s", fb_exc)
                yield "error", fb_exc

        text_buffer = ""
        tool_calls: list[ToolCall] = []
        reasoning_text = ""
        reasoning_details: list[dict[str, Any]] | None = None
        finish_reason: str | None = None
        stream_failed = False
        # Unit 6 — tombstone bookkeeping. Track tool_use ids the model emitted
        # so that on Ctrl+C / cancellation we can synthesize matching error
        # tool_results. Anthropic-format providers reject the next turn if a
        # tool_use is missing its tool_result.
        emitted_tool_use_ids: list[tuple[str, str]] = []  # (id, name)
        assistant_msg_appended = False
        interrupted = False

        try:
            async for kind, payload in _drive_with_fallback():
                if kind == "event":
                    if isinstance(payload, ToolCallStartEvent):
                        emitted_tool_use_ids.append(
                            (payload.tool_call.id, payload.tool_call.name)
                        )
                    yield payload
                elif kind == "result":
                    text_buffer, tool_calls, reasoning_text, \
                        reasoning_details, finish_reason, _ = payload
                elif kind == "error":
                    logger.error("Provider streaming failed: %s", payload)
                    yield ErrorEvent(
                        error=f"Provider error: {payload}", recoverable=False,
                    )
                    await _fire("session_end", reason="provider_error")
                    stream_failed = True
                    break
            if stream_failed:
                return

            # Inspect finish_reason. "length" means model hit max_tokens — surface
            # a TruncatedEvent so the UI can warn the user instead of silently
            # committing a partial answer as final.
            if finish_reason == "length":
                yield TruncatedEvent(content=text_buffer, finish_reason=finish_reason)

            # 4. No tool calls → final answer
            if not tool_calls:
                final_msg: Message = Message(role="assistant", content=text_buffer)
                if reasoning_text:
                    final_msg["reasoning_text"] = reasoning_text
                if reasoning_details:
                    final_msg["reasoning_details"] = reasoning_details
                conversation.append(final_msg)
                # If in plan mode, buffer the response as plan content
                if plan_mode.is_active and text_buffer.strip():
                    plan_mode.append(text_buffer)
                if trajectory is not None and hasattr(trajectory, "record_turn"):
                    user_msg_for_record = _pick_new_user_msg(
                        conversation, last_recorded_conversation_len,
                    )
                    trajectory.record_turn(
                        assistant_msg=final_msg,
                        user_msg=user_msg_for_record,
                        tool_calls=[],
                        tool_results=[],
                        reasoning_text=reasoning_text or None,
                    )
                    last_recorded_conversation_len = len(conversation)
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
            if reasoning_text:
                assistant_msg["reasoning_text"] = reasoning_text
            if reasoning_details:
                assistant_msg["reasoning_details"] = reasoning_details
            conversation.append(assistant_msg)
            assistant_msg_appended = True

            if trajectory is not None and hasattr(trajectory, "record_turn"):
                user_msg_for_record = _pick_new_user_msg(
                    conversation[:-1], last_recorded_conversation_len,
                )
                trajectory.record_turn(
                    assistant_msg=assistant_msg,
                    user_msg=user_msg_for_record,
                    tool_calls=list(tool_calls),
                    tool_results=[],
                    reasoning_text=reasoning_text or None,
                )

            # If the streaming dispatch already kicked off the tools, just
            # collect them. Otherwise (e.g. provider that doesn't emit
            # tool_call_complete until done), fall back to the batch path.
            if pending_dispatched:
                results = await executor.collect(pending_dispatched, tool_calls)
            else:
                results = await executor.execute_tool_calls(tool_calls, perm_cb)
            for result in results:
                yield ToolResultEvent(result=result)
                conversation.append(result.to_message())
            if trajectory is not None and hasattr(trajectory, "attach_tool_results"):
                trajectory.attach_tool_results(list(results))
            last_recorded_conversation_len = len(conversation)

            # Periodic nudge: count actual tool calls, fire once per
            # interval crossing. Learning container must expose
            # ``nudge`` (NudgeEngine-like) and ``recorder`` (TrajectoryRecorder);
            # anything else silently skips.
            nudge_tool_call_count += len(tool_calls)
            if (
                learning is not None
                and hasattr(learning, "nudge")
                and hasattr(learning, "recorder")
                and config.agent.nudge_interval > 0
                and nudge_tool_call_count // config.agent.nudge_interval
                > last_nudge_bucket
            ):
                last_nudge_bucket = (
                    nudge_tool_call_count // config.agent.nudge_interval
                )
                try:
                    await learning.nudge.maybe_nudge(
                        learning.recorder.turns,
                        session_id=getattr(learning, "session_id", "unknown"),
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Periodic nudge raised: %s", exc)
        except (KeyboardInterrupt, asyncio.CancelledError) as exc:
            interrupted = True
            logger.warning("Agent loop interrupted: %s", type(exc).__name__)
            # Tombstone repair: ensure the conversation stays protocol-valid.
            #
            # Case A — interruption during stream BEFORE the assistant
            # message was appended: any tool_use ids the provider emitted
            # are orphan because no assistant message holds them. Drop them
            # — nothing to repair.
            #
            # Case B — interruption AFTER assistant message was appended
            # (so tool_use blocks are now persisted) but BEFORE all
            # tool_results came back. Append synthetic error tool_results
            # for any ids without a matching result so the next turn can
            # be sent to Anthropic without protocol error.
            if assistant_msg_appended:
                already_resulted = {
                    m.get("tool_call_id")
                    for m in conversation
                    if m.get("role") == "tool"
                }
                for tc in tool_calls:
                    if tc.id not in already_resulted:
                        conversation.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "name": tc.name,
                            "content": "[interrupted by user before tool completed]",
                        })  # type: ignore[arg-type]
            # If reasoning_details was captured mid-stream but the assistant
            # message never landed (Case A), the data simply goes out of
            # scope — Anthropic's extended-thinking protocol forbids
            # re-sending a half-thought without its completion.
            yield ErrorEvent(
                error=f"Interrupted by user ({type(exc).__name__})",
                recoverable=True,
            )
            await _fire("session_end", reason="interrupted")
            # Re-raise so the outer caller (CLI / tests) sees the cancellation
            # and can decide what to do next.
            raise
        finally:
            # Cancel and drain any dispatched tool tasks still pending on any
            # exit path (normal completion, error, interrupt). Without this,
            # tasks leak on:
            #   - primary failed, no fallback configured
            #   - primary failed, fallback build failed
            #   - fallback stream also failed
            #   - Ctrl+C / CancelledError
            # Completed tasks are safe to cancel() (it's a no-op on done tasks).
            if pending_dispatched:
                for _tid, t in list(pending_dispatched.items()):
                    if not t.done():
                        t.cancel()
                try:
                    await asyncio.gather(
                        *pending_dispatched.values(), return_exceptions=True,
                    )
                except Exception:
                    logger.debug("Drain of pending_dispatched tasks raised", exc_info=True)
                pending_dispatched.clear()

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


def _pick_new_user_msg(
    conversation: list[Message],
    last_recorded_len: int,
) -> Message | None:
    """Return the most recent user message introduced AFTER the last recorded
    turn boundary, or None if no new user message has arrived.

    The trajectory recorder attaches each user message to exactly one turn —
    the next turn after the user spoke. Subsequent agentic turns (tool calls
    without new user input) record ``user_msg=None``.
    """
    for msg in conversation[last_recorded_len:]:
        if msg.get("role") == "user" and msg.get("content"):
            return msg
    return None


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
