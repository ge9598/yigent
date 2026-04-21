import pytest
from unittest.mock import MagicMock, AsyncMock
from pathlib import Path
from collections.abc import AsyncGenerator

from src.core.agent_loop import agent_loop
from src.core.types import (
    Message, StreamChunk, ToolCall, ToolSchema, PermissionLevel,
    ToolDefinition, ToolContext, PermissionDecision,
    TokenEvent, FinalAnswerEvent, ToolCallStartEvent, ToolResultEvent,
    BudgetExhaustedEvent,
)
from src.core.iteration_budget import IterationBudget
from src.core.plan_mode import PlanMode
from src.core.env_injector import EnvironmentInjector
from src.core.streaming_executor import StreamingExecutor
from src.core.config import load_config
from src.tools.registry import ToolRegistry


def _text_provider(text: str):
    """Mock provider: yields text tokens then done."""
    provider = MagicMock()
    async def stream_message(**kwargs) -> AsyncGenerator:
        for word in text.split():
            yield StreamChunk(type="token", data=word + " ")
        yield StreamChunk(type="done", data="stop")
    provider.stream_message = stream_message
    return provider


def _tool_then_text_provider(tool_name: str, args: dict, answer: str):
    """Mock provider: first call yields tool call, second call yields text."""
    provider = MagicMock()
    call_count = {"n": 0}
    async def stream_message(**kwargs) -> AsyncGenerator:
        call_count["n"] += 1
        if call_count["n"] == 1:
            yield StreamChunk(type="tool_call_start", data={"id": "c1", "name": tool_name})
            yield StreamChunk(type="tool_call_complete",
                            data=ToolCall(id="c1", name=tool_name, arguments=args))
            yield StreamChunk(type="done", data="tool_calls")
        else:
            for w in answer.split():
                yield StreamChunk(type="token", data=w + " ")
            yield StreamChunk(type="done", data="stop")
    provider.stream_message = stream_message
    return provider


def _make_registry() -> ToolRegistry:
    reg = ToolRegistry()
    async def greet(name: str) -> str:
        return f"Hello, {name}!"
    reg.register(ToolDefinition(
        name="greet", description="Greet someone", handler=greet,
        schema=ToolSchema(
            name="greet", description="Greet someone",
            parameters={"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
            permission_level=PermissionLevel.READ_ONLY,
        ),
    ))
    return reg


def _make_deps(reg=None, provider=None, budget_total=10):
    reg = reg or _make_registry()
    config = load_config()
    pm = PlanMode()
    ctx = ToolContext(plan_mode=pm, registry=reg, config=config, working_dir=Path.cwd())
    executor = StreamingExecutor(reg, ctx)
    return {
        "tools": reg,
        "budget": IterationBudget(budget_total),
        "provider": provider or _text_provider("Hello"),
        "executor": executor,
        "env_injector": EnvironmentInjector(),
        "plan_mode": pm,
        "config": config,
    }


@pytest.mark.asyncio
async def test_text_only_response():
    deps = _make_deps(provider=_text_provider("Hello world"))
    conversation = [Message(role="user", content="Hi")]
    events = []
    async for event in agent_loop(conversation=conversation, **deps):
        events.append(event)
    types = [type(e).__name__ for e in events]
    assert "TokenEvent" in types
    assert "FinalAnswerEvent" in types


@pytest.mark.asyncio
async def test_tool_call_then_response():
    reg = _make_registry()
    deps = _make_deps(reg=reg, provider=_tool_then_text_provider("greet", {"name": "World"}, "Done"))
    conversation = [Message(role="user", content="Greet World")]
    events = []
    async for event in agent_loop(conversation=conversation, **deps):
        events.append(event)
    types = [type(e).__name__ for e in events]
    assert "ToolCallStartEvent" in types
    assert "ToolResultEvent" in types
    assert "FinalAnswerEvent" in types
    tool_results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert "Hello, World!" in tool_results[0].result.content


@pytest.mark.asyncio
async def test_budget_exhaustion():
    deps = _make_deps(provider=_tool_then_text_provider("greet", {"name": "X"}, "ok"), budget_total=1)
    # With budget=1, after first tool call round budget.consume(1) exhausts it
    # The loop should yield BudgetExhaustedEvent instead of continuing
    conversation = [Message(role="user", content="test")]
    events = []
    async for event in agent_loop(conversation=conversation, **deps):
        events.append(event)
    types = [type(e).__name__ for e in events]
    assert "BudgetExhaustedEvent" in types


@pytest.mark.asyncio
async def test_conversation_grows():
    deps = _make_deps(provider=_text_provider("Hello"))
    conversation = [Message(role="user", content="Hi")]
    async for _ in agent_loop(conversation=conversation, **deps):
        pass
    # After loop, conversation should have user + assistant messages
    assert len(conversation) == 2
    assert conversation[1]["role"] == "assistant"


# ---------------------------------------------------------------------------
# Unit 5 — stop_reason inspection (TruncatedEvent on max_tokens) + provider
# fallback chain on stream failure.
# ---------------------------------------------------------------------------

def _truncating_provider(partial_text: str):
    """Mock provider that streams partial text then says it hit max_tokens."""
    provider = MagicMock()

    async def stream_message(**kwargs) -> AsyncGenerator:
        for word in partial_text.split():
            yield StreamChunk(type="token", data=word + " ")
        # The harness's finish_reason vocabulary is OpenAI-style: "length"
        # means the model truncated due to max_tokens.
        yield StreamChunk(type="done", data="length")
    provider.stream_message = stream_message
    return provider


def _failing_provider():
    """Mock provider whose stream raises immediately."""
    provider = MagicMock()

    async def stream_message(**kwargs) -> AsyncGenerator:
        if False:
            yield  # make this a generator
        raise RuntimeError("primary down")
    provider.stream_message = stream_message
    return provider


@pytest.mark.asyncio
async def test_truncated_event_on_max_tokens():
    """When the provider reports finish_reason=length, the loop must yield a
    TruncatedEvent so the UI can warn the user — not silently commit the
    partial text as the final answer."""
    from src.core.types import TruncatedEvent
    deps = _make_deps(provider=_truncating_provider("partial answer here"))
    conversation = [Message(role="user", content="give me a long answer")]
    events = []
    async for event in agent_loop(conversation=conversation, **deps):
        events.append(event)
    types = [type(e).__name__ for e in events]
    assert "TruncatedEvent" in types
    truncated = [e for e in events if isinstance(e, TruncatedEvent)]
    assert truncated[0].finish_reason == "length"
    assert "partial" in truncated[0].content


@pytest.mark.asyncio
async def test_provider_fallback_on_stream_exception(monkeypatch):
    """When the primary provider fails, the loop should retry once with the
    configured fallback provider and yield a ProviderFallbackEvent."""
    from src.core.config import AgentConfig, ProviderConfig, ProviderSection
    from src.core.types import ProviderFallbackEvent

    # Build a config with an explicit fallback. The fallback's name doesn't
    # matter for the test because we monkeypatch _build_provider to return
    # our mock fallback regardless of inputs.
    cfg = AgentConfig(
        provider=ProviderSection(
            name="openai_compat",
            api_key="primary",
            base_url="https://primary.test/v1",
            model="primary-model",
            fallback=ProviderConfig(
                name="anthropic_compat",
                api_key="fallback",
                base_url="https://fallback.test",
                model="fallback-model",
            ),
        )
    )

    fallback_called = {"count": 0}

    def _fb_provider():
        provider = MagicMock()

        async def stream_message(**kwargs):
            fallback_called["count"] += 1
            for w in "I am the fallback".split():
                yield StreamChunk(type="token", data=w + " ")
            yield StreamChunk(type="done", data="stop")
        provider.stream_message = stream_message
        return provider

    fb_instance = _fb_provider()

    def fake_build_provider(**kwargs):
        return fb_instance

    import src.providers.resolver as resolver_mod
    monkeypatch.setattr(resolver_mod, "_build_provider", fake_build_provider)

    reg = _make_registry()
    pm = PlanMode()
    ctx = ToolContext(plan_mode=pm, registry=reg, config=cfg, working_dir=Path.cwd())
    executor = StreamingExecutor(reg, ctx)
    deps = {
        "tools": reg,
        "budget": IterationBudget(10),
        "provider": _failing_provider(),
        "executor": executor,
        "env_injector": EnvironmentInjector(),
        "plan_mode": pm,
        "config": cfg,
    }

    conversation = [Message(role="user", content="hi")]
    events = []
    async for event in agent_loop(conversation=conversation, **deps):
        events.append(event)

    types = [type(e).__name__ for e in events]
    assert "ProviderFallbackEvent" in types
    assert "FinalAnswerEvent" in types
    fb_event = next(e for e in events if isinstance(e, ProviderFallbackEvent))
    assert "primary down" in fb_event.reason
    assert fallback_called["count"] == 1
    final = [e for e in events if type(e).__name__ == "FinalAnswerEvent"][0]
    assert "fallback" in final.content


# ---------------------------------------------------------------------------
# Unit 6 — interruption tombstone repair
# ---------------------------------------------------------------------------

def _slow_tool_provider():
    """Provider that produces one tool call that we'll interrupt."""
    provider = MagicMock()

    async def stream_message(**kwargs):
        yield StreamChunk(type="tool_call_start", data={"id": "c1", "name": "slow"})
        yield StreamChunk(type="tool_call_complete",
                          data=ToolCall(id="c1", name="slow", arguments={"text": "x"}))
        yield StreamChunk(type="done", data="tool_calls")
    provider.stream_message = stream_message
    return provider


@pytest.mark.asyncio
async def test_cancelled_during_executor_appends_tool_result_tombstones():
    """If cancellation hits while the executor is running, every emitted
    tool_use must end up with a matching tool_result so the Anthropic
    protocol stays valid on retry."""
    import asyncio
    from src.core.types import (
        ToolDefinition, ToolSchema, PermissionLevel,
    )

    reg = ToolRegistry()

    async def slow_handler(text: str) -> str:
        await asyncio.sleep(5)
        return "should never appear"

    reg.register(ToolDefinition(
        name="slow", description="slow", handler=slow_handler,
        schema=ToolSchema(
            name="slow", description="slow",
            parameters={"type": "object", "properties": {"text": {"type": "string"}}},
            permission_level=PermissionLevel.READ_ONLY, timeout=10,
        ),
    ))
    cfg = load_config()
    cfg.provider.fallback = None  # no fallback so primary failure surfaces directly
    pm = PlanMode()
    ctx = ToolContext(plan_mode=pm, registry=reg, config=cfg, working_dir=Path.cwd())
    executor = StreamingExecutor(reg, ctx)
    deps = {
        "tools": reg,
        "budget": IterationBudget(10),
        "provider": _slow_tool_provider(),
        "executor": executor,
        "env_injector": EnvironmentInjector(),
        "plan_mode": pm,
        "config": cfg,
    }
    conversation = [Message(role="user", content="run slow")]

    async def _drive():
        async for _ in agent_loop(conversation=conversation, **deps):
            pass

    task = asyncio.create_task(_drive())
    # Let it begin executing the slow tool, then cancel.
    await asyncio.sleep(0.1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # The conversation should now contain the assistant message with tool_calls
    # AND a synthetic tool_result for c1 — never an orphan tool_use.
    tool_use_ids = set()
    tool_result_ids = set()
    for msg in conversation:
        if msg.get("role") == "assistant":
            for tc in msg.get("tool_calls") or []:
                tool_use_ids.add(tc["id"])
        if msg.get("role") == "tool":
            tool_result_ids.add(msg.get("tool_call_id"))
    assert tool_use_ids == {"c1"}
    assert "c1" in tool_result_ids


@pytest.mark.asyncio
async def test_cancelled_before_assistant_message_drops_orphan_tool_use():
    """If cancellation hits before the assistant message is appended, the
    conversation must NOT contain dangling tool_use entries."""
    import asyncio
    from src.core.types import (
        ToolDefinition, ToolSchema, PermissionLevel,
    )

    reg = ToolRegistry()

    async def echo(text: str) -> str:
        return text

    reg.register(ToolDefinition(
        name="echo", description="echo", handler=echo,
        schema=ToolSchema(
            name="echo", description="echo",
            parameters={"type": "object", "properties": {"text": {"type": "string"}}},
            permission_level=PermissionLevel.READ_ONLY, timeout=2,
        ),
    ))

    # Build a provider whose stream blocks indefinitely after emitting
    # tool_call_start but before tool_call_complete — simulates Ctrl+C
    # mid-stream while the model was still emitting.
    async def stream_message(**kwargs):
        yield StreamChunk(type="tool_call_start", data={"id": "orphan", "name": "echo"})
        # Block forever — caller will cancel.
        await asyncio.sleep(60)
        yield StreamChunk(type="done", data="tool_calls")
    provider = MagicMock()
    provider.stream_message = stream_message

    cfg = load_config()
    cfg.provider.fallback = None
    pm = PlanMode()
    ctx = ToolContext(plan_mode=pm, registry=reg, config=cfg, working_dir=Path.cwd())
    executor = StreamingExecutor(reg, ctx)
    deps = {
        "tools": reg,
        "budget": IterationBudget(10),
        "provider": provider,
        "executor": executor,
        "env_injector": EnvironmentInjector(),
        "plan_mode": pm,
        "config": cfg,
    }
    conversation = [Message(role="user", content="run echo")]

    async def _drive():
        async for _ in agent_loop(conversation=conversation, **deps):
            pass

    task = asyncio.create_task(_drive())
    await asyncio.sleep(0.1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # Conversation should contain the user message; possibly nothing else
    # (no assistant message because stream never finished). Critically: NO
    # role:assistant message with tool_calls referencing 'orphan'.
    for msg in conversation:
        assert msg.get("role") != "tool", "Tombstone leaked into conversation without assistant"
        for tc in msg.get("tool_calls") or []:
            assert tc["id"] != "orphan", "Orphan tool_use must not be persisted"


@pytest.mark.asyncio
async def test_no_fallback_configured_emits_error():
    """Without a fallback provider, primary failure must surface as ErrorEvent."""
    from src.core.types import ErrorEvent
    deps = _make_deps(provider=_failing_provider())
    # default.yaml ships with a fallback openai_compat entry — for this test
    # we want the "no fallback" path, so explicitly clear it on the loaded config.
    deps["config"].provider.fallback = None
    conversation = [Message(role="user", content="hi")]
    events = []
    async for event in agent_loop(conversation=conversation, **deps):
        events.append(event)
    types = [type(e).__name__ for e in events]
    assert "ErrorEvent" in types
    err = next(e for e in events if isinstance(e, ErrorEvent))
    assert "primary down" in err.error
