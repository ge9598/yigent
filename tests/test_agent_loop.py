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
