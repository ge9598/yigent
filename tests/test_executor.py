import pytest
import asyncio
from unittest.mock import AsyncMock
from pathlib import Path

from src.core.streaming_executor import StreamingExecutor
from src.core.types import (
    ToolCall, ToolResult, ToolContext, PermissionDecision, PermissionLevel,
    ToolDefinition, ToolSchema,
)
from src.core.plan_mode import PlanMode
from src.core.config import load_config
from src.tools.registry import ToolRegistry


def _make_registry() -> ToolRegistry:
    reg = ToolRegistry()

    async def echo_handler(text: str) -> str:
        return f"echo: {text}"

    async def slow_handler(text: str) -> str:
        await asyncio.sleep(5)
        return "slow done"

    async def fail_handler(text: str) -> str:
        raise RuntimeError("tool exploded")

    for name, handler, perm in [
        ("echo", echo_handler, PermissionLevel.READ_ONLY),
        ("slow", slow_handler, PermissionLevel.READ_ONLY),
        ("fail", fail_handler, PermissionLevel.READ_ONLY),
        ("writer", echo_handler, PermissionLevel.WRITE),
    ]:
        reg.register(ToolDefinition(
            name=name, description=name, handler=handler,
            schema=ToolSchema(
                name=name, description=name,
                parameters={"type": "object", "properties": {"text": {"type": "string"}}},
                permission_level=perm, timeout=2,
            ),
        ))
    return reg


def _make_ctx(reg, pm=None):
    return ToolContext(
        plan_mode=pm or PlanMode(),
        registry=reg, config=load_config(),
        working_dir=Path.cwd(),
    )


class TestStreamingExecutor:
    def setup_method(self):
        self.reg = _make_registry()
        self.ctx = _make_ctx(self.reg)
        self.executor = StreamingExecutor(self.reg, self.ctx)

    @pytest.mark.asyncio
    async def test_single_tool(self):
        tc = ToolCall(id="1", name="echo", arguments={"text": "hi"})
        results = await self.executor.execute_tool_calls(
            [tc], permission_callback=AsyncMock(return_value=PermissionDecision.ALLOW))
        assert len(results) == 1
        assert results[0].content == "echo: hi"
        assert not results[0].is_error

    @pytest.mark.asyncio
    async def test_parallel_execution(self):
        calls = [
            ToolCall(id="1", name="echo", arguments={"text": "a"}),
            ToolCall(id="2", name="echo", arguments={"text": "b"}),
        ]
        results = await self.executor.execute_tool_calls(
            calls, permission_callback=AsyncMock(return_value=PermissionDecision.ALLOW))
        assert len(results) == 2
        assert {r.content for r in results} == {"echo: a", "echo: b"}

    @pytest.mark.asyncio
    async def test_timeout_error_result(self):
        tc = ToolCall(id="1", name="slow", arguments={"text": "x"})
        results = await self.executor.execute_tool_calls(
            [tc], permission_callback=AsyncMock(return_value=PermissionDecision.ALLOW))
        assert results[0].is_error
        assert "timed out" in results[0].content.lower()

    @pytest.mark.asyncio
    async def test_exception_error_result(self):
        tc = ToolCall(id="1", name="fail", arguments={"text": "x"})
        results = await self.executor.execute_tool_calls(
            [tc], permission_callback=AsyncMock(return_value=PermissionDecision.ALLOW))
        assert results[0].is_error
        assert "exploded" in results[0].content

    @pytest.mark.asyncio
    async def test_read_tool_auto_allowed(self):
        tc = ToolCall(id="1", name="echo", arguments={"text": "x"})
        cb = AsyncMock(return_value=PermissionDecision.ALLOW)
        await self.executor.execute_tool_calls([tc], permission_callback=cb)
        cb.assert_not_called()

    @pytest.mark.asyncio
    async def test_write_tool_asks_permission(self):
        tc = ToolCall(id="1", name="writer", arguments={"text": "x"})
        cb = AsyncMock(return_value=PermissionDecision.ALLOW)
        await self.executor.execute_tool_calls([tc], permission_callback=cb)
        cb.assert_called_once()

    @pytest.mark.asyncio
    async def test_write_blocked_in_plan_mode(self):
        pm = PlanMode()
        pm.enter("s1")
        ctx = _make_ctx(self.reg, pm)
        executor = StreamingExecutor(self.reg, ctx)
        tc = ToolCall(id="1", name="writer", arguments={"text": "x"})
        results = await executor.execute_tool_calls(
            [tc], permission_callback=AsyncMock(return_value=PermissionDecision.ALLOW))
        assert results[0].is_error
        assert "plan mode" in results[0].content.lower()

    @pytest.mark.asyncio
    async def test_unknown_tool_blocked(self):
        tc = ToolCall(id="1", name="nonexistent", arguments={})
        results = await self.executor.execute_tool_calls(
            [tc], permission_callback=AsyncMock(return_value=PermissionDecision.ALLOW))
        assert results[0].is_error
        assert "unknown" in results[0].content.lower()
