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


class TestStreamingDispatch:
    """Unit 7 — dispatch() starts execution immediately so the agent loop can
    overlap tool work with continued model streaming."""

    @pytest.mark.asyncio
    async def test_dispatch_returns_task_immediately(self):
        """dispatch() must return before the tool finishes — that's the point."""
        from src.core.types import ToolDefinition, ToolSchema
        reg = ToolRegistry()

        async def slow(text: str) -> str:
            await asyncio.sleep(0.3)
            return text

        reg.register(ToolDefinition(
            name="slow", description="slow", handler=slow,
            schema=ToolSchema(
                name="slow", description="slow",
                parameters={"type": "object", "properties": {"text": {"type": "string"}}},
                permission_level=PermissionLevel.READ_ONLY, timeout=5,
            ),
        ))
        ctx = _make_ctx(reg)
        executor = StreamingExecutor(reg, ctx)

        tc = ToolCall(id="1", name="slow", arguments={"text": "x"})
        t0 = asyncio.get_running_loop().time()
        task = await executor.dispatch(
            tc, AsyncMock(return_value=PermissionDecision.ALLOW))
        t1 = asyncio.get_running_loop().time()
        # Dispatch returned without waiting for the tool to finish.
        assert (t1 - t0) < 0.1
        assert not task.done()
        # Now wait for it.
        results = await executor.collect({"1": task}, [tc])
        assert results[0].content == "x"

    @pytest.mark.asyncio
    async def test_collect_preserves_input_order(self):
        from src.core.types import ToolDefinition, ToolSchema
        reg = ToolRegistry()

        async def fast(text: str) -> str:
            return f"fast: {text}"

        async def slow(text: str) -> str:
            await asyncio.sleep(0.2)
            return f"slow: {text}"

        for n, h in [("fast", fast), ("slow", slow)]:
            reg.register(ToolDefinition(
                name=n, description=n, handler=h,
                schema=ToolSchema(
                    name=n, description=n,
                    parameters={"type": "object", "properties": {"text": {"type": "string"}}},
                    permission_level=PermissionLevel.READ_ONLY, timeout=5,
                ),
            ))
        ctx = _make_ctx(reg)
        executor = StreamingExecutor(reg, ctx)
        tc1 = ToolCall(id="1", name="slow", arguments={"text": "a"})
        tc2 = ToolCall(id="2", name="fast", arguments={"text": "b"})
        t1 = await executor.dispatch(tc1, AsyncMock(return_value=PermissionDecision.ALLOW))
        t2 = await executor.dispatch(tc2, AsyncMock(return_value=PermissionDecision.ALLOW))
        results = await executor.collect({"1": t1, "2": t2}, [tc1, tc2])
        # Even though tc2 finishes first, results come back in input order.
        assert results[0].content == "slow: a"
        assert results[1].content == "fast: b"

    @pytest.mark.asyncio
    async def test_exclusive_tools_serialized_against_each_other(self):
        """Two exclusive tools dispatched concurrently must run serially —
        the second waits for the first to finish."""
        from src.core.types import ToolDefinition, ToolSchema
        reg = ToolRegistry()

        running_concurrently = {"max": 0, "current": 0}

        async def exclusive_op(text: str) -> str:
            running_concurrently["current"] += 1
            running_concurrently["max"] = max(
                running_concurrently["max"], running_concurrently["current"]
            )
            await asyncio.sleep(0.1)
            running_concurrently["current"] -= 1
            return text

        reg.register(ToolDefinition(
            name="excl", description="excl", handler=exclusive_op,
            schema=ToolSchema(
                name="excl", description="excl",
                parameters={"type": "object", "properties": {"text": {"type": "string"}}},
                permission_level=PermissionLevel.READ_ONLY, timeout=5,
                exclusive=True,
            ),
        ))
        ctx = _make_ctx(reg)
        executor = StreamingExecutor(reg, ctx)
        cb = AsyncMock(return_value=PermissionDecision.ALLOW)
        calls = [
            ToolCall(id=str(i), name="excl", arguments={"text": str(i)})
            for i in range(3)
        ]
        tasks = {}
        for tc in calls:
            tasks[tc.id] = await executor.dispatch(tc, cb)
        results = await executor.collect(tasks, calls)
        # Three exclusive calls — at most one ran at a time.
        assert running_concurrently["max"] == 1
        assert [r.content for r in results] == ["0", "1", "2"]

    @pytest.mark.asyncio
    async def test_non_exclusive_tools_run_in_parallel(self):
        """Sanity: regular tools still parallelize via dispatch."""
        from src.core.types import ToolDefinition, ToolSchema
        reg = ToolRegistry()

        running_concurrently = {"max": 0, "current": 0}

        async def parallel_op(text: str) -> str:
            running_concurrently["current"] += 1
            running_concurrently["max"] = max(
                running_concurrently["max"], running_concurrently["current"]
            )
            await asyncio.sleep(0.1)
            running_concurrently["current"] -= 1
            return text

        reg.register(ToolDefinition(
            name="par", description="par", handler=parallel_op,
            schema=ToolSchema(
                name="par", description="par",
                parameters={"type": "object", "properties": {"text": {"type": "string"}}},
                permission_level=PermissionLevel.READ_ONLY, timeout=5,
                # exclusive=False (the default)
            ),
        ))
        ctx = _make_ctx(reg)
        executor = StreamingExecutor(reg, ctx)
        cb = AsyncMock(return_value=PermissionDecision.ALLOW)
        calls = [
            ToolCall(id=str(i), name="par", arguments={"text": str(i)})
            for i in range(3)
        ]
        tasks = {tc.id: await executor.dispatch(tc, cb) for tc in calls}
        await executor.collect(tasks, calls)
        # With 3 truly-parallel sleeps, max concurrency should be 3.
        assert running_concurrently["max"] == 3


class TestSiblingAbort:
    """Unit 6 — fatal tool errors must cancel pending siblings."""

    @pytest.mark.asyncio
    async def test_normal_exception_lets_siblings_finish(self):
        """A regular exception in one tool returns an error stub; siblings
        still execute to completion. (Existing Phase 1 behavior.)"""
        reg = _make_registry()
        ctx = _make_ctx(reg)
        executor = StreamingExecutor(reg, ctx)
        calls = [
            ToolCall(id="1", name="fail", arguments={"text": "x"}),
            ToolCall(id="2", name="echo", arguments={"text": "ok"}),
        ]
        results = await executor.execute_tool_calls(
            calls, permission_callback=AsyncMock(return_value=PermissionDecision.ALLOW))
        result_by_id = {r.tool_call_id: r for r in results}
        assert result_by_id["1"].is_error
        assert not result_by_id["2"].is_error
        assert result_by_id["2"].content == "echo: ok"

    @pytest.mark.asyncio
    async def test_fatal_tool_error_cancels_siblings(self):
        """A FatalToolError in one tool must propagate and cancel pending siblings."""
        from src.core.types import FatalToolError, ToolDefinition, ToolSchema

        reg = ToolRegistry()
        siblings_finished: list[str] = []

        async def fatal_handler(text: str) -> str:
            raise FatalToolError("the world is broken")

        async def slow_sibling(text: str) -> str:
            await asyncio.sleep(0.5)
            siblings_finished.append(text)
            return f"finished: {text}"

        for name, handler in [("fatal", fatal_handler), ("slow", slow_sibling)]:
            reg.register(ToolDefinition(
                name=name, description=name, handler=handler,
                schema=ToolSchema(
                    name=name, description=name,
                    parameters={"type": "object", "properties": {"text": {"type": "string"}}},
                    permission_level=PermissionLevel.READ_ONLY, timeout=2,
                ),
            ))

        ctx = _make_ctx(reg)
        executor = StreamingExecutor(reg, ctx)
        calls = [
            ToolCall(id="1", name="fatal", arguments={"text": "boom"}),
            ToolCall(id="2", name="slow", arguments={"text": "should-be-cancelled"}),
        ]
        with pytest.raises(FatalToolError):
            await executor.execute_tool_calls(
                calls, permission_callback=AsyncMock(return_value=PermissionDecision.ALLOW))
        # The slow sibling never appended because TaskGroup cancelled it.
        assert siblings_finished == []


class TestPostToolUseHook:
    """Unit 1 — executor must fire post_tool_use after each handler returns."""

    def setup_method(self):
        from src.safety.hook_system import HookSystem
        from src.safety.permission_gate import PermissionGate
        self.reg = _make_registry()
        self.ctx = _make_ctx(self.reg)
        self.hooks = HookSystem()
        self.fired: list[dict] = []

        async def _record(**data):
            self.fired.append(data)

        self.hooks.register("post_tool_use", _record)
        self.gate = PermissionGate(registry=self.reg, ctx=self.ctx, hooks=self.hooks)
        self.executor = StreamingExecutor(self.reg, self.ctx, permission_gate=self.gate)

    @pytest.mark.asyncio
    async def test_fires_on_success(self):
        tc = ToolCall(id="1", name="echo", arguments={"text": "hi"})
        await self.executor.execute_tool_calls(
            [tc], permission_callback=AsyncMock(return_value=PermissionDecision.ALLOW))
        assert len(self.fired) == 1
        ev = self.fired[0]
        assert ev["tool_call"].id == "1"
        assert ev["result"].content == "echo: hi"
        assert ev["blocked"] is False
        assert "duration" in ev

    @pytest.mark.asyncio
    async def test_fires_on_error(self):
        tc = ToolCall(id="1", name="fail", arguments={"text": "x"})
        await self.executor.execute_tool_calls(
            [tc], permission_callback=AsyncMock(return_value=PermissionDecision.ALLOW))
        assert len(self.fired) == 1
        assert self.fired[0]["result"].is_error is True
        assert self.fired[0]["blocked"] is False

    @pytest.mark.asyncio
    async def test_fires_on_blocked(self):
        # Plan mode blocks writer
        pm = PlanMode()
        pm.enter("s1")
        ctx = _make_ctx(self.reg, pm)
        from src.safety.hook_system import HookSystem
        from src.safety.permission_gate import PermissionGate
        hooks = HookSystem()
        fired: list[dict] = []

        async def _record(**data):
            fired.append(data)

        hooks.register("post_tool_use", _record)
        gate = PermissionGate(registry=self.reg, ctx=ctx, hooks=hooks)
        executor = StreamingExecutor(self.reg, ctx, permission_gate=gate)
        tc = ToolCall(id="1", name="writer", arguments={"text": "x"})
        await executor.execute_tool_calls(
            [tc], permission_callback=AsyncMock(return_value=PermissionDecision.ALLOW))
        assert len(fired) == 1
        assert fired[0]["blocked"] is True
