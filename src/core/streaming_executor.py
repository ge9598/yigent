"""Parallel tool execution with inline Phase 1 permission gate."""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Callable, Awaitable

from src.core.types import (
    PermissionDecision, PermissionLevel,
    ToolCall, ToolContext, ToolResult,
)

if TYPE_CHECKING:
    from src.safety.permission_gate import PermissionGate
    from src.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

PermissionCallback = Callable[[ToolCall], Awaitable[PermissionDecision]]


class StreamingExecutor:
    def __init__(
        self,
        registry: ToolRegistry,
        ctx: ToolContext,
        permission_gate: PermissionGate | None = None,
    ) -> None:
        self._registry = registry
        self._ctx = ctx
        self._gate = permission_gate

    async def execute_tool_calls(
        self, tool_calls: list[ToolCall],
        permission_callback: PermissionCallback,
    ) -> list[ToolResult]:
        """Execute all tool calls, returning results in same order as input."""
        # Pre-check permissions
        decisions: list[tuple[ToolCall, PermissionDecision]] = []
        for tc in tool_calls:
            decision = await self._check_permission(tc, permission_callback)
            decisions.append((tc, decision))

        # Execute permitted calls in parallel
        results: dict[str, ToolResult] = {}

        async def _run_one(tc: ToolCall, decision: PermissionDecision) -> None:
            if decision == PermissionDecision.BLOCK:
                results[tc.id] = ToolResult(
                    tool_call_id=tc.id, name=tc.name,
                    content=self._block_reason(tc), is_error=True,
                )
                return
            results[tc.id] = await self._execute_single(tc)

        try:
            async with asyncio.TaskGroup() as tg:
                for tc, decision in decisions:
                    tg.create_task(_run_one(tc, decision))
        except* Exception as eg:
            for exc in eg.exceptions:
                logger.error("Unexpected executor error: %s", exc)

        # Return in original order
        return [
            results.get(tc.id, ToolResult(
                tool_call_id=tc.id, name=tc.name,
                content="Error: execution failed", is_error=True,
            ))
            for tc in tool_calls
        ]

    async def _check_permission(
        self, tc: ToolCall, callback: PermissionCallback,
    ) -> PermissionDecision:
        """Delegate to PermissionGate when wired; else use Phase 1 inline gate."""
        if self._gate is not None:
            return await self._gate.check(tc, callback)
        # --- Phase 1 fallback (preserved for tests / minimal setups) ---
        if self._registry.get_handler(tc.name) is None:
            return PermissionDecision.BLOCK
        if not self._ctx.plan_mode.is_tool_allowed(tc.name):
            return PermissionDecision.BLOCK
        defn = self._registry.get_definition(tc.name)
        if defn is None:
            return PermissionDecision.BLOCK
        level = defn.schema.permission_level
        if level == PermissionLevel.READ_ONLY:
            return PermissionDecision.ALLOW
        if level == PermissionLevel.DESTRUCTIVE:
            return PermissionDecision.BLOCK
        return await callback(tc)

    def _block_reason(self, tc: ToolCall) -> str:
        if self._gate is not None and self._gate.last_block_reason:
            return self._gate.last_block_reason
        if self._registry.get_handler(tc.name) is None:
            return f"Error: unknown tool '{tc.name}'"
        if not self._ctx.plan_mode.is_tool_allowed(tc.name):
            return f"Error: '{tc.name}' blocked by plan mode"
        return f"Error: '{tc.name}' blocked by permission policy"

    async def _execute_single(self, tc: ToolCall) -> ToolResult:
        """Execute one tool with timeout + error handling."""
        handler_info = self._registry.get_handler(tc.name)
        if handler_info is None:
            return ToolResult(
                tool_call_id=tc.id, name=tc.name,
                content=f"Error: tool '{tc.name}' not found", is_error=True,
            )
        handler, needs_ctx = handler_info

        # Timeout: config override > schema default > 30s
        defn = self._registry.get_definition(tc.name)
        timeout = 30
        if defn:
            timeout = defn.schema.timeout
        config_timeout = getattr(self._ctx.config.tools.timeouts, tc.name, None)
        if config_timeout is not None:
            timeout = config_timeout

        try:
            if needs_ctx:
                coro = handler(self._ctx, **tc.arguments)
            else:
                coro = handler(**tc.arguments)
            content = await asyncio.wait_for(coro, timeout=timeout)
            return ToolResult(
                tool_call_id=tc.id, name=tc.name,
                content=content, is_error=False,
            )
        except asyncio.TimeoutError:
            return ToolResult(
                tool_call_id=tc.id, name=tc.name,
                content=f"Error: '{tc.name}' timed out after {timeout}s",
                is_error=True,
            )
        except Exception as e:
            logger.warning("Tool %s raised: %s", tc.name, e)
            return ToolResult(
                tool_call_id=tc.id, name=tc.name,
                content=f"Error: {type(e).__name__}: {e}", is_error=True,
            )
