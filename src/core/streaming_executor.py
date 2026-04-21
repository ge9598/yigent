"""Parallel tool execution with inline Phase 1 permission gate."""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Callable, Awaitable

from src.core.types import (
    FatalToolError, PermissionDecision, PermissionLevel,
    ToolCall, ToolContext, ToolResult,
)

if TYPE_CHECKING:
    from src.safety.permission_gate import PermissionGate
    from src.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

PermissionCallback = Callable[[ToolCall], Awaitable[PermissionDecision]]


class StreamingExecutor:
    """Executes tool calls concurrently. Supports two invocation styles:

    - ``execute_tool_calls(tool_calls, perm_cb)``: batch mode, collect-then-
      execute. Retained for callers that already have the full list
      (tests, multi-agent delegation).
    - ``dispatch(tc, perm_cb)``: streaming mode (Unit 7). Kicks off one tool
      call immediately while the provider stream is still emitting others.
      Returns an ``asyncio.Task[ToolResult]`` the caller awaits later via
      ``collect(pending, original_order)``. Exclusive tools (``schema.exclusive
      = True``) are serialized through an internal lock so e.g. ``ask_user``
      never races with parallel IO.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        ctx: ToolContext,
        permission_gate: PermissionGate | None = None,
    ) -> None:
        self._registry = registry
        self._ctx = ctx
        self._gate = permission_gate
        self._exclusive_lock = asyncio.Lock()

    # -- streaming-mode API (Unit 7) ----------------------------------------

    async def dispatch(
        self,
        tc: ToolCall,
        permission_callback: PermissionCallback,
    ) -> asyncio.Task[ToolResult]:
        """Start executing ``tc`` immediately. Returns a Task the caller
        awaits later. Permission check runs synchronously before returning
        so the caller sees BLOCK decisions before scheduling.

        Exclusive tools go through an internal lock: an exclusive tool
        won't start until all currently-running exclusive tools finish. A
        non-exclusive tool never blocks behind the lock.
        """
        decision = await self._check_permission(tc, permission_callback)

        async def _runner() -> ToolResult:
            return await self._run_one_with_hooks(tc, decision)

        defn = self._registry.get_definition(tc.name)
        is_exclusive = bool(defn and defn.schema.exclusive)

        if is_exclusive:
            async def _exclusive_runner() -> ToolResult:
                async with self._exclusive_lock:
                    return await _runner()
            return asyncio.create_task(_exclusive_runner())
        return asyncio.create_task(_runner())

    async def collect(
        self,
        pending: dict[str, asyncio.Task[ToolResult]],
        order: list[ToolCall],
    ) -> list[ToolResult]:
        """Await every dispatched task and return results in ``order``.

        Handles the same FatalToolError / generic-exception split as
        ``execute_tool_calls``: a fatal error cancels remaining siblings
        and re-raises; other exceptions become error-stub ToolResults.
        """
        if not pending:
            return []
        tasks = list(pending.values())
        try:
            await asyncio.gather(*tasks, return_exceptions=False)
        except FatalToolError:
            for t in tasks:
                if not t.done():
                    t.cancel()
            # Suppress CancelledError from siblings; re-raise the fatal.
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        except Exception as exc:  # noqa: BLE001
            logger.error("Unexpected dispatch error: %s", exc)

        results: dict[str, ToolResult] = {}
        for tc in order:
            t = pending.get(tc.id)
            if t is None or not t.done():
                results[tc.id] = ToolResult(
                    tool_call_id=tc.id, name=tc.name,
                    content="Error: execution failed", is_error=True,
                )
                continue
            try:
                results[tc.id] = t.result()
            except Exception as exc:  # noqa: BLE001
                results[tc.id] = ToolResult(
                    tool_call_id=tc.id, name=tc.name,
                    content=f"Error: {type(exc).__name__}: {exc}", is_error=True,
                )
        return [results[tc.id] for tc in order]

    # -- batch-mode API (existing) ------------------------------------------

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
            results[tc.id] = await self._run_one_with_hooks(tc, decision)

        try:
            async with asyncio.TaskGroup() as tg:
                for tc, decision in decisions:
                    defn = self._registry.get_definition(tc.name)
                    if defn and defn.schema.exclusive:
                        # Wrap in the exclusive lock; still scheduled in TG.
                        async def _exclusive(tc=tc, decision=decision):
                            async with self._exclusive_lock:
                                results[tc.id] = await self._run_one_with_hooks(tc, decision)
                        tg.create_task(_exclusive())
                    else:
                        tg.create_task(_run_one(tc, decision))
        except* FatalToolError as eg:
            # A fatal error (permission subsystem panic, OOM, etc.) cancels
            # all sibling tasks. Re-raise the first one so the agent loop
            # sees the fatal failure instead of silently downgrading it.
            for exc in eg.exceptions:
                logger.error("FATAL executor error — siblings cancelled: %s", exc)
            # asyncio.TaskGroup already cancelled siblings on the first
            # FatalToolError; re-raise so the agent loop sees the failure.
            raise eg.exceptions[0]
        except* Exception as eg:
            # Non-fatal errors: log and let other siblings finish. Each
            # affected call returns a stub via the dict-default in the
            # comprehension below.
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

    async def _run_one_with_hooks(
        self, tc: ToolCall, decision: PermissionDecision,
    ) -> ToolResult:
        """Shared body between batch and streaming modes: runs the tool,
        fires post_tool_use, handles BLOCK."""
        hook_system = self._gate.hooks if self._gate is not None else None
        if decision == PermissionDecision.BLOCK:
            blocked = ToolResult(
                tool_call_id=tc.id, name=tc.name,
                content=self._block_reason(tc), is_error=True,
            )
            if hook_system is not None:
                await hook_system.fire(
                    "post_tool_use",
                    tool_call=tc, result=blocked, blocked=True,
                )
            return blocked
        start = asyncio.get_running_loop().time()
        res = await self._execute_single(tc)
        if hook_system is not None:
            duration = asyncio.get_running_loop().time() - start
            await hook_system.fire(
                "post_tool_use",
                tool_call=tc, result=res, blocked=False, duration=duration,
            )
        return res

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
        except FatalToolError:
            # Propagate — TaskGroup will cancel sibling tasks. Do NOT swallow
            # into a ToolResult; the agent loop needs to see the failure.
            raise
        except Exception as e:
            logger.warning("Tool %s raised: %s", tc.name, e)
            return ToolResult(
                tool_call_id=tc.id, name=tc.name,
                content=f"Error: {type(e).__name__}: {e}", is_error=True,
            )
