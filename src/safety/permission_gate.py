"""Five-layer permission gate.

Replaces the inline 3-point check that lived in ``StreamingExecutor`` in
Phase 1. Each tool call passes through the chain in order; any layer can
return ``BLOCK`` to short-circuit. The first ``BLOCK`` wins.

Layer order — taken from ``docs/ARCHITECTURE.md`` Section G:

    1. Schema validation        — tool exists in registry
    2. Tool self-check          — the tool's own ``validate()`` callable
                                  returns a ``ValidateResult`` (allow / ask /
                                  deny + optional ``updated_input``)
    3. Plan-mode check          — authoritative; non-overridable
    4. Hook check               — ``pre_tool_use`` hooks may deny
    5. Permission level         — read-only/write/execute/destructive →
                                  ALLOW / ASK_USER / BLOCK
                                  (upgrades to ASK if layer 2 requested it)

Rationale for putting plan-mode at layer 3 (before hooks):
    A user-defined hook should never be able to *unblock* a tool that plan
    mode forbids. Architectural rule from ``docs/DECISIONS.md`` D7.

Rationale for ``ask`` semantics (design note, 2026-04-20):
    A validator returning ``ask`` does NOT bypass plan-mode or hook layers —
    those still run and can still BLOCK. Only layer 5 is affected: the tool
    is prompted to the user even if it's otherwise READ_ONLY or YOLO-mode
    would have auto-allowed it. This preserves the plan-mode architectural
    guarantee while letting tools escalate borderline calls.
"""

from __future__ import annotations

import inspect
import logging
from typing import TYPE_CHECKING, Awaitable, Callable

from src.core.types import (
    PermissionDecision, PermissionLevel, ToolCall, ToolContext, ValidateResult,
)

if TYPE_CHECKING:
    from src.safety.hook_system import HookSystem
    from src.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


PermissionCallback = Callable[[ToolCall], Awaitable[PermissionDecision]]


class PermissionGate:
    """Five-layer permission gate. Use ``check()`` for each tool call."""

    def __init__(
        self,
        registry: ToolRegistry,
        ctx: ToolContext,
        hooks: HookSystem | None = None,
        yolo_mode: bool = False,
    ) -> None:
        self._registry = registry
        self._ctx = ctx
        self._hooks = hooks
        self._yolo = yolo_mode
        self._last_block_reason: str = ""

    # -- public --------------------------------------------------------------

    @property
    def last_block_reason(self) -> str:
        """Reason string for the most recent BLOCK decision."""
        return self._last_block_reason

    async def check(
        self, tc: ToolCall, callback: PermissionCallback,
    ) -> PermissionDecision:
        """Run the 5-layer chain. Returns ALLOW / ASK_USER / BLOCK.

        Side effect: if the tool's validator returns ``updated_input``, the
        caller's ``tc.arguments`` dict is replaced before the chain continues
        — downstream layers (plan-mode, hooks) see the rewritten args.
        """
        # Layer 1 — schema validation
        defn = self._registry.get_definition(tc.name)
        if defn is None:
            self._last_block_reason = f"Error: unknown tool '{tc.name}'"
            return PermissionDecision.BLOCK

        # Layer 2 — tool self-check
        validator_wants_ask = False
        if defn.validate is not None:
            result = await self._run_validator(defn.validate, tc)
            if result.decision == "deny":
                self._last_block_reason = (
                    f"Error: '{tc.name}' validation failed: "
                    f"{result.reason or '(no reason given)'}"
                )
                return PermissionDecision.BLOCK
            if result.updated_input is not None:
                tc.arguments = result.updated_input
            if result.decision == "ask":
                validator_wants_ask = True

        # Layer 3 — plan-mode check (AUTHORITATIVE — hooks can't override)
        if not self._ctx.plan_mode.is_tool_allowed(tc.name):
            self._last_block_reason = (
                f"Error: '{tc.name}' blocked by plan mode"
            )
            return PermissionDecision.BLOCK

        # Layer 4 — pre_tool_use hooks
        if self._hooks is not None:
            result = await self._hooks.fire(
                "pre_tool_use", tool_call=tc,
            )
            if result == "deny":
                self._last_block_reason = (
                    f"Error: '{tc.name}' blocked by hook"
                )
                return PermissionDecision.BLOCK

        # Layer 5 — permission level (with layer 2's ASK override honored)
        level = defn.schema.permission_level
        if level == PermissionLevel.DESTRUCTIVE:
            self._last_block_reason = (
                f"Error: '{tc.name}' is destructive and always blocked"
            )
            return PermissionDecision.BLOCK
        if validator_wants_ask:
            # Validator explicitly asked for user confirmation — override the
            # default auto-allow path. YOLO mode does NOT suppress this: the
            # tool itself asked, so we ask.
            return await callback(tc)
        if level == PermissionLevel.READ_ONLY:
            return PermissionDecision.ALLOW
        if self._yolo:
            # YOLO: skip user prompts for write/execute. Shadow classifier
            # for dangerous ops is a Phase 2b add — for now YOLO is honest.
            return PermissionDecision.ALLOW
        return await callback(tc)

    # -- internals -----------------------------------------------------------

    async def _run_validator(
        self, validator: Callable, tc: ToolCall,
    ) -> ValidateResult:
        """Invoke the tool's validator with ctx kwarg, normalising errors."""
        try:
            maybe = validator(ctx=self._ctx, **tc.arguments)
            if inspect.isawaitable(maybe):
                maybe = await maybe
        except TypeError as exc:
            # Validator signature doesn't accept ctx — retry without it.
            # Lets older validators keep working while we migrate.
            if "ctx" in str(exc):
                try:
                    maybe = validator(**tc.arguments)
                    if inspect.isawaitable(maybe):
                        maybe = await maybe
                except Exception as exc2:  # noqa: BLE001
                    logger.warning(
                        "Tool self-check raised for %s: %s", tc.name, exc2,
                    )
                    return ValidateResult(
                        decision="deny", reason=f"validate() raised: {exc2}",
                    )
            else:
                logger.warning("Tool self-check raised for %s: %s", tc.name, exc)
                return ValidateResult(
                    decision="deny", reason=f"validate() raised: {exc}",
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Tool self-check raised for %s: %s", tc.name, exc)
            return ValidateResult(
                decision="deny", reason=f"validate() raised: {exc}",
            )

        if not isinstance(maybe, ValidateResult):
            logger.warning(
                "Validator for %s returned %r instead of ValidateResult",
                tc.name, type(maybe).__name__,
            )
            return ValidateResult(
                decision="deny",
                reason=f"validator returned {type(maybe).__name__}, expected ValidateResult",
            )
        return maybe
