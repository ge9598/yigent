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
import re
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
        aux_provider: object | None = None,
        yolo_breaker_threshold: int = 5,
        yolo_cache_size: int = 128,
    ) -> None:
        from collections import OrderedDict

        from src.context.circuit_breaker import CircuitBreaker

        self._registry = registry
        self._ctx = ctx
        self._hooks = hooks
        self._yolo = yolo_mode
        self._aux_provider = aux_provider
        self._last_block_reason: str = ""
        # Trip the YOLO shadow classifier off after N consecutive aux-LLM
        # failures (upstream overload, key expired, etc.). Regex pre-filter
        # still runs — only the aux-LLM step short-circuits to "risky".
        # Default 5: tolerates a handful of transient 503s without flipping
        # the breaker. Audit Top10 #10.
        self._yolo_breaker = CircuitBreaker(threshold=yolo_breaker_threshold)
        # Bounded LRU cache for shadow-classifier results so a long YOLO
        # session can't monotonically grow memory. Audit Top10 #10.
        self._yolo_cache: OrderedDict[str, str] = OrderedDict()
        self._yolo_cache_size = yolo_cache_size

    # -- public --------------------------------------------------------------

    @property
    def last_block_reason(self) -> str:
        """Reason string for the most recent BLOCK decision."""
        return self._last_block_reason

    @property
    def hooks(self) -> "HookSystem | None":
        """Hook system reference (so the executor can fire post_tool_use)."""
        return self._hooks

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
            # Unit 10 — destructive tools require explicit user confirmation
            # (not unconditional block). Per ARCHITECTURE.md §G "require
            # confirmation for destructive". The user must respond to the
            # prompt; YOLO does NOT bypass this — destructive is the one
            # category that always interrupts.
            return await callback(tc)
        if validator_wants_ask:
            # Validator explicitly asked for user confirmation — override the
            # default auto-allow path. YOLO mode does NOT suppress this: the
            # tool itself asked, so we ask.
            return await callback(tc)
        if level == PermissionLevel.READ_ONLY:
            return PermissionDecision.ALLOW
        if self._yolo:
            # Unit 10 — YOLO shadow classifier. A regex pre-filter catches
            # the obvious-bad operations (`rm -rf /`, fork bombs, …); for
            # everything else, an aux-LLM classifies safe / risky / dangerous.
            # Dangerous → BLOCK, risky → ask user, safe → auto-allow.
            shadow = await self._yolo_shadow_classify(tc)
            if shadow == "dangerous":
                self._last_block_reason = (
                    f"Error: '{tc.name}' blocked by YOLO shadow classifier "
                    "(dangerous operation detected)"
                )
                return PermissionDecision.BLOCK
            if shadow == "risky":
                return await callback(tc)
            return PermissionDecision.ALLOW
        return await callback(tc)

    # -- Unit 10 — YOLO shadow classifier -----------------------------------

    # Regex pre-filter: obviously-bad bash patterns. Hits short-circuit to
    # "dangerous" without paying for an aux-LLM call. Pre-compiled so the
    # check loop in _yolo_shadow_classify doesn't re-compile on every call
    # (audit B9 / Top10 #12).
    _YOLO_DANGER_PATTERNS = tuple(
        re.compile(p) for p in (
            r"\brm\s+-rf?\s+/(?:\s|$)",          # rm -rf /
            r"\brm\s+-rf?\s+~(?:\s|$)",          # rm -rf ~
            r"\bdd\s+.*of=/dev/(?:sd|nvme|hd)",  # dd of=/dev/sda
            r"\bmkfs\b",                          # mkfs.*
            r":\(\)\s*\{[^}]*:\|\s*:[^}]*\}\s*;\s*:",  # bash fork bomb
            r"\bcurl\b[^|]*\|\s*(?:sudo\s+)?(?:bash|sh)\b",   # curl ... | sh
            r"\bwget\b[^|]*\|\s*(?:sudo\s+)?(?:bash|sh)\b",   # wget ... | sh
            r"\bchmod\s+(?:-R\s+)?[0-7]*7[0-7]*\s+/(?:bin|etc|usr|var)",
            r"\bformat\s+[a-z]:",                # Windows: format c:
            r">\s*/dev/sd[a-z]",                 # > /dev/sda
        )
    )

    async def _yolo_shadow_classify(self, tc: ToolCall) -> str:
        """Classify a tool call as 'safe' | 'risky' | 'dangerous'.

        Step 1 — regex pre-filter on bash-shaped args (catches the canonical
        bad commands without an LLM call).
        Step 2 — aux-LLM classification using the gate's auxiliary provider.

        On aux failure we default to 'risky' (ask the user) rather than 'safe'
        — the regex pre-filter only catches ~10 canonical bad patterns, so
        defaulting-safe turned aux outages into a silent auto-allow channel
        for anything outside those patterns. See audit Top-10 #1.
        """
        # Pull the most likely "command" string from the args. bash takes
        # ``command``; other tools may have different shapes — string-flatten.
        flat_args = " ".join(
            str(v) for v in tc.arguments.values() if isinstance(v, (str, int, float))
        )
        for pattern in self._YOLO_DANGER_PATTERNS:
            if pattern.search(flat_args):
                logger.warning("YOLO regex caught dangerous: %s", pattern.pattern)
                return "dangerous"

        # Aux-LLM step (skipped if no provider OR breaker tripped).
        provider = getattr(self, "_aux_provider", None)
        if provider is None:
            # No classifier wired at all. The operator consciously opted into
            # YOLO without a shadow classifier — trust that choice and let
            # WRITE/EXECUTE pass. DESTRUCTIVE still blocks at layer 5.
            return "safe"
        if self._yolo_breaker.is_open:
            # Aux repeatedly failed this session. Default to risky (prompt
            # the user) instead of silently auto-allowing — the breaker must
            # not become a bypass channel. Saves the round-trip but preserves
            # the security surface.
            return "risky"
        # Cache by (tool, args-fingerprint) to avoid re-classifying identical
        # calls. We hash a short repr (truncated 200 chars) rather than the
        # full args — a 50KB write_file content does not need full re-hashing
        # to find a cache hit. Bounded LRU prevents long-session OOM.
        # Audit Top10 #10.
        import hashlib
        import json
        args_repr = repr(sorted(tc.arguments.items()))[:200]
        key = hashlib.sha256(f"{tc.name}::{args_repr}".encode()).hexdigest()
        cache = self._yolo_cache
        if key in cache:
            cache.move_to_end(key)  # LRU touch
            return cache[key]

        prompt_msgs = [
            {
                "role": "system",
                "content": (
                    "You classify tool calls for safety. Reply with EXACTLY one "
                    "word: 'safe' (no harm possible), 'risky' (could cause harm "
                    "in edge cases — recommend user confirmation), or 'dangerous' "
                    "(would cause data loss, system damage, or security breach)."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Tool: {tc.name}\n"
                    f"Arguments: {json.dumps(tc.arguments, default=str)}\n"
                    "Classification:"
                ),
            },
        ]
        try:
            text = ""
            async for chunk in provider.stream_message(
                messages=prompt_msgs, temperature=0.0,
            ):
                if chunk.type == "token":
                    text += chunk.data
                elif chunk.type == "done":
                    break
            classification = text.strip().lower().split()[0] if text.strip() else "safe"
            if classification not in ("safe", "risky", "dangerous"):
                classification = "safe"  # unknown = err on the side of letting
                                         # the operator proceed (they enabled YOLO)
            cache[key] = classification
            # Bounded LRU eviction: drop oldest when over cap.
            while len(cache) > self._yolo_cache_size:
                cache.popitem(last=False)
            self._yolo_breaker.record_success()
            return classification
        except Exception as exc:  # noqa: BLE001
            self._yolo_breaker.record_failure()
            exc_name = type(exc).__name__
            if self._yolo_breaker.is_open:
                logger.warning(
                    "YOLO shadow classifier failed (%s) — breaker tripped "
                    "after %d consecutive failures; subsequent WRITE/EXECUTE "
                    "calls will prompt the user (regex pre-filter still active)",
                    exc_name, self._yolo_breaker.failures,
                )
            else:
                logger.warning(
                    "YOLO shadow classifier failed (%s) — defaulting risky "
                    "(%d/%d before breaker trips)",
                    exc_name, self._yolo_breaker.failures, self._yolo_breaker.threshold,
                )
            # Fail-safe: default to 'risky' (prompt the user) instead of
            # 'safe' (silent auto-allow). See audit Top-10 #1.
            return "risky"

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
