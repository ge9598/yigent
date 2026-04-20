"""Lifecycle hook system.

Hooks fire at well-defined points in the agent's lifecycle and can be used
for telemetry, validation, blocking, or side-effect orchestration. They are
the primary extensibility seam for users who want to customize behaviour
without forking the harness.

Eight events are defined in ``docs/ARCHITECTURE.md`` Section H. Each event
carries a ``data: dict`` payload that hook callables receive as **kwargs.
Hooks can return ``"deny"`` from ``pre_tool_use`` to block a tool call —
the permission gate honors this.

Hooks may be sync or async callables, or shell commands (str — Phase 2b).
A misbehaving hook is logged and isolated; it does not break the chain.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections import defaultdict
from typing import Any, Awaitable, Callable, Literal

logger = logging.getLogger(__name__)


HookResult = Literal["allow", "deny"] | None
HookCallable = Callable[..., Awaitable[HookResult] | HookResult]

EventName = Literal[
    "session_start",
    "pre_tool_use",
    "post_tool_use",
    "pre_compression",
    "post_compression",
    "plan_approved",
    "budget_warning",
    "session_end",
]

_VALID_EVENTS: frozenset[str] = frozenset({
    "session_start", "pre_tool_use", "post_tool_use",
    "pre_compression", "post_compression",
    "plan_approved", "budget_warning", "session_end",
})


class HookSystem:
    """Lifecycle hook registry. ``fire`` returns whether any hook denied."""

    def __init__(self) -> None:
        self._hooks: dict[str, list[HookCallable]] = defaultdict(list)

    # -- registration --------------------------------------------------------

    def register(self, event: EventName, hook: HookCallable) -> None:
        if event not in _VALID_EVENTS:
            raise ValueError(
                f"Unknown hook event '{event}'. "
                f"Valid: {sorted(_VALID_EVENTS)}"
            )
        self._hooks[event].append(hook)

    def clear(self, event: EventName | None = None) -> None:
        if event is None:
            self._hooks.clear()
        else:
            self._hooks.pop(event, None)

    def count(self, event: EventName) -> int:
        return len(self._hooks.get(event, []))

    # -- firing --------------------------------------------------------------

    async def fire(self, event: EventName, **data: Any) -> HookResult:
        """Fire all hooks for ``event``. Returns ``"deny"`` if any hook denies.

        A hook raising an exception is logged and treated as ``allow`` — one
        broken hook must not break the chain.
        """
        if event not in _VALID_EVENTS:
            logger.warning("Firing unknown hook event '%s'", event)
            return None

        denied = False
        for hook in self._hooks.get(event, []):
            try:
                result = hook(**data)
                if inspect.isawaitable(result):
                    result = await result
            except Exception as exc:  # noqa: BLE001 — isolation is the point
                logger.error(
                    "Hook %r for event %r raised: %s",
                    getattr(hook, "__name__", hook), event, exc,
                )
                continue
            if result == "deny":
                denied = True
                # Keep firing remaining hooks for telemetry, but the
                # outcome is locked in.
        return "deny" if denied else "allow"


# ---------------------------------------------------------------------------
# Loader — read hooks from configs/hooks.yaml
# ---------------------------------------------------------------------------

def load_hooks_from_config(hooks_config: dict[str, Any]) -> HookSystem:
    """Build a HookSystem from the parsed ``hooks:`` dict in default.yaml.

    Phase 2 supports only Python dotted-path hook references:

        pre_tool_use:
          - "my_module.my_hook"

    Shell-command hooks (string commands) are deferred to Phase 2b.
    """
    system = HookSystem()
    if not hooks_config:
        return system

    for event, hook_refs in hooks_config.items():
        if event not in _VALID_EVENTS:
            logger.warning("Skipping unknown hook event '%s' in config", event)
            continue
        if not isinstance(hook_refs, list):
            logger.warning(
                "Hooks for event '%s' must be a list, got %r",
                event, type(hook_refs).__name__,
            )
            continue
        for ref in hook_refs:
            if not isinstance(ref, str):
                logger.warning("Skipping non-string hook ref %r", ref)
                continue
            try:
                hook = _import_dotted(ref)
            except (ImportError, AttributeError) as exc:
                logger.warning("Failed to import hook %r: %s", ref, exc)
                continue
            system.register(event, hook)  # type: ignore[arg-type]
    return system


def _import_dotted(path: str) -> HookCallable:
    """Import a callable from a dotted path like 'pkg.mod.func'."""
    module_path, _, attr = path.rpartition(".")
    if not module_path:
        raise ImportError(f"Bad dotted path: {path!r}")
    import importlib
    mod = importlib.import_module(module_path)
    fn = getattr(mod, attr)
    if not callable(fn):
        raise AttributeError(f"{path!r} is not callable")
    return fn
