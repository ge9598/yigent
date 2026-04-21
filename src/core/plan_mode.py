"""Plan mode — three-phase cycle: Plan → Approve → Execute.

Enforced at permission layer, not via prompts. When active, the executor
blocks all tools except the allowed read-only set.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.safety.hook_system import HookSystem
    from src.tools.registry import ToolRegistry


class PlanMode:
    """Session-level plan mode state.

    The allowlist of permitted tools is computed dynamically when ``enter()``
    is called: every READ_ONLY tool in the registry plus ``exit_plan_mode``
    (WRITE-level but specially allowed). This means MCP tools and auto-created
    skills tagged READ_ONLY flow through plan mode automatically — no need
    to maintain a hardcoded list.

    For setups without a registry (e.g. tests), a static fallback set is
    used. Keep the fallback in sync with the canonical built-in READ_ONLY
    tools.
    """

    # Static fallback when no registry has been bound (legacy tests).
    # Must stay in sync with built-in READ_ONLY tools — production wires
    # a registry via set_registry() and the dynamic allowlist supersedes.
    # Note: exit_plan_mode is WRITE permission but specially allowed.
    ALLOWED_TOOLS: frozenset[str] = frozenset({
        "read_file",
        "list_dir",
        "search_files",
        "web_search",
        "tool_search",
        "ask_user",
        "exit_plan_mode",
        "read_memory",
        "list_memory",
    })

    def __init__(
        self,
        save_dir: str | Path = "plans/",
        hook_system: "HookSystem | None" = None,
    ) -> None:
        self._active = False
        self._session_id: str | None = None
        self._entered_at: dt.datetime | None = None
        self._save_dir = Path(save_dir)
        self._plan_buffer: str = ""
        self._hook_system = hook_system
        self._approved: bool = False
        self._rejection_note: str | None = None
        # Dynamic allowlist computed at enter() time from a registry.
        # None means "not bound" — fall back to the static ALLOWED_TOOLS.
        self._dynamic_allowlist: frozenset[str] | None = None
        # Optional registry the dynamic allowlist queries. set_registry()
        # binds it; without binding we use ALLOWED_TOOLS.
        self._registry: Any = None  # ToolRegistry, but avoid import cycle

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def session_id(self) -> str | None:
        return self._session_id

    # ------------------------------------------------------------------
    # Transitions
    # ------------------------------------------------------------------

    def set_registry(self, registry: "ToolRegistry") -> None:
        """Bind a tool registry so the allowlist can be computed dynamically.

        Without this binding, plan mode falls back to the static
        ``ALLOWED_TOOLS`` frozenset (back-compat for tests).
        """
        self._registry = registry

    def enter(self, session_id: str) -> None:
        """Activate plan mode for the given session.

        Recomputes the dynamic allowlist from the bound registry: every
        READ_ONLY tool is allowed, plus ``exit_plan_mode`` (WRITE-level but
        specially allowed). MCP tools registered as READ_ONLY automatically
        flow through.
        """
        self._active = True
        self._session_id = session_id
        self._entered_at = dt.datetime.now()
        self._plan_buffer = ""
        self._approved = False
        self._rejection_note = None
        self._dynamic_allowlist = self._compute_allowlist()

    def _compute_allowlist(self) -> frozenset[str] | None:
        """Build the allowlist from the bound registry, or None to fall back.

        Lazy import to avoid the ToolRegistry import cycle.
        """
        if self._registry is None:
            return None
        from src.core.types import PermissionLevel
        try:
            all_tools = self._registry.all()
        except AttributeError:
            return None
        names: set[str] = set()
        for defn in all_tools:
            level = getattr(defn.schema, "permission_level", None)
            if level == PermissionLevel.READ_ONLY:
                names.add(defn.name)
        names.add("exit_plan_mode")  # special — WRITE but always allowed
        return frozenset(names)

    @property
    def is_approved(self) -> bool:
        return self._approved

    @property
    def rejection_note(self) -> str | None:
        return self._rejection_note

    def set_hook_system(self, hooks: "HookSystem | None") -> None:
        """Late binding for hook system (CLI builds hooks after plan_mode)."""
        self._hook_system = hooks

    async def approve(self) -> str:
        """User-side approval. Fires plan_approved. Plan mode stays active until
        the model calls exit_plan_mode (the model's signal that it's done planning).

        The flag flips so the LLM's next assistant turn sees the approval state
        in the system prompt and knows it can now call write tools by exiting
        plan mode.
        """
        if not self._active:
            return "Plan mode is not active."
        self._approved = True
        if self._hook_system is not None:
            await self._hook_system.fire(
                "plan_approved",
                session_id=self._session_id,
                plan_content=self._plan_buffer,
            )
        return "Plan approved. Agent may now exit plan mode and execute."

    async def reject(self, note: str = "") -> str:
        """User-side rejection. Plan mode stays active so the model can revise."""
        if not self._active:
            return "Plan mode is not active."
        self._approved = False
        self._rejection_note = note or "(no reason given)"
        return f"Plan rejected: {self._rejection_note}. Plan mode remains active for revision."

    def append(self, content: str) -> None:
        """Append content to the internal plan buffer."""
        self._plan_buffer += content

    def get_plan_content(self) -> str:
        """Return the current plan buffer."""
        return self._plan_buffer

    def exit(self) -> str:
        """Deactivate plan mode. Auto-saves buffer to file if it has content.

        Returns a human-readable result message.
        """
        if not self._active:
            return "Plan mode was not active."

        session_id = self._session_id or "unknown"
        entered_at = self._entered_at
        plan_content = self._plan_buffer

        self._active = False
        self._session_id = None
        self._entered_at = None
        self._plan_buffer = ""

        if not plan_content.strip():
            return "Plan mode deactivated."

        self._save_dir.mkdir(parents=True, exist_ok=True)
        ts = (entered_at or dt.datetime.now()).strftime("%Y%m%d_%H%M%S")
        path = self._save_dir / f"{session_id}_{ts}.md"
        header = (
            f"# Plan\n\n"
            f"Session: {session_id}\n"
            f"Created: {ts}\n\n"
            f"---\n\n"
        )
        path.write_text(header + plan_content, encoding="utf-8")
        return f"Plan saved to {path}. Plan mode deactivated."

    # ------------------------------------------------------------------
    # Permission enforcement
    # ------------------------------------------------------------------

    def is_tool_allowed(self, tool_name: str) -> bool:
        """While active: only whitelisted tools allowed. While inactive: all allowed.

        Uses the dynamic allowlist (computed from registry at ``enter()``)
        when bound, otherwise falls back to the static ``ALLOWED_TOOLS``.
        """
        if not self._active:
            return True
        allowlist = self._dynamic_allowlist if self._dynamic_allowlist is not None else self.ALLOWED_TOOLS
        return tool_name in allowlist
