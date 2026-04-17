"""Plan mode — three-phase cycle: Plan → Approve → Execute.

Enforced at permission layer, not via prompts. When active, the executor
blocks all tools except the allowed read-only set.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path


class PlanMode:
    """Session-level plan mode state."""

    # Tools permitted while plan mode is active.
    # Note: exit_plan_mode is WRITE permission but specially allowed.
    ALLOWED_TOOLS: frozenset[str] = frozenset({
        "read_file",
        "list_dir",
        "search_files",
        "web_search",
        "tool_search",
        "ask_user",
        "exit_plan_mode",
    })

    def __init__(self, save_dir: str | Path = "plans/") -> None:
        self._active = False
        self._session_id: str | None = None
        self._entered_at: dt.datetime | None = None
        self._save_dir = Path(save_dir)

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

    def enter(self, session_id: str) -> None:
        """Activate plan mode for the given session."""
        self._active = True
        self._session_id = session_id
        self._entered_at = dt.datetime.now()

    def exit(self, approved: bool, plan_content: str = "") -> str:
        """Deactivate. If approved and content provided, save to ``plans/``.

        Returns a human-readable result message.
        """
        if not self._active:
            return "Plan mode was not active."

        session_id = self._session_id or "unknown"
        entered_at = self._entered_at
        self._active = False
        self._session_id = None
        self._entered_at = None

        if not approved:
            return "Plan discarded. Plan mode deactivated."

        if not plan_content.strip():
            return "Plan approved but no content to save. Plan mode deactivated."

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
        return f"Plan approved and saved to {path}. Plan mode deactivated."

    # ------------------------------------------------------------------
    # Permission enforcement
    # ------------------------------------------------------------------

    def is_tool_allowed(self, tool_name: str) -> bool:
        """While active: only whitelisted tools allowed. While inactive: all allowed."""
        if not self._active:
            return True
        return tool_name in self.ALLOWED_TOOLS
