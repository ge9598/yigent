"""Tests for SlashDispatcher — Aider-style cmd_* introspection."""

from __future__ import annotations

import pytest

from src.ui.slash_commands import DispatchResult, SlashDispatcher


# ---------------------------------------------------------------------------
# looks_like_command — the cheap prefix heuristic
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text, expected",
    [
        ("/quit", True),
        ("/tools", True),
        ("/plan-mode", True),
        ("/tool_search", True),
        ("  /quit", True),       # leading whitespace tolerated
        ("hello", False),
        ("", False),
        ("/", False),            # bare slash
        ("//escape", False),     # slash + slash → not a command
        ("explain /tmp/foo", False),   # slash not at start
        ("/ spaced", False),     # slash + space: no command token
    ],
)
def test_looks_like_command(text: str, expected: bool) -> None:
    assert SlashDispatcher.looks_like_command(text) is expected


# ---------------------------------------------------------------------------
# dispatch — the core routing behavior
# ---------------------------------------------------------------------------

class _Handlers:
    """Test double with a mix of sync/async/quit/unknown-reporter methods."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.unknown: list[tuple[str, list[str]]] = []

    def cmd_tools(self, args: str) -> None:
        """List tools."""
        self.calls.append(("tools", args))

    async def cmd_budget(self, args: str) -> None:
        """Show budget."""
        self.calls.append(("budget", args))

    def cmd_quit(self, args: str):
        """Exit."""
        self.calls.append(("quit", args))
        return SlashDispatcher.QUIT_SENTINEL

    def cmd_plan_mode(self, args: str) -> None:
        """Enter plan mode."""
        self.calls.append(("plan-mode", args))

    def report_unknown_command(self, name: str, known: list[str]) -> None:
        self.unknown.append((name, known))


@pytest.mark.asyncio
async def test_dispatch_known_sync_command() -> None:
    h = _Handlers()
    d = SlashDispatcher(h)
    result = await d.dispatch("/tools")
    assert result is DispatchResult.HANDLED
    assert h.calls == [("tools", "")]


@pytest.mark.asyncio
async def test_dispatch_known_async_command() -> None:
    h = _Handlers()
    d = SlashDispatcher(h)
    result = await d.dispatch("/budget")
    assert result is DispatchResult.HANDLED
    assert h.calls == [("budget", "")]


@pytest.mark.asyncio
async def test_dispatch_quit_sentinel() -> None:
    h = _Handlers()
    d = SlashDispatcher(h)
    result = await d.dispatch("/quit")
    assert result is DispatchResult.QUIT


@pytest.mark.asyncio
async def test_dispatch_passes_arguments() -> None:
    h = _Handlers()
    d = SlashDispatcher(h)
    await d.dispatch("/tools verbose")
    assert h.calls == [("tools", "verbose")]


@pytest.mark.asyncio
async def test_dispatch_underscore_to_dash_naming() -> None:
    """cmd_plan_mode is invoked as /plan-mode, not /plan_mode."""
    h = _Handlers()
    d = SlashDispatcher(h)
    result = await d.dispatch("/plan-mode")
    assert result is DispatchResult.HANDLED
    assert h.calls == [("plan-mode", "")]


@pytest.mark.asyncio
async def test_dispatch_unknown_command_is_handled_not_passthrough() -> None:
    """Bug we're fixing: /model used to leak through to the LLM."""
    h = _Handlers()
    d = SlashDispatcher(h)
    result = await d.dispatch("/model")
    assert result is DispatchResult.HANDLED   # NOT NOT_COMMAND
    assert h.calls == []                       # no handler fired
    assert h.unknown and h.unknown[0][0] == "model"
    # The 'known' list passed to the reporter should include our registered cmds.
    known = h.unknown[0][1]
    assert "tools" in known and "quit" in known and "plan-mode" in known


@pytest.mark.asyncio
async def test_dispatch_non_slash_input_is_not_command() -> None:
    h = _Handlers()
    d = SlashDispatcher(h)
    result = await d.dispatch("hello there")
    assert result is DispatchResult.NOT_COMMAND
    assert h.calls == []


@pytest.mark.asyncio
async def test_dispatch_path_in_middle_is_not_command() -> None:
    h = _Handlers()
    d = SlashDispatcher(h)
    result = await d.dispatch("explain /tmp/foo please")
    assert result is DispatchResult.NOT_COMMAND


# ---------------------------------------------------------------------------
# commands property — powers /help
# ---------------------------------------------------------------------------

def test_commands_property_uses_first_docstring_line() -> None:
    class H:
        def cmd_foo(self, args: str) -> None:
            """First line.

            Second line — should not appear.
            """

        def cmd_bar(self, args: str) -> None:
            """One line only."""

    d = SlashDispatcher(H())
    assert d.commands == {"foo": "First line.", "bar": "One line only."}


def test_commands_ignores_non_cmd_methods() -> None:
    class H:
        def cmd_foo(self, args: str) -> None:
            """Real command."""

        def helper(self) -> None:  # not a command
            pass

        def _private(self) -> None:
            pass

    d = SlashDispatcher(H())
    assert list(d.commands) == ["foo"]


# ---------------------------------------------------------------------------
# fallback reporter when handlers object has no report_unknown_command
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_default_unknown_reporter_prints_to_stdout(capsys) -> None:
    class H:
        def cmd_foo(self, args: str) -> None:
            """noop"""

    d = SlashDispatcher(H())
    result = await d.dispatch("/bogus")
    assert result is DispatchResult.HANDLED
    out = capsys.readouterr().out
    assert "Unknown command: /bogus" in out
    assert "/help" in out
