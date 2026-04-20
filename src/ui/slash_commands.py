"""Slash command dispatcher for the CLI.

Design — Aider's ``cmd_*`` introspection pattern (aider/commands.py):
    Register a plain object whose methods named ``cmd_<name>`` become
    commands. The docstring's first line becomes the ``/help`` description.
    Unknown commands are intercepted with an error, never passed to the LLM.

Why this shape (vs. decorators or dict registration):
    - Zero registration boilerplate when adding a new command — just write
      a method named ``cmd_foo``.
    - The object holding the methods naturally closes over CLI state
      (console, registry, budget, plan_mode, session_id, conversation)
      so handlers don't need a context parameter.
    - ``/help`` is generated from docstrings, keeping help text colocated
      with implementation.

Unknown-command policy — HANDLED, not passed through. Sending a typo'd
``/modle`` to the model wastes tokens and often triggers spurious
implementation attempts (the bug that motivated this module). Gemini CLI
passes unknown commands to the model; Claude Code and Aider reject. We
follow the latter.
"""

from __future__ import annotations

import inspect
from enum import Enum
from typing import Any, Awaitable, Callable

Handler = Callable[[str], Any]  # may be sync or async, returns anything


class DispatchResult(str, Enum):
    """Outcome of dispatching an input line."""
    NOT_COMMAND = "not_command"      # didn't look like a slash command
    HANDLED = "handled"              # handled, continue the REPL
    QUIT = "quit"                    # handled, exit the REPL


class SlashDispatcher:
    """Dispatches slash commands via ``cmd_*`` introspection.

    Usage::

        class MyCommands:
            def cmd_tools(self, args: str) -> None:
                '''Show registered tools.'''
                print(...)

        dispatcher = SlashDispatcher(MyCommands())
        result = await dispatcher.dispatch(user_input)
    """

    # Sentinel return values a handler can use to signal "exit the REPL".
    QUIT_SENTINEL = object()

    def __init__(self, handlers_obj: object) -> None:
        self._handlers_obj = handlers_obj
        self._commands: dict[str, tuple[Handler, str]] = {}
        for attr in dir(handlers_obj):
            if not attr.startswith("cmd_"):
                continue
            fn = getattr(handlers_obj, attr)
            if not callable(fn):
                continue
            name = attr[len("cmd_"):].replace("_", "-")
            doc = (inspect.getdoc(fn) or "").strip().split("\n", 1)[0]
            self._commands[name] = (fn, doc)

    # -- public API ----------------------------------------------------------

    @property
    def commands(self) -> dict[str, str]:
        """Map of command name → short description (from docstring)."""
        return {name: doc for name, (_, doc) in self._commands.items()}

    @staticmethod
    def looks_like_command(user_input: str) -> bool:
        """True iff the input appears intended as a slash command.

        Heuristic: starts with a literal '/' and the next character is an
        alphanumeric (or dash/underscore). This rejects:
          - plain text:                "hello"
          - paths typed alone:         "/usr/bin/env"  (slash followed by slash → fails)
                                       — technically accepted; realistic user input
                                       like "explain /tmp/foo" is safe because the
                                       '/' is not at position 0.
          - bare slash:                "/"
        """
        stripped = user_input.lstrip()
        if len(stripped) < 2 or not stripped.startswith("/"):
            return False
        second = stripped[1]
        return second.isalnum() or second in ("-", "_")

    async def dispatch(self, user_input: str) -> DispatchResult:
        """Handle a line from the user.

        Returns:
            NOT_COMMAND — caller should treat as normal LLM input.
            HANDLED     — caller should ``continue`` the REPL.
            QUIT        — caller should break the REPL.
        """
        if not self.looks_like_command(user_input):
            return DispatchResult.NOT_COMMAND

        stripped = user_input.lstrip()[1:]  # drop leading '/'
        parts = stripped.split(maxsplit=1)
        name = parts[0] if parts else ""
        args = parts[1] if len(parts) > 1 else ""

        handler_entry = self._commands.get(name)
        if handler_entry is None:
            self._report_unknown(name)
            return DispatchResult.HANDLED

        handler, _ = handler_entry
        result = handler(args)
        if inspect.isawaitable(result):
            result = await result

        if result is self.QUIT_SENTINEL:
            return DispatchResult.QUIT
        return DispatchResult.HANDLED

    # -- overridable for customized error/help rendering ---------------------

    def _report_unknown(self, name: str) -> None:
        """Report an unknown command. Subclasses may override for pretty UI."""
        reporter = getattr(self._handlers_obj, "report_unknown_command", None)
        if callable(reporter):
            reporter(name, sorted(self._commands.keys()))
        else:
            print(f"Unknown command: /{name}")
            print("Type /help for a list of commands.")
