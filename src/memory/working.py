"""L0 working memory.

A thin typed wrapper around the in-session conversation list and todo list.
The conversation is the source of truth for the agent loop; this class
exists to give other modules a typed object to hold (instead of passing the
raw ``list[Message]`` around) and to centralise mutation in case we want
to add invariants later (e.g. preventing two assistant messages in a row).

Phase 2 use is mostly cosmetic; Phase 3 may add: turn counters, todo state
persistence, intermediate "scratchpad" notes that don't go to the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.types import Message


@dataclass
class WorkingMemory:
    """In-memory state for one session. Cleared at session end."""

    conversation: list[Message] = field(default_factory=list)
    todo: list[str] = field(default_factory=list)

    # -- conversation --------------------------------------------------------

    def append(self, message: Message) -> None:
        self.conversation.append(message)

    def extend(self, messages: list[Message]) -> None:
        self.conversation.extend(messages)

    def last_user_text(self) -> str:
        for msg in reversed(self.conversation):
            if msg.get("role") == "user" and msg.get("content"):
                return msg["content"]
        return ""

    @property
    def turn_count(self) -> int:
        """Number of (user, assistant) pairs."""
        return sum(1 for m in self.conversation if m.get("role") == "user")

    # -- todo ----------------------------------------------------------------

    def add_todo(self, item: str) -> None:
        self.todo.append(item)

    def complete_todo(self, item: str) -> bool:
        try:
            self.todo.remove(item)
            return True
        except ValueError:
            return False

    def clear_todo(self) -> None:
        self.todo.clear()
