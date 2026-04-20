"""Prompt cache prefix tracking.

Modern providers (Anthropic, DeepSeek, OpenAI) compute a KV cache for a
stable prefix of the input. If the first N tokens are byte-identical across
turns, the provider charges less and answers faster. This module just hashes
the frozen system-prompt prefix at session init so we can cheaply check
whether a built message list is still cache-compatible (useful for tests
and for Fork inheritance later in Phase 2b).

This is a pure helper — it does not call any provider API. Cache hits are
implicit on the provider side; we just maintain awareness.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.types import Message


def hash_messages(messages: list[Message]) -> str:
    """Stable SHA-256 of a message list. Used as a cache identity."""
    serialized = json.dumps(messages, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class PromptCache:
    """Tracks the cache identity of the frozen system-prompt prefix."""

    def __init__(self, frozen_system: list[Message]) -> None:
        self._frozen_system = list(frozen_system)
        self._prefix_hash = hash_messages(self._frozen_system)

    @property
    def prefix_hash(self) -> str:
        return self._prefix_hash

    @property
    def prefix_length(self) -> int:
        return len(self._frozen_system)

    def is_cache_compatible(self, messages: list[Message]) -> bool:
        """True iff ``messages`` starts with the cached prefix byte-for-byte."""
        if len(messages) < self.prefix_length:
            return False
        return hash_messages(messages[: self.prefix_length]) == self._prefix_hash

    def on_fork(self) -> "PromptCache":
        """Fork inherits the same prefix → shared cache identity."""
        return PromptCache(self._frozen_system)

    def on_subagent(self, new_system: list["Message"]) -> "PromptCache":
        """Subagent gets an independent prefix → different cache identity."""
        return PromptCache(new_system)
