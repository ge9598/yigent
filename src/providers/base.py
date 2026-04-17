"""LLM Provider abstract base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator

from src.core.types import Message, StreamChunk, ToolSchema


class LLMProvider(ABC):
    """Abstract base for all LLM providers.

    Implementations must yield StreamChunk events from a streaming API call.
    """

    @abstractmethod
    async def stream_message(
        self,
        messages: list[Message],
        model: str | None = None,
        tools: list[ToolSchema] | None = None,
        temperature: float = 0.0,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream a chat completion, yielding StreamChunk events.

        Chunk types:
          token            — text delta
          tool_call_start  — new tool call detected (id + name)
          tool_call_delta  — incremental argument JSON fragment
          tool_call_complete — fully accumulated ToolCall
          done             — stream finished (data = finish_reason)
        """
        yield  # pragma: no cover — make this a generator
