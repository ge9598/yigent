"""OpenAI-compatible streaming provider.

Works with any OpenAI-compatible API (OpenAI, DeepSeek, Qwen, local vLLM, etc.)
by changing ``base_url``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field

from openai import AsyncOpenAI

from src.core.types import Message, StreamChunk, ToolCall, ToolSchema

from .base import LLMProvider

logger = logging.getLogger(__name__)


@dataclass
class _ToolCallAccumulator:
    """Accumulates incremental tool call deltas from the stream."""
    index: int
    id: str = ""
    name: str = ""
    arguments_buffer: str = ""


class OpenAICompatProvider(LLMProvider):
    """Provider for any OpenAI-compatible chat completions API."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
        default_timeout: float = 120.0,
    ) -> None:
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=default_timeout,
        )
        self._default_model = model

    # -- public interface ----------------------------------------------------

    async def stream_message(
        self,
        messages: list[Message],
        model: str | None = None,
        tools: list[ToolSchema] | None = None,
        temperature: float = 0.0,
    ) -> AsyncGenerator[StreamChunk, None]:
        model = model or self._default_model

        kwargs: dict = {
            "model": model,
            "messages": messages,  # type: ignore[arg-type]
            "temperature": temperature,
            "stream": True,
        }
        if tools:
            kwargs["tools"] = [t.to_openai_tool() for t in tools]

        stream = await self._client.chat.completions.create(**kwargs)

        accumulators: dict[int, _ToolCallAccumulator] = {}

        async for chunk in stream:
            if not chunk.choices:
                continue

            choice = chunk.choices[0]
            delta = choice.delta
            finish_reason = choice.finish_reason

            # --- text tokens ---
            if delta.content:
                yield StreamChunk(type="token", data=delta.content, model=model)

            # --- tool call deltas ---
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index

                    if idx not in accumulators:
                        acc = _ToolCallAccumulator(index=idx)
                        accumulators[idx] = acc
                        if tc_delta.id:
                            acc.id = tc_delta.id
                        if tc_delta.function and tc_delta.function.name:
                            acc.name = tc_delta.function.name
                        yield StreamChunk(
                            type="tool_call_start",
                            data={"id": acc.id, "name": acc.name},
                            model=model,
                        )
                    else:
                        acc = accumulators[idx]

                    if tc_delta.function and tc_delta.function.arguments:
                        acc.arguments_buffer += tc_delta.function.arguments

            # --- stream finished ---
            if finish_reason is not None:
                if finish_reason == "tool_calls" and accumulators:
                    for acc in sorted(accumulators.values(), key=lambda a: a.index):
                        tool_call = self._parse_accumulated(acc)
                        yield StreamChunk(
                            type="tool_call_complete",
                            data=tool_call,
                            model=model,
                        )
                yield StreamChunk(type="done", data=finish_reason, model=model)
                return

    # -- internals -----------------------------------------------------------

    @staticmethod
    def _parse_accumulated(acc: _ToolCallAccumulator) -> ToolCall:
        try:
            arguments = json.loads(acc.arguments_buffer) if acc.arguments_buffer else {}
        except json.JSONDecodeError:
            logger.warning(
                "Failed to parse tool call arguments for %s (id=%s): %s",
                acc.name, acc.id, acc.arguments_buffer,
            )
            arguments = {"_raw": acc.arguments_buffer}
        return ToolCall(id=acc.id, name=acc.name, arguments=arguments)
