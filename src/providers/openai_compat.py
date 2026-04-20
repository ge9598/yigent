"""OpenAI-compatible streaming provider.

Works with any OpenAI-compatible API (OpenAI, DeepSeek, Qwen, local vLLM, etc.)
by changing ``base_url``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from openai import AsyncOpenAI

from src.core.types import Message, StreamChunk, ToolCall, ToolSchema

from .base import LLMProvider

if TYPE_CHECKING:
    from .credential_pool import CredentialPool

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
        debug: bool = False,
        credential_pool: "CredentialPool | None" = None,
    ) -> None:
        self._base_url = base_url
        self._fallback_key = api_key
        self._credential_pool = credential_pool
        self._default_timeout = default_timeout
        self._default_model = model
        self._debug = debug
        if credential_pool is None:
            self._client = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=default_timeout,
            )
        else:
            self._client = None  # type: ignore[assignment]

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
            "max_tokens": 4096,
        }
        if tools:
            kwargs["tools"] = [t.to_openai_tool() for t in tools]

        if self._debug:
            logger.warning(
                "REQUEST model=%s messages=%d tools=%d",
                kwargs.get("model"),
                len(kwargs.get("messages", [])),
                len(kwargs.get("tools", []) or []),
            )
            for i, m in enumerate(kwargs.get("messages", [])):
                role = m.get("role", "?")
                content = (m.get("content") or "")[:100]
                logger.warning("  msg[%d] role=%s content=%s...", i, role, content)

        if self._credential_pool is not None:
            key = self._credential_pool.acquire()
            client = AsyncOpenAI(
                api_key=key,
                base_url=self._base_url,
                timeout=self._default_timeout,
            )
        else:
            key = self._fallback_key
            client = self._client  # type: ignore[assignment]

        try:
            stream = await client.chat.completions.create(**kwargs)
        except Exception as exc:
            if self._credential_pool is not None:
                status = getattr(exc, "status_code", None) or getattr(
                    getattr(exc, "response", None), "status_code", None
                )
                self._credential_pool.mark_error(key, status=status)
            raise

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
