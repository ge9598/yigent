"""Anthropic-compatible streaming provider.

Works with native Anthropic API and any endpoint speaking the same wire format
(e.g. MiniMax ``api.minimaxi.com/anthropic``) by changing ``base_url``.

Design note:
    The rest of the harness speaks the OpenAI chat-completions dialect. This
    provider translates between that dialect and the Anthropic messages API:
    on the way in, it extracts the ``system`` role, rewrites ``tool`` results
    as user content blocks, and rebuilds prior assistant tool calls as
    ``tool_use`` blocks; on the way out, it maps the SSE event stream back to
    the harness's unified :class:`StreamChunk` types.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from anthropic import AsyncAnthropic

from src.core.types import Message, StreamChunk, ToolCall, ToolCallDict, ToolSchema

from .base import LLMProvider
from .endpoint_quirks import detect_quirks

if TYPE_CHECKING:
    from .credential_pool import CredentialPool

logger = logging.getLogger(__name__)


@dataclass
class _ToolUseAccumulator:
    """Accumulates a single tool_use content block across SSE events."""
    index: int
    id: str = ""
    name: str = ""
    arguments_buffer: str = ""


@dataclass
class _ThinkingAccumulator:
    """Accumulates a single ``thinking`` content block across SSE events.

    The text is emitted to the caller as ``reasoning_text``; the signed
    block (``{"type": "thinking", "thinking": "...", "signature": "..."}``)
    is preserved verbatim in ``reasoning_details`` for multi-turn recall
    — but only on the official Anthropic endpoint (third-party gateways
    reject the signature).
    """
    index: int
    text_buffer: str = ""
    signature: str = ""


class AnthropicCompatProvider(LLMProvider):
    """Provider for any Anthropic-messages-compatible API."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.anthropic.com",
        model: str = "claude-sonnet-4-5",
        default_timeout: float = 120.0,
        max_tokens: int = 4096,
        debug: bool = False,
        credential_pool: "CredentialPool | None" = None,
    ) -> None:
        self._base_url = base_url
        self._fallback_key = api_key
        self._credential_pool = credential_pool
        self._default_timeout = default_timeout
        self._default_model = model
        self._max_tokens = max_tokens
        self._debug = debug
        self._quirks = detect_quirks(base_url)
        if credential_pool is None:
            self._client = AsyncAnthropic(
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

        system_text, anthropic_messages = self._translate_messages(
            messages, strip_thinking_signature=self._quirks.strip_thinking_signature,
        )

        if self._quirks.forbids_zero_temperature and temperature <= 0.0:
            temperature = self._quirks.min_temperature

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": anthropic_messages,
            "temperature": temperature,
            "max_tokens": self._max_tokens,
        }
        if system_text:
            kwargs["system"] = system_text
        if tools:
            kwargs["tools"] = [self._to_anthropic_tool(t) for t in tools]

        if self._debug:
            logger.warning(
                "REQUEST model=%s system=%d messages=%d tools=%d",
                model,
                len(system_text),
                len(anthropic_messages),
                len(tools or []),
            )
            for i, m in enumerate(anthropic_messages):
                role = m.get("role", "?")
                preview = self._preview_content(m.get("content"))
                logger.warning("  msg[%d] role=%s content=%s", i, role, preview)

        # Accumulators are keyed by tool_use.id (stable across SDK chunks),
        # not by event.index. Some Anthropic-compatible endpoints (e.g.
        # MiniMax /anthropic) deliver content_block_delta events whose
        # event.index ordering can be unstable; the id from
        # content_block_start is the only reliable join key.
        accumulators: dict[str, _ToolUseAccumulator] = {}
        index_to_id: dict[int, str] = {}
        emitted: set[str] = set()
        stop_reason: str | None = None
        had_tools = False
        # One thinking block per index — Anthropic emits them before text/tool blocks.
        thinking_accs: dict[int, _ThinkingAccumulator] = {}

        if self._credential_pool is not None:
            key = self._credential_pool.acquire()
            client = AsyncAnthropic(
                api_key=key,
                base_url=self._base_url,
                timeout=self._default_timeout,
            )
        else:
            key = self._fallback_key
            client = self._client  # type: ignore[assignment]

        try:
            async with client.messages.stream(**kwargs) as stream:
                async for event in stream:
                    etype = getattr(event, "type", None)

                    if etype == "content_block_start":
                        block = event.content_block
                        btype = getattr(block, "type", None)
                        if btype == "tool_use":
                            had_tools = True
                            idx = event.index
                            tool_id = block.id
                            index_to_id[idx] = tool_id
                            acc = _ToolUseAccumulator(
                                index=idx, id=tool_id, name=block.name,
                            )
                            accumulators[tool_id] = acc
                            yield StreamChunk(
                                type="tool_call_start",
                                data={"id": acc.id, "name": acc.name},
                                model=model,
                            )
                        elif btype == "thinking":
                            # Start of an extended-thinking block. Accumulate
                            # text + signature silently; emit as a single
                            # reasoning chunk on block_stop.
                            thinking_accs[event.index] = _ThinkingAccumulator(
                                index=event.index,
                            )

                    elif etype == "content_block_delta":
                        delta = event.delta
                        dtype = getattr(delta, "type", None)
                        if dtype == "text_delta":
                            yield StreamChunk(type="token", data=delta.text, model=model)
                        elif dtype == "thinking_delta":
                            t_acc = thinking_accs.get(event.index)
                            fragment = getattr(delta, "thinking", "") or ""
                            if t_acc is not None:
                                t_acc.text_buffer += fragment
                            if fragment:
                                yield StreamChunk(
                                    type="reasoning_delta",
                                    data=fragment,
                                    model=model,
                                )
                        elif dtype == "signature_delta":
                            t_acc = thinking_accs.get(event.index)
                            if t_acc is not None:
                                t_acc.signature += getattr(delta, "signature", "") or ""
                        elif dtype == "input_json_delta":
                            idx = event.index
                            tool_id = index_to_id.get(idx)
                            if tool_id is None:
                                # Defensive: a delta arrived before the matching
                                # start event. Skip — there's nothing to merge into.
                                logger.debug(
                                    "input_json_delta with no prior start (index=%d)", idx,
                                )
                                continue
                            acc = accumulators.get(tool_id)
                            if acc is not None:
                                acc.arguments_buffer += delta.partial_json
                                yield StreamChunk(
                                    type="tool_call_delta",
                                    data={"id": acc.id, "fragment": delta.partial_json},
                                    model=model,
                                )

                    elif etype == "content_block_stop":
                        # Emit tool_call_complete as soon as a block closes,
                        # rather than waiting for the whole message to end. Better
                        # streaming UX with multiple tool calls per turn.
                        idx = getattr(event, "index", None)
                        if idx is None:
                            continue
                        tool_id = index_to_id.get(idx)
                        if tool_id is None or tool_id in emitted:
                            continue
                        acc = accumulators.get(tool_id)
                        if acc is not None:
                            emitted.add(tool_id)
                            yield StreamChunk(
                                type="tool_call_complete",
                                data=self._parse_accumulated(acc),
                                model=model,
                            )

                    elif etype == "message_delta":
                        # Anthropic's message_delta carries stop_reason when the
                        # response ends. Record and emit after the stream closes.
                        reason = getattr(event.delta, "stop_reason", None)
                        if reason:
                            stop_reason = reason
        except Exception as exc:
            if self._credential_pool is not None:
                status = getattr(exc, "status_code", None) or getattr(
                    getattr(exc, "response", None), "status_code", None
                )
                self._credential_pool.mark_error(key, status=status)
            raise

        # Defensive: emit any tool blocks that did not receive an explicit
        # content_block_stop (some endpoints omit it). Preserve insertion
        # order — dict iteration is insertion-ordered on Python 3.7+.
        for tool_id, acc in accumulators.items():
            if tool_id in emitted:
                continue
            yield StreamChunk(
                type="tool_call_complete",
                data=self._parse_accumulated(acc),
                model=model,
            )

        # Emit accumulated reasoning, if any. We send ONE reasoning chunk per
        # turn (concatenating multiple thinking blocks) rather than per-block
        # — consumers persist this as assistant.reasoning_text.
        if thinking_accs:
            text = "\n\n".join(
                t.text_buffer for t in thinking_accs.values() if t.text_buffer
            )
            # Only preserve signed detail blocks on the official Anthropic
            # endpoint. Third-party /anthropic gateways (MiniMax, Bedrock
            # proxies) will reject signatures on the next turn.
            details: list[dict[str, Any]] | None = None
            if not self._quirks.strip_thinking_signature:
                details = [
                    {
                        "type": "thinking",
                        "thinking": t.text_buffer,
                        "signature": t.signature,
                    }
                    for t in thinking_accs.values()
                    if t.text_buffer
                ]
            if text or details:
                yield StreamChunk(
                    type="reasoning",
                    data={"text": text, "details": details},
                    model=model,
                )

        finish_reason = self._map_stop_reason(stop_reason, has_tools=had_tools)
        yield StreamChunk(type="done", data=finish_reason, model=model)

    # -- internals -----------------------------------------------------------

    @staticmethod
    def _to_anthropic_tool(schema: ToolSchema) -> dict[str, Any]:
        """Convert a :class:`ToolSchema` to Anthropic's tool format."""
        return {
            "name": schema.name,
            "description": schema.description,
            "input_schema": schema.parameters,
        }

    @staticmethod
    def _parse_accumulated(acc: _ToolUseAccumulator) -> ToolCall:
        try:
            arguments = (
                json.loads(acc.arguments_buffer) if acc.arguments_buffer else {}
            )
        except json.JSONDecodeError:
            logger.warning(
                "Failed to parse tool_use input for %s (id=%s): %s",
                acc.name, acc.id, acc.arguments_buffer,
            )
            arguments = {"_raw": acc.arguments_buffer}
        return ToolCall(id=acc.id, name=acc.name, arguments=arguments)

    @staticmethod
    def _map_stop_reason(reason: str | None, *, has_tools: bool) -> str:
        """Translate Anthropic stop_reason to the harness's finish_reason."""
        if reason == "tool_use" or (reason is None and has_tools):
            return "tool_calls"
        if reason == "end_turn":
            return "stop"
        if reason == "max_tokens":
            return "length"
        if reason == "stop_sequence":
            return "stop"
        return reason or "stop"

    @classmethod
    def _translate_messages(
        cls,
        messages: list[Message],
        *,
        strip_thinking_signature: bool = False,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Translate OpenAI-style messages to Anthropic (system, messages).

        When ``strip_thinking_signature`` is True, any preserved
        ``reasoning_details`` on assistant messages is dropped — third-party
        /anthropic gateways reject Anthropic-proprietary signatures.
        """
        system_parts: list[str] = []
        out: list[dict[str, Any]] = []
        # Pending tool_result blocks that must be flushed as a single user
        # message before the next assistant turn.
        pending_tool_results: list[dict[str, Any]] = []

        def _flush_tool_results() -> None:
            if pending_tool_results:
                out.append({"role": "user", "content": list(pending_tool_results)})
                pending_tool_results.clear()

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")

            if role == "system":
                if content:
                    system_parts.append(content if isinstance(content, str) else str(content))
                continue

            if role == "tool":
                # Accumulate; flush right before the next assistant/user msg.
                pending_tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": msg.get("tool_call_id", ""),
                    "content": content or "",
                })
                continue

            # Any non-tool message flushes queued tool_results first.
            _flush_tool_results()

            if role == "user":
                out.append({"role": "user", "content": content or ""})
                continue

            if role == "assistant":
                # Only pass reasoning_details through on endpoints that accept
                # signatures (i.e. official Anthropic). Third-party gateways
                # must see a clean assistant message.
                reasoning = (
                    None if strip_thinking_signature
                    else msg.get("reasoning_details")
                )
                blocks = cls._assistant_to_blocks(
                    content, msg.get("tool_calls"), reasoning_details=reasoning,
                )
                if blocks:
                    out.append({"role": "assistant", "content": blocks})
                continue

            # Unknown role — skip with a warning rather than crashing.
            logger.warning("Dropping message with unknown role=%r", role)

        _flush_tool_results()

        return "\n\n".join(system_parts), out

    @staticmethod
    def _assistant_to_blocks(
        content: str | None,
        tool_calls: list[ToolCallDict] | None,
        *,
        reasoning_details: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Build an Anthropic content-block list from an assistant message.

        Thinking blocks come FIRST, per Anthropic's content-block ordering
        contract for extended-thinking multi-turn tool use.
        """
        blocks: list[dict[str, Any]] = []
        for detail in reasoning_details or []:
            # Trust the stored shape — it was captured verbatim from a prior
            # Anthropic response. Only pass through blocks we recognise.
            if isinstance(detail, dict) and detail.get("type") in (
                "thinking", "redacted_thinking",
            ):
                blocks.append(dict(detail))
        if content:
            blocks.append({"type": "text", "text": content})
        for tc in tool_calls or []:
            fn = tc.get("function", {}) or {}
            raw_args = fn.get("arguments", "") or ""
            try:
                arguments = json.loads(raw_args) if raw_args else {}
            except json.JSONDecodeError:
                arguments = {"_raw": raw_args}
            blocks.append({
                "type": "tool_use",
                "id": tc.get("id", ""),
                "name": fn.get("name", ""),
                "input": arguments,
            })
        return blocks

    @staticmethod
    def _preview_content(content: Any) -> str:
        if isinstance(content, str):
            return content[:100]
        if isinstance(content, list):
            return f"[{len(content)} blocks]"
        return str(content)[:100]
