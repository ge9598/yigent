"""Five-layer progressive context compression.

Layered from cheap & lossless to expensive & lossy:

    Layer 1 — truncate large tool_result strings (free, mostly lossless)
    Layer 2 — dedup repeated reads of the same file (free, lossless)
    Layer 3 — aux-LLM summarize the earliest 1/3 of turns
    Layer 4 — aux-LLM full rewrite, keep only the last 5 turns verbatim
    Layer 5 — hard truncate to the last 4 turns (emergency)

The engine takes a target token budget and applies layers in order until the
conversation fits — or layer 5 forces it to fit. Layers 3 and 4 wrap their
LLM calls through per-layer ``CircuitBreaker``s; if a layer's breaker is
open, that layer is skipped.

A ``compression_cursor`` is maintained on AgentState (Phase 2b); within one
compress() call the cursor lives only as a local variable. Summary messages
get a ``compressed: True`` marker so future passes don't re-summarize them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Awaitable, Callable

import tiktoken

from src.context.circuit_breaker import CircuitBreaker

if TYPE_CHECKING:
    from src.core.types import Message
    from src.providers.base import LLMProvider

logger = logging.getLogger(__name__)


# Tokenizer is shared & cached. cl100k_base is OpenAI's GPT-4 / GPT-3.5
# tokenizer; it's a reasonable approximation for any modern provider. We
# only need rough counts for budget decisions, not exact billing accuracy.
_TOKENIZER = tiktoken.get_encoding("cl100k_base")


def estimate_tokens(messages: list[Message] | str) -> int:
    """Rough token count. Adds ~4 tokens of overhead per message envelope."""
    if isinstance(messages, str):
        return len(_TOKENIZER.encode(messages))
    total = 0
    for msg in messages:
        total += 4  # role / separators
        content = msg.get("content")
        if isinstance(content, str):
            total += len(_TOKENIZER.encode(content))
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    text = block.get("text") or block.get("content") or ""
                    if isinstance(text, str):
                        total += len(_TOKENIZER.encode(text))
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {}) or {}
            args = fn.get("arguments", "") or ""
            if isinstance(args, str):
                total += len(_TOKENIZER.encode(args))
    return total


_TRUNCATION_MARKER = "\n…[truncated]"
_DEDUP_MARKER = "[duplicate file read elided]"


@dataclass
class CompressionEngine:
    """Five-layer compression engine. See module docstring."""

    auxiliary_provider: LLMProvider | None = None
    layer3_breaker: CircuitBreaker = field(default_factory=CircuitBreaker)
    layer4_breaker: CircuitBreaker = field(default_factory=CircuitBreaker)
    tool_result_cap: int = 3000
    layer4_keep_turns: int = 5
    layer5_keep_turns: int = 4

    async def compress(
        self,
        conversation: list[Message],
        target_tokens: int,
        on_layer: Callable[[int], Awaitable[None]] | None = None,
    ) -> list[Message]:
        """Apply layers 1→5 in order, stopping as soon as fits in ``target_tokens``."""
        msgs = list(conversation)

        if estimate_tokens(msgs) <= target_tokens:
            return msgs

        # Layer 1 — truncate large tool results
        msgs = self._layer1_truncate_tool_results(msgs)
        if on_layer:
            await on_layer(1)
        if estimate_tokens(msgs) <= target_tokens:
            return msgs

        # Layer 2 — dedup repeated file reads
        msgs = self._layer2_dedup_file_reads(msgs)
        if on_layer:
            await on_layer(2)
        if estimate_tokens(msgs) <= target_tokens:
            return msgs

        # Layer 3 — summarize earliest 1/3 (LLM, breaker-protected)
        if not self.layer3_breaker.is_open and self.auxiliary_provider is not None:
            try:
                msgs = await self._layer3_summarize_early(msgs)
                self.layer3_breaker.record_success()
                if on_layer:
                    await on_layer(3)
                if estimate_tokens(msgs) <= target_tokens:
                    return msgs
            except Exception as exc:  # noqa: BLE001 — breaker pattern
                logger.warning("Compression layer 3 failed: %s", exc)
                self.layer3_breaker.record_failure()

        # Layer 4 — full rewrite, keep last N turns
        if not self.layer4_breaker.is_open and self.auxiliary_provider is not None:
            try:
                msgs = await self._layer4_rewrite(msgs)
                self.layer4_breaker.record_success()
                if on_layer:
                    await on_layer(4)
                if estimate_tokens(msgs) <= target_tokens:
                    return msgs
            except Exception as exc:  # noqa: BLE001
                logger.warning("Compression layer 4 failed: %s", exc)
                self.layer4_breaker.record_failure()

        # Layer 5 — hard truncate
        msgs = self._layer5_hard_truncate(msgs)
        if on_layer:
            await on_layer(5)
        return msgs

    # -- layers --------------------------------------------------------------

    def _layer1_truncate_tool_results(self, msgs: list[Message]) -> list[Message]:
        out: list[Message] = []
        for m in msgs:
            if m.get("role") == "tool":
                content = m.get("content") or ""
                if isinstance(content, str) and len(content) > self.tool_result_cap:
                    new = dict(m)
                    new["content"] = content[: self.tool_result_cap] + _TRUNCATION_MARKER
                    out.append(new)  # type: ignore[arg-type]
                    continue
            out.append(m)
        return out

    def _layer2_dedup_file_reads(self, msgs: list[Message]) -> list[Message]:
        """If the same file is read multiple times with identical content,
        keep only the latest read; replace earlier ones with a stub."""
        # Map "name:path-or-args" → list of indices into msgs that hold a
        # tool_result for that read. We use the *prior* assistant tool_call
        # to determine the args.
        last_seen_content: dict[str, str] = {}
        latest_idx: dict[str, int] = {}

        # Build (key, index) list for all read_file tool_results.
        for i, m in enumerate(msgs):
            if m.get("role") != "tool":
                continue
            tool_name = m.get("name") or ""
            if tool_name not in ("read_file", "list_dir"):
                continue
            # Key by tool_call_id-prefix's preceding assistant call args.
            tcid = m.get("tool_call_id", "")
            key = self._lookup_call_args(msgs, tcid, tool_name)
            if key is None:
                continue
            content = m.get("content") or ""
            if not isinstance(content, str):
                continue
            if last_seen_content.get(key) == content:
                # Same content as previous read — older one becomes stub.
                older = latest_idx.get(key)
                if older is not None:
                    msgs[older] = dict(msgs[older])  # type: ignore[index]
                    msgs[older]["content"] = _DEDUP_MARKER  # type: ignore[index]
            last_seen_content[key] = content
            latest_idx[key] = i
        return msgs

    @staticmethod
    def _lookup_call_args(
        msgs: list[Message], tool_call_id: str, tool_name: str
    ) -> str | None:
        for m in msgs:
            if m.get("role") != "assistant":
                continue
            for tc in m.get("tool_calls") or []:
                if tc.get("id") == tool_call_id:
                    fn = tc.get("function", {}) or {}
                    return f"{tool_name}:{fn.get('arguments', '')}"
        return None

    async def _layer3_summarize_early(self, msgs: list[Message]) -> list[Message]:
        """Summarize the first 1/3 of (non-system, non-already-compressed) turns."""
        # Skip any leading system messages and already-compressed summaries.
        head_skip = 0
        for m in msgs:
            if m.get("role") == "system" or m.get("compressed"):
                head_skip += 1
            else:
                break

        body = msgs[head_skip:]
        if len(body) < 6:
            return msgs  # not enough to summarize

        cut = max(2, len(body) // 3)
        early = body[:cut]
        late = body[cut:]

        summary_text = await self._llm_summarize(early)
        summary_msg: Message = {
            "role": "system",
            "content": f"[Earlier conversation summary]\n{summary_text}",
            "compressed": True,  # type: ignore[typeddict-unknown-key]
        }
        return msgs[:head_skip] + [summary_msg] + late

    async def _layer4_rewrite(self, msgs: list[Message]) -> list[Message]:
        """Replace everything except the last N turns with a single summary."""
        head_skip = 0
        for m in msgs:
            if m.get("role") == "system":
                head_skip += 1
            else:
                break

        body = msgs[head_skip:]
        if len(body) <= self.layer4_keep_turns:
            return msgs

        keep = body[-self.layer4_keep_turns :]
        drop = body[: -self.layer4_keep_turns]

        summary_text = await self._llm_summarize(drop)
        summary_msg: Message = {
            "role": "system",
            "content": f"[Conversation summary up to this point]\n{summary_text}",
            "compressed": True,  # type: ignore[typeddict-unknown-key]
        }
        return msgs[:head_skip] + [summary_msg] + keep

    def _layer5_hard_truncate(self, msgs: list[Message]) -> list[Message]:
        """Last-resort truncation. Keep system msgs + last N turns."""
        head: list[Message] = []
        body: list[Message] = []
        for m in msgs:
            if m.get("role") == "system":
                head.append(m)
            else:
                body.append(m)
        return head + body[-self.layer5_keep_turns :]

    # -- aux LLM call --------------------------------------------------------

    async def _llm_summarize(self, msgs: list[Message]) -> str:
        if self.auxiliary_provider is None:
            raise RuntimeError("No auxiliary provider configured for compression")

        # Render messages as a flat transcript for the summarizer.
        lines: list[str] = []
        for m in msgs:
            role = m.get("role", "?")
            content = m.get("content") or ""
            if isinstance(content, str):
                lines.append(f"[{role}] {content}")
            else:
                lines.append(f"[{role}] (structured content)")
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function", {}) or {}
                lines.append(f"  → tool {fn.get('name')}({fn.get('arguments', '')})")
        transcript = "\n".join(lines)

        prompt: list[Message] = [
            {
                "role": "system",
                "content": (
                    "You compress agent conversations. Produce a faithful, "
                    "compact summary that preserves: (1) user goals, "
                    "(2) decisions made, (3) tools called and key results, "
                    "(4) open questions. Keep it under 500 words."
                ),
            },
            {
                "role": "user",
                "content": f"Summarize this transcript:\n\n{transcript}",
            },
        ]

        text = ""
        async for chunk in self.auxiliary_provider.stream_message(
            messages=prompt, temperature=0.0,
        ):
            if chunk.type == "token":
                text += chunk.data
            elif chunk.type == "done":
                break
        return text.strip() or "(empty summary)"
