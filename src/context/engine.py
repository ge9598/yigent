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
    from src.safety.hook_system import HookSystem

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


def _safe_split_index(msgs: list["Message"], desired: int) -> int:
    """Return a split index that doesn't orphan a tool_result from its tool_use.

    A ``role="tool"`` message must always be preceded (transitively, through
    consecutive ``role="tool"`` siblings) by an ``role="assistant"`` message
    whose ``tool_calls`` list contains the matching id. If ``desired`` falls
    in the middle of such a group, we walk forward until the split lands
    either at an assistant boundary or past all tool messages.

    Walking *forward* (not backward) is deliberate: the late slice is what we
    keep verbatim, so losing one extra old turn is acceptable; orphaning a
    tool_result would cause Anthropic API 400 errors.

    If the walk would exhaust ``msgs`` entirely, return ``len(msgs)`` so the
    caller can detect the degenerate case via ``split >= len(msgs)`` and skip
    / fall back.
    """
    if desired <= 0 or desired >= len(msgs):
        return desired
    idx = desired
    while idx < len(msgs) and msgs[idx].get("role") == "tool":
        idx += 1
    return idx


@dataclass
class CompressionEngine:
    """Five-layer compression engine. See module docstring."""

    auxiliary_provider: LLMProvider | None = None
    layer3_breaker: CircuitBreaker = field(default_factory=CircuitBreaker)
    layer4_breaker: CircuitBreaker = field(default_factory=CircuitBreaker)
    tool_result_cap: int = 3000
    layer4_keep_turns: int = 5
    layer5_keep_turns: int = 4
    hook_system: "HookSystem | None" = None
    # Compression cursor — index into the conversation marking the boundary
    # between "already summarized" (left) and "original" (right). Persisted
    # across compress() calls within a session so layer 3 doesn't keep
    # re-summarizing the same head turns. ARCHITECTURE.md §I makes this an
    # explicit slot rather than a per-call local.
    compression_cursor: int = 0

    async def compress(
        self,
        conversation: list[Message],
        target_tokens: int,
        on_layer: Callable[[int], Awaitable[None]] | None = None,
    ) -> list[Message]:
        """Apply layers 1→5 in order, stopping as soon as fits in ``target_tokens``."""
        msgs = list(conversation)
        before_tokens = estimate_tokens(msgs)

        if before_tokens <= target_tokens:
            return msgs

        if self.hook_system is not None:
            await self.hook_system.fire(
                "pre_compression",
                conversation=msgs, before_tokens=before_tokens,
                target_tokens=target_tokens,
            )
        layers_run: list[int] = []

        async def _finish(result_msgs: list[Message]) -> list[Message]:
            if self.hook_system is not None:
                await self.hook_system.fire(
                    "post_compression",
                    conversation=result_msgs,
                    before_tokens=before_tokens,
                    after_tokens=estimate_tokens(result_msgs),
                    layers_run=list(layers_run),
                )
            return result_msgs

        # Layer 1 — truncate large tool results
        msgs = self._layer1_truncate_tool_results(msgs)
        layers_run.append(1)
        if on_layer:
            await on_layer(1)
        if estimate_tokens(msgs) <= target_tokens:
            return await _finish(msgs)

        # Layer 2 — dedup repeated file reads
        msgs = self._layer2_dedup_file_reads(msgs)
        layers_run.append(2)
        if on_layer:
            await on_layer(2)
        if estimate_tokens(msgs) <= target_tokens:
            return await _finish(msgs)

        # Layer 3 — summarize earliest 1/3 (LLM, breaker-protected)
        if not self.layer3_breaker.is_open and self.auxiliary_provider is not None:
            try:
                msgs = await self._layer3_summarize_early(msgs)
                self.layer3_breaker.record_success()
                layers_run.append(3)
                if on_layer:
                    await on_layer(3)
                if estimate_tokens(msgs) <= target_tokens:
                    return await _finish(msgs)
            except Exception as exc:  # noqa: BLE001 — breaker pattern
                logger.warning("Compression layer 3 failed: %s", exc)
                self.layer3_breaker.record_failure()

        # Layer 4 — full rewrite, keep last N turns
        if not self.layer4_breaker.is_open and self.auxiliary_provider is not None:
            try:
                msgs = await self._layer4_rewrite(msgs)
                self.layer4_breaker.record_success()
                layers_run.append(4)
                if on_layer:
                    await on_layer(4)
                if estimate_tokens(msgs) <= target_tokens:
                    return await _finish(msgs)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Compression layer 4 failed: %s", exc)
                self.layer4_breaker.record_failure()

        # Layer 5 — hard truncate
        msgs = self._layer5_hard_truncate(msgs)
        layers_run.append(5)
        if on_layer:
            await on_layer(5)
        return await _finish(msgs)

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

    @staticmethod
    def _is_read_shaped_tool(name: str) -> bool:
        """True if the tool's name suggests it returns content the model
        might re-fetch (read_file, search_files, MCP read_*/get_*).

        Used to widen Layer-2 dedup beyond the original {read_file, list_dir}
        whitelist so MCP tools and search_files also benefit.
        """
        if name in ("read_file", "list_dir", "search_files"):
            return True
        # MCP tools follow `{server}__{tool}` naming; check the suffix part.
        suffix = name.rsplit("__", 1)[-1]
        return suffix.startswith(("read_", "get_", "fetch_", "list_", "show_"))

    def _layer2_dedup_file_reads(self, msgs: list[Message]) -> list[Message]:
        """If the same read-shaped tool is called twice with identical RESULT
        content, keep only the latest result; replace earlier ones with a
        stub. Dedup runs by content hash, not by call args, so two different
        calls that happen to return the same bytes still dedup.
        """
        import hashlib
        # content_hash → latest index in msgs holding it
        latest_idx_by_hash: dict[str, int] = {}

        for i, m in enumerate(msgs):
            if m.get("role") != "tool":
                continue
            tool_name = m.get("name") or ""
            if not self._is_read_shaped_tool(tool_name):
                continue
            content = m.get("content") or ""
            if not isinstance(content, str) or len(content) < 64:
                # Too small to bother deduping — savings would be negligible.
                continue
            # Hash by tool name + content so different tools' identical-text
            # outputs (e.g. one's a JSON pretty-print of the other) stay
            # distinct.
            key = hashlib.sha256(
                f"{tool_name}::{content}".encode("utf-8", errors="replace")
            ).hexdigest()
            older = latest_idx_by_hash.get(key)
            if older is not None:
                msgs[older] = dict(msgs[older])  # type: ignore[index]
                msgs[older]["content"] = _DEDUP_MARKER  # type: ignore[index]
            latest_idx_by_hash[key] = i
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
        """Summarize the first 1/3 of (non-system, non-already-compressed) turns.

        Honors ``compression_cursor``: head turns at-or-before the cursor are
        treated as already-summarized and skipped, so we don't re-summarize.
        Updates the cursor after summarizing.
        """
        # Skip leading system messages, already-compressed summaries, and
        # anything before the persistent cursor.
        head_skip = 0
        for i, m in enumerate(msgs):
            if (
                m.get("role") == "system"
                or m.get("compressed")
                or i < self.compression_cursor
            ):
                head_skip += 1
            else:
                break

        body = msgs[head_skip:]
        if len(body) < 6:
            return msgs  # not enough to summarize

        cut = max(2, len(body) // 3)
        # Don't split a tool_use/tool_result pair — would orphan the tool_result
        # on the "late" side and cause Anthropic 400s.
        cut = _safe_split_index(body, cut)
        if cut >= len(body):
            return msgs  # degenerate: entire body is tool chain, skip layer
        early = body[:cut]
        late = body[cut:]

        summary_text = await self._llm_summarize(early)
        summary_msg: Message = {
            "role": "system",
            "content": f"[Earlier conversation summary]\n{summary_text}",
            "compressed": True,  # type: ignore[typeddict-unknown-key]
        }
        # Advance the cursor past the new summary boundary so subsequent
        # compress() calls don't re-summarize this region.
        self.compression_cursor = head_skip + 1
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

        split = len(body) - self.layer4_keep_turns
        # Walk forward so the boundary doesn't orphan tool_results.
        split = _safe_split_index(body, split)
        if split >= len(body):
            return msgs  # degenerate: tool chain extends into the tail
        keep = body[split:]
        drop = body[:split]

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
        if len(body) <= self.layer5_keep_turns:
            return head + body
        split = len(body) - self.layer5_keep_turns
        split = _safe_split_index(body, split)
        if split >= len(body):
            # Degenerate: tail is entirely tool_results. Prefer an over-budget
            # but protocol-valid conversation over a 400 from orphan results.
            logger.warning(
                "Layer 5 could not find a safe split — keeping full body"
            )
            return head + body
        return head + body[split:]

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

        # Unit 9 — structured summary template (Hermes pattern). Forced section
        # headers (Goal / Constraints / Progress / Key Decisions / Relevant
        # Files / Next Steps / Critical Context) preserve recall quality.
        from src.context.summary_template import (
            SUMMARY_SYSTEM_PROMPT, render_user_prompt,
        )
        prompt: list[Message] = [
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": render_user_prompt(transcript)},
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
