"""Reasoning-content extraction for OpenAI-protocol providers.

The OpenAI chat-completions protocol has no native "thinking" block type,
but three community conventions have emerged for exposing chain-of-thought:

  Form #2 (``delta.reasoning_content`` field)
      DeepSeek-R1, OpenAI o1, GLM-Z1. Reasoning is its own delta field,
      separate from ``delta.content``. Extract verbatim; never echo back.

  Form #3 (inline ``<think>...</think>`` tags inside ``delta.content``)
      MiniMax M2 via /v1, Qwen QwQ, some DeepSeek builds. The tags split
      arbitrarily across streaming chunks — token boundaries don't respect
      tag boundaries. Needs a 3-state FSM with cross-chunk buffering.

This module provides:
  * :func:`extract_reasoning_content` — pulls Form #2 out of a chunk dict.
  * :class:`ThinkTagStripper` — streaming FSM for Form #3.

Both return ``(text_for_user, reasoning_text)`` pairs so providers can
forward the user-facing token unchanged while accumulating a separate
``reasoning_text`` buffer for the assistant message.

Ported from CCR's ``extrathinktag.transformer.ts`` (3-state logic) and
Hermes's ``anthropic_adapter`` (Form #2 mirror). ``<think>`` detection is
case-insensitive because Qwen and some local models emit upper-case tags.
"""

from __future__ import annotations

from dataclasses import dataclass, field


_OPEN_TAG = "<think>"
_CLOSE_TAG = "</think>"
# Longest partial prefix of either tag we might carry across chunks.
# "</think" is 7 chars — that's the carry limit.
_MAX_CARRY = max(len(_OPEN_TAG), len(_CLOSE_TAG)) - 1


def extract_reasoning_content(delta_reasoning: str | None) -> str:
    """Return a reasoning fragment from an OpenAI ``delta.reasoning_content``.

    Trivial wrapper — exists so callers don't hard-code the field name and
    so we can extend it later (e.g. Gemini's ``thought`` field).
    """
    return delta_reasoning or ""


@dataclass
class ThinkTagStripper:
    """Streaming FSM that splits ``<think>...</think>`` out of text deltas.

    State machine:

        SEARCHING  ── "<think>" found ──>  THINKING
        THINKING   ── "</think>" found ──> SEARCHING

    Tags may be split across chunks. We hold back a small tail from each
    input so a tag boundary can complete on the next chunk.

    Usage::

        stripper = ThinkTagStripper()
        for delta_content in stream:
            user_text, reasoning_text = stripper.feed(delta_content)
            # emit user_text to UI, accumulate reasoning_text separately
        user_text, reasoning_text = stripper.finish()  # flush carry

    Case handling: we normalize to lower-case only for tag detection; the
    original casing of user-visible text is preserved. Reasoning text is
    returned with the tag stripped (inner content only).
    """

    state: str = "searching"  # "searching" | "thinking"
    _carry: str = field(default="", repr=False)

    def feed(self, chunk: str) -> tuple[str, str]:
        """Process one streaming chunk. Returns ``(user_visible, reasoning)``."""
        if not chunk:
            return "", ""
        buf = self._carry + chunk
        self._carry = ""
        user_out: list[str] = []
        reason_out: list[str] = []

        while buf:
            if self.state == "searching":
                # Look for the opening tag (case-insensitive).
                idx = buf.lower().find(_OPEN_TAG)
                if idx == -1:
                    # No full tag in this buffer. Find the longest suffix
                    # that COULD be the start of "<think>" and carry it.
                    carry_len = _longest_tag_prefix_suffix(buf)
                    if carry_len:
                        user_out.append(buf[:-carry_len])
                        self._carry = buf[-carry_len:]
                    else:
                        user_out.append(buf)
                    break
                # Emit text before the tag as user-visible, then enter THINKING.
                user_out.append(buf[:idx])
                buf = buf[idx + len(_OPEN_TAG):]
                self.state = "thinking"
            else:  # thinking
                idx = buf.lower().find(_CLOSE_TAG)
                if idx == -1:
                    carry_len = _longest_tag_prefix_suffix(buf)
                    if carry_len:
                        reason_out.append(buf[:-carry_len])
                        self._carry = buf[-carry_len:]
                    else:
                        reason_out.append(buf)
                    break
                reason_out.append(buf[:idx])
                buf = buf[idx + len(_CLOSE_TAG):]
                self.state = "searching"

        return "".join(user_out), "".join(reason_out)

    def finish(self) -> tuple[str, str]:
        """Flush any held-back carry at end of stream.

        If we end mid-``<think>`` block (malformed output), the trailing
        reasoning text is still returned — better than losing it.
        """
        if not self._carry:
            return "", ""
        leftover = self._carry
        self._carry = ""
        if self.state == "searching":
            return leftover, ""
        return "", leftover


def _longest_tag_prefix_suffix(buf: str) -> int:
    """Longest suffix of ``buf`` that is a non-empty prefix of either tag.

    Returns 0 if no such suffix exists. Example::

        "text<thi"   → 4   ("<thi" is a prefix of "<think>")
        "text</th"   → 4   ("</th" is a prefix of "</think>")
        "hello"      → 0   (no tag-prefix suffix)
        "x<"         → 1   ("<" is a prefix of both tags)

    Bounded by ``_MAX_CARRY`` — we never carry more than a tag's length minus
    one, because a full tag would already have been matched by ``find``.
    """
    low = buf.lower()
    limit = min(_MAX_CARRY, len(low))
    for k in range(limit, 0, -1):
        tail = low[-k:]
        if _OPEN_TAG.startswith(tail) or _CLOSE_TAG.startswith(tail):
            return k
    return 0
