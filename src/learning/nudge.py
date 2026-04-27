"""Periodic nudge — aux LLM extracts patterns from recent activity.

Every N tool calls (``nudge_interval``, default 15), the nudge engine shows
the last M turns of the trajectory to an auxiliary LLM and asks "is there
one non-obvious pattern worth saving to L1 memory?" If yes, the pattern is
persisted via MarkdownMemoryStore; if no, nothing happens.

Design notes:
- Nudge runs OUTSIDE the agent's tool-use loop. The agent does not see or
  control nudges; they happen between its turns.
- Per-session circuit breaker: 3 consecutive aux-LLM failures disable
  nudging for the rest of the session (logged once).
- Aux-LLM output must be valid JSON or the literal string ``null``.
  Malformed output is silently dropped — we'd rather skip a nudge than
  pollute memory with half-parsed garbage.
- Nudge runs are fire-and-forget from the agent loop's perspective: if
  memory write fails, the error is logged and the loop continues. Never
  block the user on a nudge.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

from src.context.circuit_breaker import CircuitBreaker
from src.learning._aux_json import parse_aux_json
from src.learning.nudge_prompt import (
    NUDGE_SYSTEM_PROMPT, NUDGE_USER_TEMPLATE, format_turns,
)

if TYPE_CHECKING:
    from src.learning.trajectory import TurnRecord
    from src.memory.markdown_store import MarkdownMemoryStore
    from src.providers.base import LLMProvider

logger = logging.getLogger(__name__)


class NudgeReason(str, Enum):
    """Why a nudge run produced its outcome.

    StrEnum-compatible (subclasses str) so existing comparisons against
    string literals in tests and logs keep working. Audit C5 / Top10 #13.
    """
    NO_PROVIDER = "no_provider"   # aux LLM not configured
    NO_TURNS = "no_turns"         # nothing to review yet
    BREAKER_OPEN = "breaker_open" # aux failed too many times
    AUX_ERROR = "aux_error"       # this run's aux call failed
    SKIPPED = "skipped"           # parsed null / nothing worth saving
    WRITE_ERROR = "write_error"   # memory store rejected the write
    WROTE = "wrote"               # successfully persisted a topic


@dataclass
class NudgeResult:
    """Outcome of one nudge run."""

    saved: bool
    topic: str | None = None
    hook: str | None = None
    reason: NudgeReason | None = None


class NudgeEngine:
    """Periodic nudge driver.

    Typical wiring::

        engine = NudgeEngine(aux_provider, memory_store, interval=15)
        # in agent_loop, after each tool-call turn:
        tool_call_count += len(turn.tool_calls)
        if tool_call_count // interval > last_triggered_bucket:
            await engine.maybe_nudge(recorder.turns, session_id)
            last_triggered_bucket = tool_call_count // interval
    """

    def __init__(
        self,
        aux_provider: "LLMProvider | None",
        memory: "MarkdownMemoryStore",
        interval: int = 15,
        window_turns: int = 8,
        breaker_threshold: int = 3,
    ) -> None:
        self._provider = aux_provider
        self._memory = memory
        self.interval = interval
        self.window_turns = window_turns
        self._breaker = CircuitBreaker(threshold=breaker_threshold)

    @property
    def is_available(self) -> bool:
        return self._provider is not None and not self._breaker.is_open

    async def maybe_nudge(
        self,
        turns: list["TurnRecord"],
        session_id: str,
    ) -> NudgeResult:
        """Run one nudge. Returns the outcome; never raises.

        Caller is responsible for timing (when to call based on interval).
        This method does no timing logic itself so it's easy to test.
        """
        if self._provider is None:
            return NudgeResult(saved=False, reason=NudgeReason.NO_PROVIDER)
        if self._breaker.is_open:
            return NudgeResult(saved=False, reason=NudgeReason.BREAKER_OPEN)
        if not turns:
            return NudgeResult(saved=False, reason=NudgeReason.NO_TURNS)

        slice_turns = turns[-self.window_turns:]
        user_prompt = NUDGE_USER_TEMPLATE.format(
            n_turns=len(slice_turns),
            trajectory_text=format_turns(slice_turns),
        )

        try:
            response_text = await self._run_aux(user_prompt)
        except Exception as exc:  # noqa: BLE001 — aux failures must not leak
            self._breaker.record_failure()
            logger.warning(
                "Nudge aux LLM call failed (%s): %s (breaker %d/%d)",
                type(exc).__name__, exc,
                self._breaker.failures, self._breaker.threshold,
            )
            return NudgeResult(saved=False, reason=NudgeReason.AUX_ERROR)

        self._breaker.record_success()

        parsed = _parse_response(response_text)
        if parsed is None:
            return NudgeResult(saved=False, reason=NudgeReason.SKIPPED)

        topic, hook, body = parsed
        try:
            # Prefer async wrappers (Top10 #10) but fall back gracefully if a
            # caller injects a custom store without the async surface.
            if hasattr(self._memory, "awrite_topic"):
                await self._memory.awrite_topic(topic, body, title=topic.replace("-", " "))
                await self._memory.arecord_index_entry(
                    topic, topic.replace("-", " "), hook,
                )
            else:
                self._memory.write_topic(topic, body, title=topic.replace("-", " "))
                self._memory.record_index_entry(topic, topic.replace("-", " "), hook)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Nudge memory write failed for topic %r: %s",
                topic, type(exc).__name__,
            )
            return NudgeResult(saved=False, reason=NudgeReason.WRITE_ERROR)

        logger.info("Nudge saved topic %r (session=%s)", topic, session_id)
        return NudgeResult(saved=True, topic=topic, hook=hook, reason=NudgeReason.WROTE)

    async def _run_aux(self, user_prompt: str) -> str:
        """Call the aux LLM (non-streaming — accumulate token chunks)."""
        assert self._provider is not None
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": NUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        text = ""
        async for chunk in self._provider.stream_message(
            messages=messages,  # type: ignore[arg-type]
            temperature=0.0,
        ):
            if chunk.type == "token":
                text += chunk.data
            elif chunk.type == "done":
                break
        return text


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

# Tighter regex for nudge: this prompt produces simple flat objects, so we
# can use the no-nesting variant and avoid catastrophic regex backtracking
# on long prose responses.
_JSON_BLOCK_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _parse_response(text: str) -> tuple[str, str, str] | None:
    """Parse the aux LLM response into (topic, hook, body) or None.

    Returns None when the JSON is missing, malformed, or doesn't have all
    three required string fields.
    """
    obj = parse_aux_json(text, block_re=_JSON_BLOCK_RE)
    if obj is None:
        return None
    topic = obj.get("topic")
    hook = obj.get("hook")
    body = obj.get("body")
    if not (isinstance(topic, str) and topic.strip()
            and isinstance(hook, str) and hook.strip()
            and isinstance(body, str) and body.strip()):
        return None
    return topic.strip(), hook.strip(), body.strip()
