"""Capability router — lightweight intent classifier.

At the start of each new user turn, asks an auxiliary LLM whether the
request is simple (``direct``) or complex enough to warrant entering plan
mode (``plan_then_execute``). Defaults to ``direct`` on any error — this
is the safe default because it preserves the existing Phase-1 behavior.

Kept deliberately minimal:
  - One-shot classifier per user turn (not per iteration)
  - No tools, no state, no memory
  - Aux LLM runs with temperature=0 for determinism
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.providers.base import LLMProvider

logger = logging.getLogger(__name__)

_VALID_STRATEGIES = frozenset({"direct", "plan_then_execute"})

_CLASSIFIER_PROMPT = """You are a task complexity classifier.

Given the user's message, pick one strategy:
  - "direct": simple, single-capability task (one edit, one file, one
    command). The agent can start executing immediately.
  - "plan_then_execute": multi-step, multi-file, or architecturally
    ambiguous task that benefits from an explicit plan phase before any
    write or execute operations.

Respond with ONLY a JSON object and nothing else:
{"strategy": "direct"|"plan_then_execute", "reason": "<one short sentence>"}

No preamble, no code fences, no explanations outside the JSON."""


@dataclass
class RoutingDecision:
    strategy: str                # "direct" | "plan_then_execute"
    reason: str = ""


class CapabilityRouter:
    """Classifies one user message into an execution strategy."""

    def __init__(self, aux_provider: "LLMProvider | None") -> None:
        self._aux = aux_provider

    async def classify(self, user_message: str) -> RoutingDecision:
        if self._aux is None:
            return RoutingDecision(strategy="direct", reason="no aux provider")

        messages = [
            {"role": "system", "content": _CLASSIFIER_PROMPT},
            {"role": "user", "content": user_message},
        ]
        buffer = ""
        try:
            async for chunk in self._aux.stream_message(
                messages=messages, temperature=0.0
            ):
                if chunk.type == "token":
                    buffer += chunk.data
                elif chunk.type == "done":
                    break
        except Exception as exc:
            logger.warning("Classifier call failed: %s — defaulting to direct", exc)
            return RoutingDecision(strategy="direct", reason=f"classifier error: {exc}")

        payload = _strip_fences(buffer.strip())
        try:
            parsed = json.loads(payload)
        except (json.JSONDecodeError, AttributeError, TypeError):
            logger.warning("Classifier output unparseable: %r", buffer[:200])
            return RoutingDecision(strategy="direct", reason="unparseable classifier output")

        strategy = parsed.get("strategy") if isinstance(parsed, dict) else None
        reason = parsed.get("reason", "") if isinstance(parsed, dict) else ""
        if strategy not in _VALID_STRATEGIES:
            return RoutingDecision(
                strategy="direct",
                reason=f"default: unknown strategy {strategy!r}",
            )
        return RoutingDecision(strategy=strategy, reason=reason or "")


def _strip_fences(text: str) -> str:
    """Remove leading/trailing markdown code fences if present."""
    if not text.startswith("```"):
        return text
    # Strip first line (```json or just ```) and trailing ```
    lines = text.splitlines()
    if lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()
