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
_VALID_CAPABILITIES = frozenset({"search", "coding", "interpreter", "file_ops"})

_CLASSIFIER_PROMPT = """You are a task complexity classifier.

Given the user's message, output two things:
  1. strategy — pick one:
     - "direct": simple, single-capability task (one edit, one file, one
       command). The agent can start executing immediately.
     - "plan_then_execute": multi-step, multi-file, or architecturally
       ambiguous task that benefits from an explicit plan phase before any
       write or execute operations.
  2. capabilities — pick zero or more from:
     - "search": needs web/code search to gather information
     - "coding": writes or edits source code
     - "interpreter": runs code (Python REPL, shell, etc.)
     - "file_ops": reads/writes files in the workspace

Respond with ONLY a JSON object and nothing else:
{"strategy": "direct"|"plan_then_execute",
 "capabilities": ["search"|"coding"|"interpreter"|"file_ops", ...],
 "reason": "<one short sentence>"}

No preamble, no code fences, no explanations outside the JSON."""


@dataclass
class RoutingDecision:
    strategy: str                # "direct" | "plan_then_execute"
    reason: str = ""
    # Unit 10 — predicted capabilities. Empty list = no specific prediction.
    # Subset of {search, coding, interpreter, file_ops}. Used to pre-load
    # the obvious tools so the model doesn't pay a ToolSearch round-trip.
    capabilities: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.capabilities is None:
            self.capabilities = []


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
        raw_caps = parsed.get("capabilities", []) if isinstance(parsed, dict) else []
        if not isinstance(raw_caps, list):
            raw_caps = []
        capabilities = [c for c in raw_caps if c in _VALID_CAPABILITIES]
        if strategy not in _VALID_STRATEGIES:
            return RoutingDecision(
                strategy="direct",
                reason=f"default: unknown strategy {strategy!r}",
                capabilities=capabilities,
            )
        return RoutingDecision(
            strategy=strategy, reason=reason or "", capabilities=capabilities,
        )


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
