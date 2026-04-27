"""LLM-as-Judge — aux LLM rates agent runs on correctness/efficiency/robustness.

The judge prompt template lives in ``configs/eval_tasks.yaml`` under
``judge_prompt:``. It's formatted with the task description, expected
check, and execution trace, then handed to the aux provider with
temperature=0.

The response must be JSON with three integers (0-10) and a reason. One
retry on parse failure — after that, the judge gives up and returns a
JudgeResult with score=0 and an error reason. Benchmark scoring then
falls back to the rule-check channel alone.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.learning._aux_json import parse_aux_json

if TYPE_CHECKING:
    from src.learning.trajectory import TurnRecord
    from src.providers.base import LLMProvider

logger = logging.getLogger(__name__)


@dataclass
class JudgeResult:
    correctness: int   # 0-10
    efficiency: int    # 0-10
    robustness: int    # 0-10
    reasoning: str

    @property
    def score(self) -> float:
        """Aggregate 0-10 score: mean of the three axes."""
        return (self.correctness + self.efficiency + self.robustness) / 3.0

    @classmethod
    def zero(cls, reason: str) -> "JudgeResult":
        return cls(correctness=0, efficiency=0, robustness=0, reasoning=reason)


DEFAULT_JUDGE_PROMPT = """\
You are evaluating an AI agent's performance on a task.

Task: {task_description}
Expected outcome: {expected_check}

Agent's execution trace:
{execution_trace}

Score the agent on:
1. Correctness (0-10): Did it achieve the goal?
2. Efficiency (0-10): Did it use a reasonable number of steps?
3. Robustness (0-10): Did it handle errors gracefully?

Return JSON: {{"correctness": N, "efficiency": N, "robustness": N, \
"reasoning": "..."}}
"""


class LLMJudge:
    def __init__(
        self,
        aux_provider: "LLMProvider | None",
        prompt_template: str = DEFAULT_JUDGE_PROMPT,
    ) -> None:
        self._provider = aux_provider
        self._template = prompt_template

    async def judge(
        self,
        task_description: str,
        expected_check: str,
        trajectory: list["TurnRecord"],
    ) -> JudgeResult:
        if self._provider is None:
            return JudgeResult.zero("no aux provider")
        trace = _format_trajectory(trajectory)
        # Literal placeholder substitution instead of str.format to avoid
        # KeyError on the `{"correctness": N, ...}` example JSON that
        # prompt templates naturally contain.
        prompt = (
            self._template
            .replace("{task_description}", task_description)
            .replace("{expected_check}", expected_check)
            .replace("{execution_trace}", trace)
        )

        # Bump temperature on retry: at temp=0, an unparseable response
        # tends to be deterministically reproduced on the second try, so the
        # retry buys nothing. A small bump (0.3) introduces variance without
        # turning the judge into a coin flip. Audit B11 / Top10 #18.
        for attempt, temperature in ((1, 0.0), (2, 0.3)):
            try:
                response = await self._run_aux(prompt, temperature=temperature)
            except Exception as exc:  # noqa: BLE001
                # Log only the exception class (audit A4) — judge prompt may
                # quote sensitive fragments of the trajectory.
                logger.warning(
                    "Judge aux LLM failed (attempt %d, %s)",
                    attempt, type(exc).__name__,
                )
                if attempt == 2:
                    return JudgeResult.zero(f"aux_error: {type(exc).__name__}")
                continue
            parsed = _parse_response(response)
            if parsed is not None:
                return parsed
            logger.info(
                "Judge response unparseable (attempt %d, len=%d): %r",
                attempt, len(response), response[:200],
            )

        return JudgeResult.zero("unparseable response after retry")

    async def _run_aux(self, prompt: str, temperature: float = 0.0) -> str:
        assert self._provider is not None
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        text = ""
        async for chunk in self._provider.stream_message(
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
        ):
            if chunk.type == "token":
                text += chunk.data
            elif chunk.type == "done":
                break
        return text


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_trajectory(trajectory: list["TurnRecord"]) -> str:
    """Compact representation for judge consumption."""
    lines: list[str] = []
    for t in trajectory:
        if t.user_msg is not None:
            content = (t.user_msg.get("content") or "").strip()
            if content:
                lines.append(f"USER: {content}")
        asst = (t.assistant_msg.get("content") or "").strip()
        if asst:
            lines.append(f"AGENT: {asst[:300]}")
        for tc in t.tool_calls:
            args = str(tc.arguments)[:200]
            lines.append(f"TOOL_CALL: {tc.name}({args})")
        for tr in t.tool_results:
            preview = (tr.content or "").replace("\n", " ")[:200]
            marker = " [ERROR]" if tr.is_error else ""
            lines.append(f"TOOL_RESULT{marker}: {preview}")
    return "\n".join(lines) if lines else "(empty)"


def _parse_response(text: str) -> JudgeResult | None:
    obj = parse_aux_json(text)
    if obj is None:
        return None

    def _as_score(v: Any) -> int | None:
        try:
            n = int(v)
        except (TypeError, ValueError):
            return None
        return max(0, min(10, n))

    c = _as_score(obj.get("correctness"))
    e = _as_score(obj.get("efficiency"))
    r = _as_score(obj.get("robustness"))
    if c is None or e is None or r is None:
        return None
    reasoning = obj.get("reasoning")
    if not isinstance(reasoning, str):
        reasoning = ""
    return JudgeResult(correctness=c, efficiency=e, robustness=r, reasoning=reasoning)
