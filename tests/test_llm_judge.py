"""Tests for LLMJudge (Unit 5 — Phase 3)."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from unittest.mock import MagicMock

import pytest

from src.core.types import Message, StreamChunk
from src.eval.judges.llm_judge import (
    DEFAULT_JUDGE_PROMPT, JudgeResult, LLMJudge, _parse_response,
)
from src.learning.trajectory import TrajectoryRecorder


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def test_parse_valid():
    text = '{"correctness": 8, "efficiency": 7, "robustness": 9, "reasoning": "ok"}'
    r = _parse_response(text)
    assert r is not None
    assert r.correctness == 8
    assert r.efficiency == 7
    assert r.robustness == 9
    assert r.reasoning == "ok"
    assert r.score == pytest.approx(8.0)


def test_parse_clamps_out_of_range():
    text = '{"correctness": 15, "efficiency": -3, "robustness": 10, "reasoning": ""}'
    r = _parse_response(text)
    assert r is not None
    assert r.correctness == 10
    assert r.efficiency == 0


def test_parse_missing_field():
    text = '{"correctness": 5, "efficiency": 5, "reasoning": "x"}'
    assert _parse_response(text) is None


def test_parse_garbage():
    assert _parse_response("not json at all") is None


def test_parse_json_in_code_fence():
    text = '```json\n{"correctness": 6, "efficiency": 7, "robustness": 8, "reasoning": "r"}\n```'
    r = _parse_response(text)
    assert r is not None
    assert r.correctness == 6


def test_judge_result_zero():
    z = JudgeResult.zero("oops")
    assert z.score == 0.0
    assert z.reasoning == "oops"


# ---------------------------------------------------------------------------
# Judge integration with mocked provider
# ---------------------------------------------------------------------------


def _mock_provider_text(response: str):
    provider = MagicMock()

    async def stream_message(**kwargs) -> AsyncGenerator:
        yield StreamChunk(type="token", data=response)
        yield StreamChunk(type="done", data="stop")

    provider.stream_message = stream_message
    return provider


def _mock_provider_seq(responses: list[str]):
    """Provider that yields a different response per stream call."""
    provider = MagicMock()
    idx = {"n": 0}

    async def stream_message(**kwargs) -> AsyncGenerator:
        n = idx["n"]
        idx["n"] += 1
        text = responses[n] if n < len(responses) else responses[-1]
        yield StreamChunk(type="token", data=text)
        yield StreamChunk(type="done", data="stop")

    provider.stream_message = stream_message
    return provider


def _simple_trajectory() -> list:
    rec = TrajectoryRecorder(session_id="s")
    rec.record_turn(
        user_msg=Message(role="user", content="do task"),
        assistant_msg=Message(role="assistant", content="done"),
    )
    return rec.turns


async def test_judge_happy_path():
    provider = _mock_provider_text(
        '{"correctness": 9, "efficiency": 8, "robustness": 7, "reasoning": "good"}'
    )
    judge = LLMJudge(provider)
    result = await judge.judge("task", "code_executes", _simple_trajectory())
    assert result.correctness == 9
    assert result.score == pytest.approx((9 + 8 + 7) / 3)


async def test_judge_retries_on_parse_failure():
    provider = _mock_provider_seq([
        "garbage not json",
        '{"correctness": 5, "efficiency": 5, "robustness": 5, "reasoning": "ok"}',
    ])
    judge = LLMJudge(provider)
    result = await judge.judge("task", "check", _simple_trajectory())
    assert result.correctness == 5


async def test_judge_gives_up_after_retry():
    provider = _mock_provider_seq(["bad1", "bad2"])
    judge = LLMJudge(provider)
    result = await judge.judge("task", "check", _simple_trajectory())
    assert result.score == 0.0
    assert "unparseable" in result.reasoning


async def test_judge_handles_aux_exception():
    provider = MagicMock()

    async def stream_message(**kwargs):
        raise RuntimeError("upstream 500")
        yield  # pragma: no cover

    provider.stream_message = stream_message
    judge = LLMJudge(provider)
    result = await judge.judge("task", "check", _simple_trajectory())
    assert result.score == 0.0
    assert "aux_error" in result.reasoning


async def test_judge_no_provider_returns_zero():
    judge = LLMJudge(None)
    result = await judge.judge("task", "check", _simple_trajectory())
    assert result.score == 0.0
    assert "no aux provider" in result.reasoning


async def test_judge_template_with_literal_braces_does_not_crash():
    """Regression: the default judge prompt in configs/eval_tasks.yaml
    contains a literal `{"correctness": N, ...}` example JSON. str.format
    would raise KeyError on those braces; the judge must substitute
    placeholders without choking on unrelated braces."""
    template = (
        "Task: {task_description}\n"
        "Check: {expected_check}\n"
        "Trace: {execution_trace}\n"
        'Return JSON: {"correctness": N, "efficiency": N, "robustness": N}'
    )
    provider = _mock_provider_text(
        '{"correctness": 7, "efficiency": 7, "robustness": 7, "reasoning": ""}'
    )
    judge = LLMJudge(provider, prompt_template=template)
    result = await judge.judge("do it", "code_executes", _simple_trajectory())
    assert result.correctness == 7
