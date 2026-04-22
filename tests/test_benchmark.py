"""Tests for BenchmarkRunner + reporter (Unit 6 — Phase 3)."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from src.core.config import load_config
from src.core.types import StreamChunk, ToolCall
from src.eval.benchmark import (
    BenchmarkRunner, EvalTask, TaskResult, _aggregate, _prepare_workspace,
    load_tasks,
)
from src.eval.judges.llm_judge import JudgeResult
from src.eval.judges.rule_checks import RuleResult
from src.eval.reporter import generate_report


# ---------------------------------------------------------------------------
# load_tasks
# ---------------------------------------------------------------------------


def _minimal_tasks_yaml(tmp_path: Path) -> Path:
    p = tmp_path / "tasks.yaml"
    p.write_text(yaml.safe_dump({
        "coding": [
            {"task": "write hello", "check": "code_executes",
             "difficulty": "easy", "timeout": 60},
        ],
        "research": [
            {"task": "compare A and B", "check": "content_quality",
             "difficulty": "medium", "timeout": 90},
        ],
        "scoring": {"rule_check_weight": 0.3, "llm_judge_weight": 0.7},
        "judge_prompt": "Score this: {task_description}",
    }), encoding="utf-8")
    return p


def test_load_tasks_all(tmp_path):
    p = _minimal_tasks_yaml(tmp_path)
    tasks, weights, jp = load_tasks(p, suite="all")
    assert len(tasks) == 2
    assert weights["rule_check_weight"] == 0.3
    assert weights["llm_judge_weight"] == 0.7
    assert "Score this" in jp


def test_load_tasks_filter_suite(tmp_path):
    p = _minimal_tasks_yaml(tmp_path)
    tasks, _, _ = load_tasks(p, suite="coding")
    assert len(tasks) == 1
    assert tasks[0].domain == "coding"


def test_load_tasks_weights_default(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text(yaml.safe_dump({
        "coding": [{"task": "t", "check": "code_executes",
                    "difficulty": "easy"}],
    }), encoding="utf-8")
    _, weights, jp = load_tasks(p, suite="all")
    assert weights["rule_check_weight"] == 0.4
    assert weights["llm_judge_weight"] == 0.6
    assert jp == ""


# ---------------------------------------------------------------------------
# Workspace setup
# ---------------------------------------------------------------------------


def test_prepare_workspace_csv(tmp_path):
    task = EvalTask(
        domain="data_analysis", task="stats", check="has_statistics",
        difficulty="easy", setup="create a sample CSV with 5 columns and 100 rows",
    )
    ws = _prepare_workspace(task, tmp_path)
    assert (ws / "data.csv").exists()
    csv = (ws / "data.csv").read_text(encoding="utf-8")
    assert csv.startswith("a,b,c,d,e")


def test_prepare_workspace_logs(tmp_path):
    task = EvalTask(
        domain="file_management", task="grep", check="errors_found",
        difficulty="medium",
        setup="create test_workspace/ with 5 log files containing mixed log levels",
    )
    ws = _prepare_workspace(task, tmp_path)
    logs = list(ws.glob("*.log"))
    assert len(logs) == 5
    assert "ERROR" in logs[0].read_text(encoding="utf-8")


def test_prepare_workspace_duplicates(tmp_path):
    task = EvalTask(
        domain="file_management", task="dedupe", check="duplicates_removed",
        difficulty="hard", setup="test_workspace with 10 files, 3 duplicates",
    )
    ws = _prepare_workspace(task, tmp_path)
    files = list(ws.glob("*.txt"))
    assert len(files) == 10


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _result(domain: str, passed: bool, score: float = 5.0,
            steps: int = 3, had_err: bool = False,
            recovered: bool = False) -> TaskResult:
    return TaskResult(
        domain=domain, difficulty="easy", task="t",
        passed=passed,
        rule=RuleResult(passed=passed, score=score, reason="r", check_name="c"),
        judge=JudgeResult(correctness=5, efficiency=5, robustness=5, reasoning=""),
        final_score=score, steps=steps, duration_s=1.0,
        had_errors=had_err, recovered=recovered,
    )


def test_aggregate_empty_to_zero_variance():
    r = _aggregate([], skill_delta=0, duration_s=0)
    assert r.total_tasks == 0
    assert r.completion_rate_overall == 0.0


def test_aggregate_completion_rates():
    results = [
        _result("coding", True), _result("coding", False),
        _result("research", True), _result("research", True),
    ]
    r = _aggregate(results, skill_delta=0, duration_s=1)
    assert r.completion_rate_overall == 0.75
    assert r.completion_rate_by_domain["coding"] == 0.5
    assert r.completion_rate_by_domain["research"] == 1.0


def test_aggregate_consistency_perfect():
    """All domains at same rate → consistency near 1.0."""
    results = [_result("coding", True), _result("research", True)]
    r = _aggregate(results, skill_delta=0, duration_s=1)
    assert r.consistency_score == pytest.approx(1.0)


def test_aggregate_consistency_poor():
    """One domain 100%, another 0% → consistency low."""
    results = [
        _result("coding", True), _result("coding", True),
        _result("research", False), _result("research", False),
    ]
    r = _aggregate(results, skill_delta=0, duration_s=1)
    assert r.consistency_score == 0.0  # max variance hit


def test_aggregate_recovery_rate():
    results = [
        _result("coding", True, had_err=True, recovered=True),
        _result("coding", False, had_err=True, recovered=False),
        _result("research", True, had_err=False),  # excluded from recovery
    ]
    r = _aggregate(results, skill_delta=0, duration_s=1)
    assert r.recovery_rate == 0.5


def test_aggregate_no_errors_recovery_one():
    results = [_result("coding", True), _result("research", True)]
    r = _aggregate(results, skill_delta=0, duration_s=1)
    assert r.recovery_rate == 1.0


# ---------------------------------------------------------------------------
# Reporter
# ---------------------------------------------------------------------------


def test_reporter_contains_summary_and_table():
    results = [
        _result("coding", True, score=8.0),
        _result("research", False, score=3.0),
    ]
    report = _aggregate(results, skill_delta=1, duration_s=12.5)
    md = generate_report(report)
    assert "# Yigent Benchmark Report" in md
    assert "Skills created: **1**" in md
    assert "| Domain |" in md
    assert "coding" in md
    assert "research" in md


# ---------------------------------------------------------------------------
# End-to-end with mocked provider
# ---------------------------------------------------------------------------


def _mock_provider_text(text: str):
    provider = MagicMock()

    async def stream_message(**kwargs) -> AsyncGenerator:
        yield StreamChunk(type="token", data=text)
        yield StreamChunk(type="done", data="stop")

    provider.stream_message = stream_message
    return provider


async def test_runner_end_to_end_single_task(tmp_path):
    # Minimal task config — one easy research task, check=content_quality
    tasks_file = tmp_path / "tasks.yaml"
    tasks_file.write_text(yaml.safe_dump({
        "research": [
            {"task": "summarize something", "check": "content_quality",
             "difficulty": "easy", "timeout": 15},
        ],
        "scoring": {"rule_check_weight": 0.4, "llm_judge_weight": 0.6},
        "judge_prompt": "{task_description} {expected_check} {execution_trace}",
    }), encoding="utf-8")

    config = load_config()
    config.provider.auxiliary = None  # no aux for this test

    # Mock both primary and aux provider resolution
    answer = (
        "This is a long answer that covers the topic in enough detail to "
        "pass the content-quality rule check which wants at least 100 "
        "characters of substantive content."
    )
    judge_resp = (
        '{"correctness": 8, "efficiency": 8, "robustness": 7, "reasoning": "ok"}'
    )
    primary = _mock_provider_text(answer)
    aux = _mock_provider_text(judge_resp)

    with patch("src.eval.benchmark.resolve_provider", return_value=primary), \
         patch("src.eval.benchmark.resolve_auxiliary", return_value=aux):
        runner = BenchmarkRunner(
            config=config, tasks_file=tasks_file, output_dir=tmp_path / "out",
        )
        report = await runner.run(suite="all")

    assert report.total_tasks == 1
    r0 = report.per_task[0]
    assert r0.domain == "research"
    # Rule check: content_quality on long answer → passes
    assert r0.rule.passed is True
    # Judge: returned positive score
    assert r0.judge.correctness == 8
    # Aggregated
    assert report.completion_rate_overall == 1.0


async def test_runner_handles_task_timeout(tmp_path):
    tasks_file = tmp_path / "tasks.yaml"
    tasks_file.write_text(yaml.safe_dump({
        "coding": [
            {"task": "infinite task", "check": "code_executes",
             "difficulty": "easy", "timeout": 1},
        ],
    }), encoding="utf-8")

    # Provider that never emits `done` — forces timeout.
    slow_provider = MagicMock()

    async def stream_message(**kwargs) -> AsyncGenerator:
        import asyncio
        while True:
            yield StreamChunk(type="token", data=".")
            await asyncio.sleep(0.1)

    slow_provider.stream_message = stream_message

    config = load_config()
    config.provider.auxiliary = None

    with patch("src.eval.benchmark.resolve_provider", return_value=slow_provider), \
         patch("src.eval.benchmark.resolve_auxiliary", return_value=None):
        runner = BenchmarkRunner(
            config=config, tasks_file=tasks_file, output_dir=tmp_path / "out",
        )
        report = await runner.run(suite="all")

    assert report.total_tasks == 1
    r0 = report.per_task[0]
    assert r0.error is not None
    assert "timeout" in r0.error.lower()
