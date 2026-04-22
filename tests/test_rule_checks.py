"""Tests for rule_checks (Unit 5 — Phase 3)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.core.types import Message, ToolCall, ToolResult
from src.eval.judges.rule_checks import RuleChecker, RuleResult
from src.learning.trajectory import TrajectoryRecorder


def _trajectory(
    *,
    final_answer: str = "",
    tool_calls: list[tuple[str, dict, str, bool]] | None = None,
    user: str = "task",
) -> list:
    """Build a test trajectory. tool_calls is list of (name, args, result_content, is_error)."""
    rec = TrajectoryRecorder(session_id="t")
    rec.record_turn(
        user_msg=Message(role="user", content=user),
        assistant_msg=Message(role="assistant", content=None),
    )
    for i, (name, args, content, is_err) in enumerate(tool_calls or []):
        rec.record_turn(
            assistant_msg=Message(role="assistant", content=None),
            tool_calls=[ToolCall(id=f"c{i}", name=name, arguments=args)],
            tool_results=[ToolResult(tool_call_id=f"c{i}", name=name,
                                     content=content, is_error=is_err)],
        )
    if final_answer:
        rec.record_turn(
            assistant_msg=Message(role="assistant", content=final_answer),
        )
    return rec.turns


def test_unknown_check_returns_zero(tmp_path):
    result = RuleChecker().check("bogus", [], tmp_path)
    assert result.passed is False
    assert result.score == 0.0
    assert "unknown" in result.reason


def test_code_executes_passes_when_python_repl_runs(tmp_path):
    traj = _trajectory(
        tool_calls=[("python_repl", {"code": "print(1)"}, "1\n", False)],
    )
    r = RuleChecker().check("code_executes", traj, tmp_path)
    assert r.passed is True


def test_code_executes_fails_on_error(tmp_path):
    traj = _trajectory(
        tool_calls=[("bash", {"cmd": "x"}, "error", True)],
    )
    r = RuleChecker().check("code_executes", traj, tmp_path)
    assert r.passed is False


def test_code_executes_fails_without_execution_tool(tmp_path):
    traj = _trajectory(
        tool_calls=[("read_file", {}, "x", False)],
    )
    r = RuleChecker().check("code_executes", traj, tmp_path)
    assert r.passed is False


def test_refactor_quality_needs_read_and_write(tmp_path):
    traj = _trajectory(tool_calls=[
        ("read_file", {}, "src", False),
        ("write_file", {}, "ok", False),
    ])
    r = RuleChecker().check("refactor_quality", traj, tmp_path)
    assert r.passed is True


def test_refactor_quality_fails_without_read(tmp_path):
    traj = _trajectory(tool_calls=[("write_file", {}, "ok", False)])
    r = RuleChecker().check("refactor_quality", traj, tmp_path)
    assert r.passed is False


def test_bug_fixed_needs_two_executions_final_clean(tmp_path):
    traj = _trajectory(tool_calls=[
        ("python_repl", {}, "IndexError", True),
        ("write_file", {}, "ok", False),
        ("python_repl", {}, "ok", False),
    ])
    r = RuleChecker().check("bug_fixed", traj, tmp_path)
    assert r.passed is True


def test_bug_fixed_fails_when_final_still_errors(tmp_path):
    traj = _trajectory(tool_calls=[
        ("python_repl", {}, "err1", True),
        ("python_repl", {}, "err2", True),
    ])
    r = RuleChecker().check("bug_fixed", traj, tmp_path)
    assert r.passed is False


def test_content_quality_passes_on_substantive_answer(tmp_path):
    traj = _trajectory(final_answer="A" * 300)
    r = RuleChecker().check("content_quality", traj, tmp_path)
    assert r.passed is True


def test_content_quality_fails_on_short_answer(tmp_path):
    traj = _trajectory(final_answer="too short")
    r = RuleChecker().check("content_quality", traj, tmp_path)
    assert r.passed is False


def test_comparison_completeness_passes_with_table(tmp_path):
    traj = _trajectory(final_answer=(
        "A | B | C\n---|---|---\nx | y | z\nmore | data | here\n"
        "additional | rows | to pad length past the 100 char threshold."
    ))
    r = RuleChecker().check("comparison_completeness", traj, tmp_path)
    assert r.passed is True


def test_has_statistics_finds_all_three(tmp_path):
    traj = _trajectory(final_answer=(
        "The mean is 5, median is 5, std is 1.2. Dataset stats above."
    ))
    r = RuleChecker().check("has_statistics", traj, tmp_path)
    assert r.passed is True


def test_has_statistics_partial_fail(tmp_path):
    traj = _trajectory(final_answer="Only the mean is present here.")
    r = RuleChecker().check("has_statistics", traj, tmp_path)
    assert r.passed is False
    assert r.score > 0  # partial credit


def test_has_groupby_passes(tmp_path):
    traj = _trajectory(final_answer="After groupby, count is 3 rows per category.")
    r = RuleChecker().check("has_groupby", traj, tmp_path)
    assert r.passed is True


def test_anomaly_detected_passes(tmp_path):
    traj = _trajectory(final_answer="Found 2 anomalies in column X. Values are outliers.")
    r = RuleChecker().check("anomaly_detected", traj, tmp_path)
    assert r.passed is True


def test_files_organized_passes_when_2_subdirs_have_files(tmp_path):
    (tmp_path / "py").mkdir()
    (tmp_path / "py" / "a.py").write_text("x")
    (tmp_path / "md").mkdir()
    (tmp_path / "md" / "b.md").write_text("y")
    r = RuleChecker().check("files_organized", [], tmp_path)
    assert r.passed is True


def test_files_organized_fails_on_flat(tmp_path):
    (tmp_path / "a.py").write_text("x")
    r = RuleChecker().check("files_organized", [], tmp_path)
    assert r.passed is False


def test_errors_found_passes_with_filename_and_line(tmp_path):
    traj = _trajectory(final_answer="In server.log line 42, ERROR: connection refused")
    r = RuleChecker().check("errors_found", traj, tmp_path)
    assert r.passed is True


def test_errors_found_fails_without_line(tmp_path):
    traj = _trajectory(final_answer="Found errors in server.log somewhere")
    r = RuleChecker().check("errors_found", traj, tmp_path)
    assert r.passed is False


def test_duplicates_removed_passes_on_unique_files(tmp_path):
    (tmp_path / "a.txt").write_text("one")
    (tmp_path / "b.txt").write_text("two")
    r = RuleChecker().check("duplicates_removed", [], tmp_path)
    assert r.passed is True


def test_duplicates_removed_fails_on_dupes(tmp_path):
    (tmp_path / "a.txt").write_text("same")
    (tmp_path / "b.txt").write_text("same")
    r = RuleChecker().check("duplicates_removed", [], tmp_path)
    assert r.passed is False


def test_check_result_carries_check_name(tmp_path):
    r = RuleChecker().check("code_executes", [], tmp_path)
    assert r.check_name == "code_executes"
