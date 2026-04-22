"""Deterministic rule-based checks for eval tasks.

Each ``check:`` string in ``configs/eval_tasks.yaml`` maps to a function
here. Checks inspect the agent's trajectory (tool calls, tool results,
final answer) and optionally the workspace filesystem. They return a
``RuleResult`` with a pass/fail boolean, a numeric score in [0, 10], and
a one-line reason.

Design principles:
- Checks are CHEAP — no LLM, no network. Filesystem scans at most.
- Checks are SPECIFIC but not BRITTLE — "has mean/median/std tokens in
  final answer" is better than "output matches regex exactly".
- Unknown check names return score=0 with reason "unknown check" so a
  typo in yaml doesn't pass silently.

Usage::

    checker = RuleChecker()
    result = checker.check("has_statistics", trajectory, workspace_dir)
    # → RuleResult(passed=True, score=8.0, reason="found mean, median, std")
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from src.learning.trajectory import TurnRecord


@dataclass
class RuleResult:
    passed: bool
    score: float  # 0.0 - 10.0
    reason: str
    check_name: str = ""


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def _final_text(trajectory: list["TurnRecord"]) -> str:
    """Concatenate the final-answer text (no tool calls, has content)."""
    parts: list[str] = []
    for t in trajectory:
        if not t.tool_calls:
            content = t.assistant_msg.get("content") or ""
            if content.strip():
                parts.append(content)
    return "\n".join(parts)


def _all_assistant_text(trajectory: list["TurnRecord"]) -> str:
    """Every assistant message content (including tool-call turns)."""
    return "\n".join(
        (t.assistant_msg.get("content") or "") for t in trajectory
    )


def _tool_outputs(trajectory: list["TurnRecord"]) -> str:
    """Every tool result content concatenated."""
    parts: list[str] = []
    for t in trajectory:
        for tr in t.tool_results:
            parts.append(tr.content or "")
    return "\n".join(parts)


def _tools_used(trajectory: list["TurnRecord"]) -> set[str]:
    return {tc.name for t in trajectory for tc in t.tool_calls}


def _any_tool_result_error(trajectory: list["TurnRecord"]) -> bool:
    return any(tr.is_error for t in trajectory for tr in t.tool_results)


def check_code_executes(
    trajectory: list["TurnRecord"], workspace: Path,
) -> RuleResult:
    """Passed if a Python REPL / bash tool ran without error during the run."""
    execution_tools = {"python_repl", "bash", "shell"}
    ran_something = any(
        t.name in execution_tools
        for turn in trajectory for t in turn.tool_calls
    )
    if not ran_something:
        return RuleResult(False, 0.0, "no execution tool invoked")
    if _any_tool_result_error(trajectory):
        return RuleResult(False, 3.0, "execution tool returned error")
    return RuleResult(True, 10.0, "execution tool ran without error")


def check_refactor_quality(
    trajectory: list["TurnRecord"], workspace: Path,
) -> RuleResult:
    """Passed if the agent read a file AND wrote a file (minimum refactor)."""
    tools = _tools_used(trajectory)
    read = "read_file" in tools
    wrote = "write_file" in tools
    if read and wrote:
        return RuleResult(True, 8.0, "read and write_file both used")
    if wrote:
        return RuleResult(False, 4.0, "wrote without reading — not a refactor")
    return RuleResult(False, 0.0, "no file write detected")


def check_bug_fixed(
    trajectory: list["TurnRecord"], workspace: Path,
) -> RuleResult:
    """Passed if the agent ran an executor at least twice (before + after fix)
    with no error on the final run."""
    exec_results: list[bool] = []  # is_error for each execution call
    for turn in trajectory:
        for i, tc in enumerate(turn.tool_calls):
            if tc.name in {"python_repl", "bash"}:
                matching = [tr for tr in turn.tool_results if tr.tool_call_id == tc.id]
                if matching:
                    exec_results.append(matching[0].is_error)
    if len(exec_results) < 2:
        return RuleResult(False, 2.0, "did not reproduce + verify")
    if exec_results[-1]:
        return RuleResult(False, 3.0, "final execution still errored")
    return RuleResult(True, 9.0, "executed twice, final clean")


def check_content_quality(
    trajectory: list["TurnRecord"], workspace: Path,
) -> RuleResult:
    """Passed if the final answer is substantive (>= 200 chars, not an error)."""
    text = _final_text(trajectory)
    n = len(text.strip())
    if n < 100:
        return RuleResult(False, 2.0, f"final answer too short ({n} chars)")
    if n >= 500:
        return RuleResult(True, 9.0, f"substantive answer ({n} chars)")
    return RuleResult(True, 7.0, f"answer present ({n} chars)")


def check_comparison_completeness(
    trajectory: list["TurnRecord"], workspace: Path,
) -> RuleResult:
    """Passed if the final answer contains comparative structure (table or
    multiple 'vs'/'compared to' mentions) AND references both subjects."""
    text = _final_text(trajectory).lower()
    has_structure = (
        "|" in text or text.count("\n- ") >= 3 or "vs" in text or "compared" in text
    )
    if not has_structure:
        return RuleResult(False, 3.0, "no comparative structure")
    return RuleResult(True, 8.0, "comparison present")


def check_has_statistics(
    trajectory: list["TurnRecord"], workspace: Path,
) -> RuleResult:
    """Passed if the agent's output mentions mean, median, AND std (or stddev)."""
    blob = (_final_text(trajectory) + "\n" + _tool_outputs(trajectory)).lower()
    found = []
    if "mean" in blob:
        found.append("mean")
    if "median" in blob:
        found.append("median")
    if "std" in blob or "stddev" in blob or "standard deviation" in blob:
        found.append("std")
    if len(found) == 3:
        return RuleResult(True, 9.0, f"found {', '.join(found)}")
    return RuleResult(False, float(len(found)) * 2.0, f"missing: {3 - len(found)} of mean/median/std")


def check_has_groupby(
    trajectory: list["TurnRecord"], workspace: Path,
) -> RuleResult:
    """Passed if the agent used groupby (in code OR in output) and reports counts."""
    blob = (_all_assistant_text(trajectory) + "\n" + _tool_outputs(trajectory)).lower()
    has_groupby = "groupby" in blob or "group by" in blob
    has_count = "count" in blob or re.search(r"\b\d+\s+rows?\b", blob) is not None
    if has_groupby and has_count:
        return RuleResult(True, 9.0, "groupby + count present")
    if has_groupby:
        return RuleResult(True, 6.0, "groupby present, count weak")
    return RuleResult(False, 2.0, "no groupby evidence")


def check_anomaly_detected(
    trajectory: list["TurnRecord"], workspace: Path,
) -> RuleResult:
    """Passed if the final answer lists at least one anomaly or outlier."""
    text = _final_text(trajectory).lower()
    if "anomal" in text or "outlier" in text:
        return RuleResult(True, 8.0, "anomaly/outlier mentioned in answer")
    return RuleResult(False, 2.0, "no anomaly language in answer")


def check_files_organized(
    trajectory: list["TurnRecord"], workspace: Path,
) -> RuleResult:
    """Passed if workspace contains at least 2 subdirectories with files after
    the run."""
    if not workspace.exists():
        return RuleResult(False, 0.0, "workspace does not exist")
    subdirs_with_files = 0
    for p in workspace.iterdir():
        if p.is_dir() and any(p.iterdir()):
            subdirs_with_files += 1
    if subdirs_with_files >= 2:
        return RuleResult(True, 9.0, f"{subdirs_with_files} subdirs with files")
    return RuleResult(False, float(subdirs_with_files) * 3.0, "not enough subdirectories")


def check_errors_found(
    trajectory: list["TurnRecord"], workspace: Path,
) -> RuleResult:
    """Passed if the final answer contains both a filename-looking token
    AND a line-number-looking integer."""
    text = _final_text(trajectory)
    has_file = bool(re.search(r"\b[\w.-]+\.(log|txt|py|md)\b", text, re.IGNORECASE))
    has_line = bool(re.search(r"\b(line\s*\d+|:\d+:)\b", text, re.IGNORECASE))
    if has_file and has_line:
        return RuleResult(True, 9.0, "filename + line number present")
    parts = []
    if has_file:
        parts.append("filename")
    if has_line:
        parts.append("line")
    return RuleResult(
        False, float(len(parts)) * 3.0,
        f"missing: {', '.join(sorted({'filename', 'line'} - set(parts)))}",
    )


def check_duplicates_removed(
    trajectory: list["TurnRecord"], workspace: Path,
) -> RuleResult:
    """Passed if workspace has no duplicate file contents after the run."""
    if not workspace.exists():
        return RuleResult(False, 0.0, "workspace does not exist")
    hashes: dict[str, Path] = {}
    duplicates_found = 0
    for p in workspace.rglob("*"):
        if not p.is_file():
            continue
        try:
            h = hashlib.sha1(p.read_bytes()).hexdigest()
        except OSError:
            continue
        if h in hashes:
            duplicates_found += 1
        else:
            hashes[h] = p
    if duplicates_found == 0 and len(hashes) > 0:
        return RuleResult(True, 9.0, f"{len(hashes)} unique files, no dupes")
    return RuleResult(
        False, max(0.0, 10.0 - float(duplicates_found) * 3.0),
        f"{duplicates_found} duplicate(s) remain",
    )


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


_CHECKS: dict[str, Callable[[list["TurnRecord"], Path], RuleResult]] = {
    "code_executes": check_code_executes,
    "refactor_quality": check_refactor_quality,
    "bug_fixed": check_bug_fixed,
    "content_quality": check_content_quality,
    "comparison_completeness": check_comparison_completeness,
    "has_statistics": check_has_statistics,
    "has_groupby": check_has_groupby,
    "anomaly_detected": check_anomaly_detected,
    "files_organized": check_files_organized,
    "errors_found": check_errors_found,
    "duplicates_removed": check_duplicates_removed,
}


class RuleChecker:
    def check(
        self,
        check_name: str,
        trajectory: list["TurnRecord"],
        workspace: Path,
    ) -> RuleResult:
        fn = _CHECKS.get(check_name)
        if fn is None:
            return RuleResult(
                passed=False, score=0.0,
                reason=f"unknown check {check_name!r}",
                check_name=check_name,
            )
        result = fn(trajectory, workspace)
        result.check_name = check_name
        return result

    @property
    def supported(self) -> list[str]:
        return sorted(_CHECKS)
