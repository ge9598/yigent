"""Render a BenchmarkReport as a Markdown report."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.eval.benchmark import BenchmarkReport, TaskResult


def generate_report(report: "BenchmarkReport") -> str:
    """Produce a Markdown report suitable for ``docs/EVAL_REPORT.md``."""
    lines: list[str] = [
        "# Yigent Benchmark Report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Tasks: **{report.total_tasks}**"
        f" | Duration: **{report.duration_s:.1f}s**"
        f" | Skills created: **{report.skill_creation_count}**",
        "",
        "## Summary",
        "",
        f"- **Overall completion rate:** {report.completion_rate_overall:.0%}",
        f"- **Overall avg score:** {report.avg_score_overall:.2f} / 10",
        f"- **Avg steps per task:** {report.avg_steps_overall:.1f}",
        f"- **Cross-domain consistency:** {report.consistency_score:.2f}",
        f"- **Error recovery rate:** {report.recovery_rate:.0%}",
        "",
        "## Per-domain metrics",
        "",
        "| Domain | Completion | Avg score | Avg steps |",
        "|---|---|---|---|",
    ]
    domains = sorted(report.completion_rate_by_domain)
    for d in domains:
        cr = report.completion_rate_by_domain[d]
        sc = report.avg_score_by_domain.get(d, 0.0)
        st = report.avg_steps_by_domain.get(d, 0.0)
        lines.append(f"| {d} | {cr:.0%} | {sc:.2f} | {st:.1f} |")

    lines += [
        "",
        "## Per-task results",
        "",
        "| Domain | Difficulty | Passed | Rule | Judge | Final | Steps | Duration |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(report.per_task, key=lambda x: (x.domain, x.difficulty)):
        passed = "✓" if r.passed else "✗"
        lines.append(
            f"| {r.domain} | {r.difficulty} | {passed} "
            f"| {r.rule.score:.1f} | {r.judge.score:.1f} "
            f"| {r.final_score:.2f} | {r.steps} | {r.duration_s:.1f}s |"
        )

    lines += ["", "## Task details", ""]
    for r in sorted(report.per_task, key=lambda x: (x.domain, x.difficulty)):
        lines.append(f"### {r.domain} / {r.difficulty}")
        lines.append("")
        lines.append(f"> {r.task}")
        lines.append("")
        lines.append(f"- Rule check (`{r.rule.check_name}`): {_checkmark(r.rule.passed)} "
                     f"score={r.rule.score:.1f} — {r.rule.reason}")
        lines.append(f"- Judge: correctness={r.judge.correctness}, "
                     f"efficiency={r.judge.efficiency}, "
                     f"robustness={r.judge.robustness}")
        if r.judge.reasoning:
            lines.append(f"  > {r.judge.reasoning}")
        if r.error:
            lines.append(f"- Error: `{r.error}`")
        lines.append("")

    return "\n".join(lines) + "\n"


def _checkmark(passed: bool) -> str:
    return "✓" if passed else "✗"
