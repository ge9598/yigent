"""Benchmark runner — 4-domain × 3-difficulty eval harness.

Loads tasks from ``configs/eval_tasks.yaml``, runs each through a fresh
AgentLoop with a dedicated IterationBudget and TrajectoryRecorder, scores
the result via ``RuleChecker`` + ``LLMJudge``, and emits a
``BenchmarkReport`` with per-domain and overall metrics.

Entry point: ``python -m src.eval.benchmark --suite all``

Metrics produced (per domain + overall):
- completion_rate — fraction of tasks whose final answer passed the rule
  check AND received a nonzero judge score
- avg_steps — mean tool-call count per task
- avg_final_score — mean of (0.4 * rule_score + 0.6 * llm_score)
- recovery_rate — for tasks where any tool returned an error, fraction
  where the task still completed
- consistency_score — 1 - normalized variance of per-domain completion
  rates (measures cross-domain robustness)
- skill_creation_count — number of skills written during the run

The runner is deliberately self-contained — it builds a minimal agent
stack (provider, registry, executor, assembler) rather than reusing the
CLI's wiring, so the benchmark can run headless without Rich/UI code.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.context.assembler import ContextAssembler
from src.context.engine import CompressionEngine
from src.core.agent_loop import agent_loop
from src.core.config import AgentConfig, load_config
from src.core.env_injector import EnvironmentInjector
from src.core.iteration_budget import IterationBudget
from src.core.plan_mode import PlanMode
from src.core.streaming_executor import StreamingExecutor
from src.core.types import (
    ErrorEvent, FinalAnswerEvent, Message, ToolContext,
)
from src.eval.judges.llm_judge import DEFAULT_JUDGE_PROMPT, JudgeResult, LLMJudge
from src.eval.judges.rule_checks import RuleChecker, RuleResult
from src.learning.trajectory import TrajectoryRecorder
from src.memory.skill_index import SkillIndex
from src.providers.base import LLMProvider
from src.providers.resolver import resolve_auxiliary, resolve_provider
from src.safety.hook_system import HookSystem
from src.safety.permission_gate import PermissionGate
from src.tools.registry import get_registry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class EvalTask:
    domain: str
    task: str
    check: str
    difficulty: str
    timeout: int = 120
    setup: str | None = None


@dataclass
class TaskResult:
    domain: str
    difficulty: str
    task: str
    passed: bool
    rule: RuleResult
    judge: JudgeResult
    final_score: float  # 0-10
    steps: int
    duration_s: float
    had_errors: bool
    recovered: bool
    error: str | None = None


@dataclass
class BenchmarkReport:
    total_tasks: int
    per_task: list[TaskResult]
    completion_rate_overall: float
    completion_rate_by_domain: dict[str, float]
    avg_steps_overall: float
    avg_steps_by_domain: dict[str, float]
    avg_score_overall: float
    avg_score_by_domain: dict[str, float]
    recovery_rate: float
    consistency_score: float
    skill_creation_count: int
    duration_s: float


# ---------------------------------------------------------------------------
# Task loading
# ---------------------------------------------------------------------------


def load_tasks(
    tasks_file: Path, suite: str = "all",
) -> tuple[list[EvalTask], dict[str, float], str]:
    """Return (tasks, scoring_weights, judge_prompt_template).

    ``suite`` filters by domain; ``"all"`` returns every task. Scoring
    weights fall back to (0.4, 0.6) if the yaml lacks them.
    """
    data = yaml.safe_load(tasks_file.read_text(encoding="utf-8")) or {}
    scoring = data.get("scoring") or {}
    weights = {
        "rule_check_weight": float(scoring.get("rule_check_weight", 0.4)),
        "llm_judge_weight": float(scoring.get("llm_judge_weight", 0.6)),
    }
    judge_prompt = data.get("judge_prompt") or ""

    tasks: list[EvalTask] = []
    for domain, entries in data.items():
        if domain in {"scoring", "judge_prompt"}:
            continue
        if not isinstance(entries, list):
            continue
        if suite != "all" and domain != suite:
            continue
        for entry in entries:
            tasks.append(EvalTask(
                domain=domain,
                task=str(entry["task"]),
                check=str(entry["check"]),
                difficulty=str(entry.get("difficulty", "medium")),
                timeout=int(entry.get("timeout", 120)),
                setup=entry.get("setup"),
            ))
    return tasks, weights, judge_prompt


# ---------------------------------------------------------------------------
# Workspace setup
# ---------------------------------------------------------------------------


def _prepare_workspace(task: EvalTask, root: Path) -> Path:
    """Create a per-task workspace. If task.setup hints at CSV / log / mixed
    files, seed the workspace with a minimal synthetic dataset."""
    workspace = root / f"{task.domain}_{task.difficulty}"
    workspace.mkdir(parents=True, exist_ok=True)
    if not task.setup:
        return workspace
    s = task.setup.lower()
    if "csv" in s:
        if "category" in s:
            rows = ["category,value"] + [
                f"cat{(i % 3)},{(i * 3) % 100}" for i in range(60)
            ]
        elif "anomal" in s:
            rows = ["x,y,z"] + [f"{i},{i*2},{i*3}" for i in range(100)]
            rows.append("999,999,999")  # injected outlier
        else:
            rows = ["a,b,c,d,e"] + [
                f"{i},{i*2},{i%7},{(i*3)%50},{i/2}" for i in range(100)
            ]
        (workspace / "data.csv").write_text("\n".join(rows), encoding="utf-8")
    if "log files" in s or "mixed log levels" in s:
        for i in range(5):
            lines = [
                f"2026-04-22 10:{j:02d}:00 "
                f"{'ERROR' if j % 7 == 0 else 'INFO'} sample msg {j}"
                for j in range(20)
            ]
            (workspace / f"server{i}.log").write_text(
                "\n".join(lines), encoding="utf-8",
            )
    if "mixed files" in s or ("test_workspace" in s and "duplicate" not in s
                              and "log files" not in s):
        for ext, content in [("py", "x=1\n"), ("txt", "hello"), ("md", "# h")]:
            for i in range(5):
                (workspace / f"f{i}.{ext}").write_text(content, encoding="utf-8")
    if "duplicate" in s:
        unique = ["alpha\n", "beta\n", "gamma\n", "delta\n",
                  "epsilon\n", "zeta\n", "eta\n"]
        for i, c in enumerate(unique):
            (workspace / f"u{i}.txt").write_text(c, encoding="utf-8")
        for i in range(3):
            (workspace / f"dup{i}.txt").write_text(unique[i], encoding="utf-8")
    # Two shapes of buggy.py depending on which coding task this is:
    # - bug_fixed (coding/hard): contains pick_third that crashes on line 15
    # - refactor_quality (coding/medium): contains only process() to be
    #   refactored, matching the task description exactly
    # Without this split the agent reads a fixture that contradicts the
    # prompt and starts fabricating.
    if task.check == "bug_fixed" or "indexerror" in s:
        # The prompt promises pick_third's crashing `return items[2]` is
        # on line 15 — layout padded so that's literally true.
        buggy = (
            "# buggy.py — pick_third(items) crashes on line 15 when\n"
            "# items has fewer than 3 elements.\n"
            "\n"
            "\n"
            "def _identity(x):\n"  # line 5
            "    return x\n"
            "\n"
            "\n"
            "def _shout(msg):\n"  # line 9
            "    return msg.upper()\n"
            "\n"
            "\n"
            "def pick_third(items):\n"  # line 13
            "    # Expected to return the 3rd element — crashes on short lists.\n"
            "    return items[2]\n"  # line 15
            "\n"
            "\n"
            "if __name__ == '__main__':\n"
            "    data = [1, 2]\n"
            "    print(pick_third(data))  # triggers IndexError\n"
        )
        (workspace / "buggy.py").write_text(buggy, encoding="utf-8")
    elif task.check == "refactor_quality" or "refactor" in s:
        buggy = (
            "def process(items):\n"
            "    total = 0\n"
            "    for i in items:\n"
            "        total += i\n"
            "    return total\n"
            "\n"
            "\n"
            "if __name__ == '__main__':\n"
            "    data = [1, 2, 3, 4]\n"
            "    print(process(data))\n"
        )
        (workspace / "buggy.py").write_text(buggy, encoding="utf-8")
    return workspace


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class BenchmarkRunner:
    def __init__(
        self,
        config: AgentConfig,
        tasks_file: Path,
        output_dir: Path | None = None,
        skill_index: SkillIndex | None = None,
    ) -> None:
        self.config = config
        self.tasks_file = Path(tasks_file)
        self.output_dir = Path(output_dir) if output_dir else Path("benchmark_runs")
        self.skill_index = skill_index or SkillIndex(
            Path(config.learning.skills_dir)
        )
        self.skill_index.rebuild()
        self._initial_skill_count = len(self.skill_index)

    async def run(self, suite: str = "all") -> BenchmarkReport:
        started = time.time()
        tasks, weights, judge_prompt = load_tasks(self.tasks_file, suite=suite)
        if not tasks:
            raise ValueError(f"No tasks found for suite {suite!r}")

        # Build shared eval dependencies once — provider, aux, judge.
        # Individual tasks still get fresh budgets / registries / recorders.
        primary = resolve_provider(self.config)
        aux = resolve_auxiliary(self.config)
        judge = LLMJudge(aux, prompt_template=judge_prompt or DEFAULT_JUDGE_PROMPT)
        checker = RuleChecker()

        self.output_dir.mkdir(parents=True, exist_ok=True)

        results: list[TaskResult] = []
        for task in tasks:
            logger.info("Running %s/%s: %s",
                        task.domain, task.difficulty, task.task[:60])
            result = await self._run_one(
                task, primary, aux, judge, checker, weights,
            )
            results.append(result)

        # Skill creation delta — how many skills were added during this run
        self.skill_index.rebuild()
        skill_delta = max(0, len(self.skill_index) - self._initial_skill_count)

        return _aggregate(results, skill_delta, duration_s=time.time() - started)

    async def _run_one(
        self,
        task: EvalTask,
        primary: LLMProvider,
        aux: LLMProvider | None,
        judge: LLMJudge,
        checker: RuleChecker,
        weights: dict[str, float],
    ) -> TaskResult:
        started = time.time()
        workspace = _prepare_workspace(task, self.output_dir)
        recorder = TrajectoryRecorder(
            session_id=f"{task.domain}_{task.difficulty}",
        )

        registry = get_registry()
        plan_mode = PlanMode()
        ctx = ToolContext(
            plan_mode=plan_mode, registry=registry,
            config=self.config, working_dir=workspace,
        )
        hooks = HookSystem()
        permission_gate = PermissionGate(
            registry=registry, ctx=ctx, hooks=hooks,
            yolo_mode=True,  # headless: auto-allow
            aux_provider=aux,
        )
        executor = StreamingExecutor(registry, ctx, permission_gate=permission_gate)
        compression = CompressionEngine(auxiliary_provider=aux, hook_system=hooks)
        assembler = ContextAssembler(
            system_prompt=[Message(role="system", content="You are Yigent, a capable AI agent.")],
            plan_mode=plan_mode,
            compression_engine=compression,
            output_reserve=self.config.context.output_reserve,
            safety_buffer=self.config.context.buffer,
            hook_system=hooks,
        )
        env_injector = EnvironmentInjector()
        budget = IterationBudget(min(self.config.agent.max_iterations, 30))

        conversation: list[Message] = [Message(role="user", content=task.task)]

        had_error = False
        try:
            async def _drive() -> None:
                nonlocal had_error
                async for event in agent_loop(
                    conversation=conversation,
                    tools=registry,
                    budget=budget,
                    provider=primary,
                    executor=executor,
                    env_injector=env_injector,
                    plan_mode=plan_mode,
                    config=self.config,
                    assembler=assembler,
                    hooks=hooks,
                    trajectory=recorder,
                ):
                    if isinstance(event, ErrorEvent):
                        had_error = True
                    elif isinstance(event, FinalAnswerEvent):
                        pass

            await asyncio.wait_for(_drive(), timeout=task.timeout)
            error: str | None = None
        except asyncio.TimeoutError:
            error = f"timeout after {task.timeout}s"
            had_error = True
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
            had_error = True

        duration = time.time() - started
        steps = sum(len(t.tool_calls) for t in recorder.turns)
        rule_result = checker.check(task.check, recorder.turns, workspace)
        judge_result = await judge.judge(
            task.task, task.check, recorder.turns,
        )

        final_score = (
            weights["rule_check_weight"] * rule_result.score
            + weights["llm_judge_weight"] * judge_result.score
        )
        passed = rule_result.passed and judge_result.score > 0
        had_tool_errors = any(
            tr.is_error for turn in recorder.turns for tr in turn.tool_results
        )
        recovered = had_tool_errors and passed

        # Save the trajectory alongside the workspace for later replay
        try:
            recorder.save(
                self.output_dir / f"{task.domain}_{task.difficulty}.json",
                fmt="sharegpt",
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Trajectory save failed: %s", exc)

        return TaskResult(
            domain=task.domain,
            difficulty=task.difficulty,
            task=task.task,
            passed=passed,
            rule=rule_result,
            judge=judge_result,
            final_score=final_score,
            steps=steps,
            duration_s=duration,
            had_errors=had_tool_errors,
            recovered=recovered,
            error=error,
        )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _aggregate(
    results: list[TaskResult], skill_delta: int, duration_s: float,
) -> BenchmarkReport:
    by_domain: dict[str, list[TaskResult]] = {}
    for r in results:
        by_domain.setdefault(r.domain, []).append(r)

    def _rate(rs: list[TaskResult]) -> float:
        return sum(1 for r in rs if r.passed) / max(1, len(rs))

    def _avg_steps(rs: list[TaskResult]) -> float:
        return statistics.mean(r.steps for r in rs) if rs else 0.0

    def _avg_score(rs: list[TaskResult]) -> float:
        return statistics.mean(r.final_score for r in rs) if rs else 0.0

    completion_by_domain = {d: _rate(rs) for d, rs in by_domain.items()}
    steps_by_domain = {d: _avg_steps(rs) for d, rs in by_domain.items()}
    score_by_domain = {d: _avg_score(rs) for d, rs in by_domain.items()}

    # Consistency: lower cross-domain variance in completion → higher
    # consistency. Normalize by max possible variance (0.25 for binomial
    # means in [0, 1]).
    if len(completion_by_domain) >= 2:
        var = statistics.pvariance(completion_by_domain.values())
        consistency = max(0.0, 1.0 - var / 0.25)
    else:
        consistency = 1.0

    error_tasks = [r for r in results if r.had_errors]
    recovery_rate = (
        sum(1 for r in error_tasks if r.recovered) / len(error_tasks)
        if error_tasks else 1.0
    )

    return BenchmarkReport(
        total_tasks=len(results),
        per_task=results,
        completion_rate_overall=_rate(results),
        completion_rate_by_domain=completion_by_domain,
        avg_steps_overall=_avg_steps(results),
        avg_steps_by_domain=steps_by_domain,
        avg_score_overall=_avg_score(results),
        avg_score_by_domain=score_by_domain,
        recovery_rate=recovery_rate,
        consistency_score=consistency,
        skill_creation_count=skill_delta,
        duration_s=duration_s,
    )


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Yigent eval benchmark")
    parser.add_argument(
        "--suite", default="all",
        choices=["all", "coding", "data_analysis", "file_management", "research"],
    )
    parser.add_argument("--tasks-file", default="configs/eval_tasks.yaml")
    parser.add_argument("--output-dir", default="benchmark_runs")
    parser.add_argument("--report-path", default="docs/EVAL_REPORT.md")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = load_config()
    runner = BenchmarkRunner(
        config=config,
        tasks_file=Path(args.tasks_file),
        output_dir=Path(args.output_dir),
    )
    report = asyncio.run(runner.run(suite=args.suite))

    # Late import — reporter is small and doesn't need to be loaded on
    # --help.
    from src.eval.reporter import generate_report
    report_md = generate_report(report)
    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_md, encoding="utf-8")

    json_path = Path(args.output_dir) / "report.json"
    json_path.write_text(
        json.dumps(asdict(report), indent=2, default=str), encoding="utf-8",
    )
    print(f"Report → {report_path}")
    print(f"JSON   → {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
