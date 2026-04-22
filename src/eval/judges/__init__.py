"""Dual-channel evaluation: rule checks + LLM-as-Judge.

Each eval task is scored on two orthogonal axes:

- ``RuleChecker`` runs a deterministic check matching the task's ``check:``
  field (code_executes, files_organized, etc.). Returns a boolean pass/fail
  plus a short reason. Fast, cheap, no LLM.
- ``LLMJudge`` asks the aux LLM to rate correctness / efficiency /
  robustness on a 0-10 scale using the ``judge_prompt`` template from
  eval_tasks.yaml. Slow, costs one LLM call, more holistic.

Final score = 0.4 × rule_score + 0.6 × llm_score (weights live in
eval_tasks.yaml's ``scoring:`` section).
"""

from src.eval.judges.llm_judge import JudgeResult, LLMJudge
from src.eval.judges.rule_checks import RuleChecker, RuleResult

__all__ = ["RuleChecker", "RuleResult", "LLMJudge", "JudgeResult"]
