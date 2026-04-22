"""Skill auto-creation — extract reusable workflows from successful runs.

When a complex task completes successfully, the skill creator asks the
auxiliary LLM to distill the trajectory into a SKILL.md file in the
agentskills.io format. Skills are the agent's procedural memory — next
time a similar task shows up, the assembler can load the skill body as
extra context (deferred to a later phase).

Gate before creating:
- outcome must be "success" (final_answer reached, budget not exhausted,
  no unhandled error)
- trajectory must have >= 4 tool calls (trivial Q&A isn't a skill)
- trajectory must use >= 2 distinct tool names (single-tool workflows
  don't need a skill — the tool is the skill)

Dedup:
- Compute the skill's tentative description from the user request
- Query skill_index.find_similar(description, tags, threshold=0.6)
- If a similar skill exists, fall through — Unit 3b (skill_improver) is
  responsible for deciding if the new run was better
- Otherwise, write and register the new skill

Failure modes are all non-fatal: aux-LLM errors, malformed JSON,
filesystem errors are logged and the run returns None. A failed
skill-creation attempt never surfaces as a user-visible error.
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Any, Literal

from src.learning.skill_format import Skill

if TYPE_CHECKING:
    from pathlib import Path

    from src.learning.trajectory import TurnRecord
    from src.memory.skill_index import SkillIndex
    from src.providers.base import LLMProvider

logger = logging.getLogger(__name__)


Outcome = Literal["success", "error", "interrupted", "budget_exhausted"]


SKILL_SYSTEM_PROMPT = """\
You extract reusable skills from agent trajectories. Given a successful \
task trace, write a SKILL.md file that will help the agent solve similar \
tasks faster next time.

Output rules:
- Respond with ONLY a JSON object (no prose, no code fences):
  {
    "slug": "<kebab-case-slug-max-40-chars>",
    "name": "<Short Title>",
    "description": "<one-sentence description, max 100 chars>",
    "tags": ["domain1", "domain2"],
    "steps": ["step 1", "step 2", ...],
    "when_to_use": "<one sentence>",
    "example_input": "<condensed original user request>"
  }
- Tags MUST be one of: coding, data_analysis, research, file_ops.
- Steps describe the WORKFLOW (what tools to call in what order), not the \
specific arguments. Keep each step under 80 chars.
- If the trajectory is not really reusable (one-off, too specific, \
unsuccessful) respond with exactly: null
"""


class SkillCreator:
    """Decides whether and what to extract from a completed trajectory."""

    def __init__(
        self,
        aux_provider: "LLMProvider | None",
        skill_index: "SkillIndex",
        min_tool_calls: int = 4,
        min_distinct_tools: int = 2,
        dedup_threshold: float = 0.6,
    ) -> None:
        self._provider = aux_provider
        self._index = skill_index
        self.min_tool_calls = min_tool_calls
        self.min_distinct_tools = min_distinct_tools
        self.dedup_threshold = dedup_threshold

    async def maybe_create_skill(
        self,
        trajectory: list["TurnRecord"],
        outcome: Outcome = "success",
    ) -> Skill | None:
        """Run one skill-creation attempt; returns the created Skill or None.

        None results from: gate failure, no provider, dedup hit, aux LLM
        error, or malformed aux output. Errors are logged, never raised.
        """
        if not self._passes_gate(trajectory, outcome):
            return None
        if self._provider is None:
            logger.debug("Skill creator: no aux provider; skipping")
            return None

        user_request = _extract_user_request(trajectory)
        expected_tool_count = _count_tool_calls(trajectory)

        # Quick dedup before spending an aux-LLM call — if the user request
        # alone is already covered by an existing skill, no need to extract.
        if user_request:
            hit = self._index.find_similar(
                user_request, tags=None, threshold=self.dedup_threshold,
            )
            if hit is not None:
                logger.info(
                    "Skill creator: dedup hit on %r (score %.2f); skipping",
                    hit.slug, hit.score,
                )
                return None

        prompt_text = _format_trajectory(trajectory)
        try:
            response = await self._run_aux(user_request, prompt_text)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skill creator aux LLM failed: %s", exc)
            return None

        parsed = _parse_response(response)
        if parsed is None:
            logger.info("Skill creator: aux LLM declined or produced garbage")
            return None

        # Second dedup using the aux-LLM-produced description (more precise
        # than the raw user request).
        hit = self._index.find_similar(
            parsed["description"], tags=parsed.get("tags"),
            threshold=self.dedup_threshold,
        )
        if hit is not None:
            logger.info(
                "Skill creator: post-extraction dedup hit on %r (score %.2f)",
                hit.slug, hit.score,
            )
            return None

        skill = _build_skill(parsed, expected_tool_count=expected_tool_count)
        try:
            self._index.register(skill)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skill creator: failed to register %r: %s", skill.slug, exc)
            return None
        logger.info("Skill creator: wrote %s (v%d)", skill.slug, skill.version)
        return skill

    # ------------------------------------------------------------------
    # Gate
    # ------------------------------------------------------------------

    def _passes_gate(
        self, trajectory: list["TurnRecord"], outcome: Outcome,
    ) -> bool:
        if outcome != "success":
            return False
        total_calls = _count_tool_calls(trajectory)
        if total_calls < self.min_tool_calls:
            return False
        distinct = _distinct_tools(trajectory)
        if len(distinct) < self.min_distinct_tools:
            return False
        return True

    # ------------------------------------------------------------------
    # Aux LLM
    # ------------------------------------------------------------------

    async def _run_aux(self, user_request: str, trajectory_text: str) -> str:
        assert self._provider is not None
        user_msg = (
            f"User request: {user_request or '(not recorded)'}\n\n"
            f"Trajectory:\n{trajectory_text}\n\n"
            f"Extract a reusable skill or respond with null."
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SKILL_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        text = ""
        async for chunk in self._provider.stream_message(
            messages=messages,  # type: ignore[arg-type]
            temperature=0.0,
        ):
            if chunk.type == "token":
                text += chunk.data
            elif chunk.type == "done":
                break
        return text


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _count_tool_calls(trajectory: list["TurnRecord"]) -> int:
    return sum(len(t.tool_calls) for t in trajectory)


def _distinct_tools(trajectory: list["TurnRecord"]) -> set[str]:
    names: set[str] = set()
    for t in trajectory:
        for tc in t.tool_calls:
            names.add(tc.name)
    return names


def _extract_user_request(trajectory: list["TurnRecord"]) -> str:
    """Return the first user message's content, or empty string."""
    for t in trajectory:
        if t.user_msg is not None:
            return (t.user_msg.get("content") or "").strip()
    return ""


def _format_trajectory(trajectory: list["TurnRecord"]) -> str:
    lines: list[str] = []
    for t in trajectory:
        if t.user_msg is not None:
            lines.append(f"[user] {(t.user_msg.get('content') or '').strip()}")
        asst = (t.assistant_msg.get("content") or "").strip()
        if asst:
            lines.append(f"[assistant] {asst[:200]}")
        for tc in t.tool_calls:
            args = str(tc.arguments)[:120]
            lines.append(f"[tool_call] {tc.name}({args})")
        for tr in t.tool_results:
            preview = (tr.content or "").replace("\n", " ")[:150]
            marker = " [ERROR]" if tr.is_error else ""
            lines.append(f"[tool_result{marker}] {tr.name}: {preview}")
    return "\n".join(lines) if lines else "(empty trajectory)"


_JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}", re.DOTALL)


def _parse_response(text: str) -> dict[str, Any] | None:
    """Parse the aux LLM response. Tolerant of code fences and surrounding
    prose. Returns None on parse failure or explicit null."""
    s = text.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:].lstrip()
    s = s.strip()
    if s == "null" or s == "":
        return None
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        m = _JSON_BLOCK_RE.search(s)
        if m is None:
            return None
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    if not isinstance(obj, dict):
        return None
    required = ("slug", "name", "description", "steps")
    for field in required:
        if field not in obj:
            return None
    if not isinstance(obj["slug"], str) or not obj["slug"].strip():
        return None
    if not isinstance(obj["name"], str) or not obj["name"].strip():
        return None
    if not isinstance(obj["description"], str) or not obj["description"].strip():
        return None
    if not isinstance(obj["steps"], list) or not obj["steps"]:
        return None
    return obj


_VALID_TAGS = {"coding", "data_analysis", "research", "file_ops"}
_SLUG_RE = re.compile(r"[^a-z0-9-]+")


def _build_skill(
    parsed: dict[str, Any], expected_tool_count: int,
) -> Skill:
    """Turn the aux-LLM output into a Skill dataclass with markdown body."""
    slug = _sanitize_slug(parsed["slug"])
    name = parsed["name"].strip()[:80]
    description = parsed["description"].strip()[:200]
    tags_raw = parsed.get("tags") or []
    if isinstance(tags_raw, str):
        tags_raw = [tags_raw]
    tags = [t for t in (str(x).strip() for x in tags_raw) if t in _VALID_TAGS]
    steps = [str(s).strip() for s in parsed["steps"] if str(s).strip()]
    when_to_use = (parsed.get("when_to_use") or "").strip()
    example = (parsed.get("example_input") or "").strip()

    body_lines: list[str] = [f"# {name}", ""]
    if when_to_use:
        body_lines += ["## When to use", when_to_use, ""]
    body_lines += ["## Steps"] + [f"{i+1}. {s}" for i, s in enumerate(steps)] + [""]
    if example:
        body_lines += ["## Example", example, ""]
    body = "\n".join(body_lines)

    return Skill(
        slug=slug,
        name=name,
        description=description,
        body=body,
        version=1,
        tags=tags,
        expected_tool_count=expected_tool_count or None,
    )


def _sanitize_slug(raw: str) -> str:
    s = raw.strip().lower()
    s = _SLUG_RE.sub("-", s)
    s = s.strip("-")
    if not s:
        s = "skill"
    return s[:40]
