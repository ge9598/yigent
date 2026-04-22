"""Skill self-improvement — iterative refinement with rollback.

When an existing skill is used and the task succeeds with fewer tool calls
than the skill's ``expected_tool_count``, the improver asks the aux LLM to
revise the skill's Steps section. The new version is written to
``skills/{slug}.md`` (bumping ``version``); the old version is preserved
under ``skills/.history/{slug}_v{n}.md`` so a regression can roll back.

Gate:
- outcome == "success"
- skill.expected_tool_count is not None (older skills might lack it)
- actual tool count < 0.8 * expected_tool_count (meaningful improvement)

Rollback:
- ``rollback_to_previous(slug)`` restores the most recent archived version
  from ``.history/`` and updates the index. Called by the benchmark runner
  (Unit 6) when it detects a score regression on tasks that used an
  improved skill.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from src.learning.skill_format import Skill, read_skill

if TYPE_CHECKING:
    from src.learning.trajectory import TurnRecord
    from src.memory.skill_index import SkillIndex
    from src.providers.base import LLMProvider

logger = logging.getLogger(__name__)


Outcome = Literal["success", "error", "interrupted", "budget_exhausted"]

HISTORY_SUBDIR = ".history"

IMPROVER_SYSTEM_PROMPT = """\
You refine existing agent skills. You will see an existing skill (with its \
Steps section) and a NEW successful trajectory that completed the same \
task using FEWER tool calls. Rewrite the Steps section to reflect the \
shorter path — if the new path is strictly better.

Output rules:
- Respond with ONLY a JSON object (no prose, no code fences):
  {
    "steps": ["step 1", "step 2", ...],
    "reason": "<one-sentence why this is better>"
  }
- Keep each step under 80 chars. Steps describe the WORKFLOW, not specific \
arguments.
- If the new trajectory is NOT genuinely better (different task, shortcut \
that won't generalize, missing a step the old version had), respond with \
exactly: null
"""


class SkillImprover:
    def __init__(
        self,
        aux_provider: "LLMProvider | None",
        skill_index: "SkillIndex",
        improvement_ratio: float = 0.8,
    ) -> None:
        self._provider = aux_provider
        self._index = skill_index
        self.improvement_ratio = improvement_ratio

    async def maybe_improve(
        self,
        skill: Skill,
        trajectory: list["TurnRecord"],
        outcome: Outcome = "success",
    ) -> Skill | None:
        """Return the new skill version if written, else None."""
        if outcome != "success":
            return None
        if skill.expected_tool_count is None:
            return None
        actual = _count_tool_calls(trajectory)
        if actual == 0:
            return None
        if actual >= skill.expected_tool_count * self.improvement_ratio:
            return None
        if self._provider is None:
            return None

        try:
            response = await self._run_aux(skill, trajectory, actual)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skill improver aux LLM failed: %s", exc)
            return None

        parsed = _parse_response(response)
        if parsed is None:
            logger.info("Skill improver: aux declined improvement for %r", skill.slug)
            return None

        # Archive the old version before overwriting.
        try:
            _archive(skill, self._index.dir)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to archive skill %r (continuing anyway): %s",
                skill.slug, exc,
            )

        new_body = _rewrite_body(skill.body, parsed["steps"])
        new_skill = Skill(
            slug=skill.slug,
            name=skill.name,
            description=skill.description,
            body=new_body,
            version=skill.version + 1,
            tags=list(skill.tags),
            expected_tool_count=actual,  # new baseline
            extra=dict(skill.extra),
        )
        self._index.register(new_skill)
        logger.info(
            "Skill improver: bumped %r to v%d (%d→%d tool calls)",
            skill.slug, new_skill.version, skill.expected_tool_count, actual,
        )
        return new_skill

    async def _run_aux(
        self, skill: Skill, trajectory: list["TurnRecord"], actual: int,
    ) -> str:
        assert self._provider is not None
        user_msg = (
            f"Existing skill (v{skill.version}):\n"
            f"Description: {skill.description}\n"
            f"Previous expected_tool_count: {skill.expected_tool_count}\n"
            f"Previous steps:\n{_extract_steps_block(skill.body)}\n\n"
            f"New successful trajectory ({actual} tool calls):\n"
            f"{_format_trajectory(trajectory)}\n\n"
            "Rewrite the steps if genuinely better, else reply null."
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": IMPROVER_SYSTEM_PROMPT},
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

    # ------------------------------------------------------------------
    # Rollback
    # ------------------------------------------------------------------

    def rollback_to_previous(self, slug: str) -> Skill | None:
        """Restore the most recent archived version. Returns the restored
        skill or None if no history exists.

        The current live file is MOVED into history (as a new archive
        entry) so rollback is itself reversible by a second rollback.
        """
        history_dir = self._index.dir / HISTORY_SUBDIR
        if not history_dir.exists():
            return None
        candidates = sorted(
            history_dir.glob(f"{slug}_v*.md"),
            key=lambda p: _version_from_archive_name(p, slug),
            reverse=True,
        )
        if not candidates:
            return None

        # Move the current live file into history (if it exists) so this
        # rollback can itself be undone.
        live = self._index.dir / f"{slug}.md"
        if live.exists():
            try:
                current = read_skill(live)
                _archive(current, self._index.dir)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not archive current %r before rollback: %s", slug, exc)

        restored_path = candidates[0]
        try:
            restored = read_skill(restored_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to read archived skill %s: %s", restored_path, exc)
            return None
        # Archived filename is ``{slug}_v{n}.md`` so read_skill() populates
        # slug with e.g. ``greet_v1``. Restore the real slug before writing.
        restored = Skill(
            slug=slug,
            name=restored.name,
            description=restored.description,
            body=restored.body,
            version=restored.version,
            tags=list(restored.tags),
            expected_tool_count=restored.expected_tool_count,
            extra=dict(restored.extra),
        )
        self._index.register(restored)
        # Remove the archive entry we just restored FROM — otherwise
        # repeated rollbacks to same version would loop.
        restored_path.unlink()
        logger.info("Skill improver: rolled back %r to v%d", slug, restored.version)
        return restored


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _count_tool_calls(trajectory: list["TurnRecord"]) -> int:
    return sum(len(t.tool_calls) for t in trajectory)


def _format_trajectory(trajectory: list["TurnRecord"]) -> str:
    lines: list[str] = []
    for t in trajectory:
        for tc in t.tool_calls:
            args = str(tc.arguments)[:80]
            lines.append(f"{tc.name}({args})")
    return "\n".join(lines) if lines else "(no tool calls)"


_STEPS_HEADER_RE = re.compile(r"^##\s+Steps\s*$", re.MULTILINE)
_NEXT_HEADER_RE = re.compile(r"^##\s+", re.MULTILINE)


def _extract_steps_block(body: str) -> str:
    """Return just the Steps section of a skill body, or empty string."""
    m = _STEPS_HEADER_RE.search(body)
    if m is None:
        return ""
    start = m.end()
    rest = body[start:]
    next_m = _NEXT_HEADER_RE.search(rest)
    end = next_m.start() if next_m else len(rest)
    return rest[:end].strip()


def _rewrite_body(old_body: str, new_steps: list[str]) -> str:
    """Replace the Steps section of old_body with the new steps list."""
    m = _STEPS_HEADER_RE.search(old_body)
    steps_block = "\n".join(f"{i+1}. {s}" for i, s in enumerate(new_steps))
    if m is None:
        # No Steps section — append one
        return old_body.rstrip() + f"\n\n## Steps\n{steps_block}\n"
    start = m.end()
    rest = old_body[start:]
    next_m = _NEXT_HEADER_RE.search(rest)
    if next_m is None:
        return old_body[:start] + f"\n{steps_block}\n"
    end = start + next_m.start()
    return old_body[:start] + f"\n{steps_block}\n\n" + old_body[end:]


_JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}", re.DOTALL)


def _parse_response(text: str) -> dict[str, Any] | None:
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
    steps = obj.get("steps")
    if not isinstance(steps, list) or not steps:
        return None
    return {"steps": [str(s).strip() for s in steps if str(s).strip()]}


def _archive(skill: Skill, skills_dir: Path) -> Path:
    """Copy the skill to skills/.history/{slug}_v{version}.md."""
    history_dir = skills_dir / HISTORY_SUBDIR
    history_dir.mkdir(parents=True, exist_ok=True)
    path = history_dir / f"{skill.slug}_v{skill.version}.md"
    path.write_text(skill.render(), encoding="utf-8")
    return path


_ARCHIVE_NAME_RE = re.compile(r"_v(\d+)\.md$")


def _version_from_archive_name(path: Path, slug: str) -> int:
    m = _ARCHIVE_NAME_RE.search(path.name)
    return int(m.group(1)) if m else 0
