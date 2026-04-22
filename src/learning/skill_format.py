"""Skill file format — agentskills.io compatible.

A skill is a Markdown file with YAML frontmatter, stored under
``skills/{slug}.md``. The frontmatter carries machine-readable metadata
(name, description, version, tags, expected_tool_count); the body is
human-readable instructions to the agent. The format is compatible with
Claude Code's SKILL.md and Hermes Agent's skill files.

Minimal example::

    ---
    name: quicksort-with-tests
    description: Implement Python quicksort with 3 test cases
    version: 1
    tags: [coding]
    expected_tool_count: 6
    ---

    # Quicksort with tests

    ## When to use
    When the user asks for a quicksort implementation in Python.

    ## Steps
    1. Write the partition helper.
    2. Write the recursive quicksort.
    3. Add three unit tests covering empty, sorted, reverse inputs.

    ## Example
    Input: "Implement quicksort"
    Tools: write_file(quicksort.py), python_repl(pytest).

This module handles (de)serialization only. Matching / search / lifecycle
are in src/memory/skill_index.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Skill:
    """Parsed skill file — frontmatter fields + markdown body."""

    slug: str
    name: str
    description: str
    body: str
    version: int = 1
    tags: list[str] = field(default_factory=list)
    expected_tool_count: int | None = None
    # Opaque frontmatter fields we don't recognize are preserved on the
    # extra dict so round-tripping doesn't lose data from other tools.
    extra: dict[str, Any] = field(default_factory=dict)

    def render(self) -> str:
        """Emit the skill back to SKILL.md text (YAML frontmatter + body)."""
        front: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "version": self.version,
        }
        if self.tags:
            front["tags"] = list(self.tags)
        if self.expected_tool_count is not None:
            front["expected_tool_count"] = self.expected_tool_count
        for k, v in self.extra.items():
            if k not in front:
                front[k] = v
        yaml_text = yaml.safe_dump(front, sort_keys=False, allow_unicode=True).strip()
        body_text = self.body.rstrip() + "\n"
        return f"---\n{yaml_text}\n---\n\n{body_text}"

    @classmethod
    def parse(cls, slug: str, text: str) -> "Skill":
        """Parse a SKILL.md file. Raises SkillFormatError on malformed input."""
        if not text.startswith("---"):
            raise SkillFormatError(
                f"Skill {slug!r} missing YAML frontmatter (must start with ---)"
            )
        try:
            _, front_text, body = text.split("---", 2)
        except ValueError as exc:
            raise SkillFormatError(
                f"Skill {slug!r} frontmatter not closed with ---"
            ) from exc
        try:
            front = yaml.safe_load(front_text) or {}
        except yaml.YAMLError as exc:
            raise SkillFormatError(
                f"Skill {slug!r} frontmatter is not valid YAML: {exc}"
            ) from exc
        if not isinstance(front, dict):
            raise SkillFormatError(f"Skill {slug!r} frontmatter must be a mapping")

        name = front.pop("name", None)
        description = front.pop("description", None)
        if not isinstance(name, str) or not name.strip():
            raise SkillFormatError(f"Skill {slug!r} missing 'name' field")
        if not isinstance(description, str) or not description.strip():
            raise SkillFormatError(f"Skill {slug!r} missing 'description' field")
        version = int(front.pop("version", 1) or 1)
        tags_raw = front.pop("tags", []) or []
        if isinstance(tags_raw, str):
            tags = [tags_raw]
        elif isinstance(tags_raw, list):
            tags = [str(t) for t in tags_raw]
        else:
            tags = []
        etc = front.pop("expected_tool_count", None)
        expected_tool_count = int(etc) if etc is not None else None

        return cls(
            slug=slug,
            name=name.strip(),
            description=description.strip(),
            body=body.lstrip("\n"),
            version=version,
            tags=tags,
            expected_tool_count=expected_tool_count,
            extra=front,
        )


class SkillFormatError(ValueError):
    """Raised on malformed SKILL.md content."""


def read_skill(path: Path) -> Skill:
    """Load and parse a single SKILL.md file."""
    text = Path(path).read_text(encoding="utf-8")
    slug = Path(path).stem
    return Skill.parse(slug, text)


def write_skill(skill: Skill, skills_dir: Path) -> Path:
    """Write a skill to ``skills_dir/{slug}.md``. Returns the written path."""
    skills_dir = Path(skills_dir)
    skills_dir.mkdir(parents=True, exist_ok=True)
    out_path = skills_dir / f"{skill.slug}.md"
    out_path.write_text(skill.render(), encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# Matching helpers — shared between skill_index and skill_creator
# ---------------------------------------------------------------------------


def tokenize(text: str) -> set[str]:
    """Very simple tokenizer for Jaccard similarity. Lowercases and splits
    on non-alphanumeric. ASCII-only for speed; skill descriptions are
    English in practice."""
    if not text:
        return set()
    out: set[str] = set()
    current: list[str] = []
    for ch in text.lower():
        if ch.isalnum():
            current.append(ch)
        else:
            if current:
                out.add("".join(current))
                current = []
    if current:
        out.add("".join(current))
    return out - _STOPWORDS


_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "he", "in", "is", "it", "its", "of", "on", "or", "that", "the", "to",
    "was", "were", "will", "with", "this",
})


def jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity in [0, 1]."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0
