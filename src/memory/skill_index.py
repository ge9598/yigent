"""Skill registry — scans ``skills/`` and answers "what skill fits this task?"

The index is a thin wrapper around the filesystem. It loads all SKILL.md
files under ``skills_dir``, caches their parsed metadata, and exposes three
operations:

- ``rebuild()`` — re-scan the directory (call after a skill_creator writes
  a new skill, or at session start)
- ``search(query, k)`` — rank skills by Jaccard similarity on
  (description + tags) vs. query; return the top-k
- ``register(skill)`` — write a new skill to disk and add to cache

Matching is intentionally simple (token-set Jaccard). Vector embeddings
would be nice but L2 semantic memory was dropped from project scope. For
the Phase 3 scale (dozens of skills), Jaccard is adequate — if the index
grows to hundreds of skills, bolt on BM25 or a lightweight embedding.

Consumption by ContextAssembler (Zone 2 skill hints) is deferred — the
index is usable standalone by skill_creator (dedup) and skill_improver
(version tracking).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from src.learning.skill_format import (
    Skill, SkillFormatError, jaccard, read_skill, tokenize, write_skill,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SkillMeta:
    """Lightweight skill descriptor for search results.

    Held separately from ``Skill`` so search doesn't have to retain the full
    body text in memory for large skill libraries.
    """

    slug: str
    name: str
    description: str
    version: int
    tags: tuple[str, ...]
    score: float = 0.0  # populated by search()


class SkillIndex:
    def __init__(self, skills_dir: Path | str) -> None:
        self._dir = Path(skills_dir)
        self._cache: dict[str, Skill] = {}
        self._tokens: dict[str, set[str]] = {}
        # mtime-per-slug snapshot for incremental rebuild (audit B10 / Top10 #17).
        self._mtimes: dict[str, float] = {}

    @property
    def dir(self) -> Path:
        return self._dir

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, slug: str) -> bool:
        return slug in self._cache

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def rebuild(self) -> None:
        """Re-scan ``skills_dir``. Missing directory is treated as empty."""
        self._cache.clear()
        self._tokens.clear()
        self._mtimes.clear()
        if not self._dir.exists():
            return
        for path in sorted(self._dir.glob("*.md")):
            try:
                skill = read_skill(path)
            except SkillFormatError as exc:
                logger.warning("Skipping malformed skill %s: %s", path.name, exc)
                continue
            self._cache[skill.slug] = skill
            self._tokens[skill.slug] = _skill_tokens(skill)
            try:
                self._mtimes[skill.slug] = path.stat().st_mtime
            except OSError:
                pass

    def rebuild_incremental(self) -> int:
        """Refresh only changed/new/deleted skill files. Returns the number
        of slugs touched (added + updated + removed).

        Cheaper than ``rebuild()`` when the directory holds many skills but
        only a handful changed since last scan. Audit B10 / Top10 #17.
        Falls back to a full ``rebuild()`` on the first call (when there's
        no mtime baseline).
        """
        if not self._mtimes:
            self.rebuild()
            return len(self._cache)
        if not self._dir.exists():
            n = len(self._cache)
            self._cache.clear()
            self._tokens.clear()
            self._mtimes.clear()
            return n

        seen_slugs: set[str] = set()
        touched = 0
        for path in self._dir.glob("*.md"):
            slug = path.stem
            seen_slugs.add(slug)
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            if mtime == self._mtimes.get(slug):
                continue  # unchanged
            try:
                skill = read_skill(path)
            except SkillFormatError as exc:
                logger.warning("Skipping malformed skill %s: %s", path.name, exc)
                continue
            self._cache[skill.slug] = skill
            self._tokens[skill.slug] = _skill_tokens(skill)
            self._mtimes[skill.slug] = mtime
            touched += 1

        # Drop deleted-on-disk slugs from cache
        for stale in set(self._cache) - seen_slugs:
            self._cache.pop(stale, None)
            self._tokens.pop(stale, None)
            self._mtimes.pop(stale, None)
            touched += 1
        return touched

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def load(self, slug: str) -> Skill:
        """Return a cached skill. Raises KeyError if missing."""
        if slug not in self._cache:
            # Try one lazy re-read before giving up — new skill might have
            # been written by another agent process.
            path = self._dir / f"{slug}.md"
            if path.exists():
                skill = read_skill(path)
                self._cache[skill.slug] = skill
                self._tokens[skill.slug] = _skill_tokens(skill)
        return self._cache[slug]

    def search(self, query: str, k: int = 3) -> list[SkillMeta]:
        """Return top-k skills by Jaccard similarity on query vs.
        (description + tags). Ties broken by slug for determinism.
        Skills with score 0 are filtered out.
        """
        q_tokens = tokenize(query)
        if not q_tokens or not self._cache:
            return []
        scored: list[tuple[float, Skill]] = []
        for slug, skill in self._cache.items():
            score = jaccard(q_tokens, self._tokens[slug])
            if score > 0:
                scored.append((score, skill))
        scored.sort(key=lambda t: (-t[0], t[1].slug))
        return [_to_meta(s, score=sc) for sc, s in scored[:k]]

    def all_meta(self) -> list[SkillMeta]:
        """Return metadata for every cached skill (no ranking)."""
        return [_to_meta(s) for s in sorted(
            self._cache.values(), key=lambda s: s.slug
        )]

    # ------------------------------------------------------------------
    # Dedup support for skill_creator
    # ------------------------------------------------------------------

    def find_similar(
        self, description: str, tags: list[str] | None = None,
        threshold: float = 0.6,
    ) -> SkillMeta | None:
        """Return the best-matching skill if its score >= threshold, else None.

        Used by skill_creator to decide "is this a duplicate of an existing
        skill?" before writing. Compares against (description + tags).
        """
        query = description + " " + " ".join(tags or [])
        results = self.search(query, k=1)
        if not results:
            return None
        best = results[0]
        return best if best.score >= threshold else None

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def register(self, skill: Skill) -> Path:
        """Write a skill to disk and add to cache. Overwrites existing file
        with the same slug."""
        path = write_skill(skill, self._dir)
        self._cache[skill.slug] = skill
        self._tokens[skill.slug] = _skill_tokens(skill)
        try:
            self._mtimes[skill.slug] = path.stat().st_mtime
        except OSError:
            pass
        return path

    def unregister(self, slug: str) -> bool:
        """Remove a skill from disk and cache. Returns True if a file was
        actually removed."""
        removed = False
        path = self._dir / f"{slug}.md"
        if path.exists():
            path.unlink()
            removed = True
        self._cache.pop(slug, None)
        self._tokens.pop(slug, None)
        self._mtimes.pop(slug, None)
        return removed


def _skill_tokens(skill: Skill) -> set[str]:
    """Token set used for similarity: description + tags + name."""
    combined = " ".join([skill.name, skill.description, " ".join(skill.tags)])
    return tokenize(combined)


def _to_meta(skill: Skill, score: float = 0.0) -> SkillMeta:
    return SkillMeta(
        slug=skill.slug,
        name=skill.name,
        description=skill.description,
        version=skill.version,
        tags=tuple(skill.tags),
        score=score,
    )
