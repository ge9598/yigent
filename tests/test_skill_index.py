"""Tests for SkillIndex (Unit 4 — Phase 3) and Skill format."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.learning.skill_format import (
    Skill, SkillFormatError, jaccard, read_skill, tokenize,
)
from src.memory.skill_index import SkillIndex


# ---------------------------------------------------------------------------
# skill_format — parsing and rendering
# ---------------------------------------------------------------------------


def test_skill_render_round_trip():
    s = Skill(
        slug="quicksort",
        name="Quicksort",
        description="Implement Python quicksort",
        body="# Quicksort\n\n## Steps\n1. Partition\n2. Recurse\n",
        version=1,
        tags=["coding", "algorithms"],
        expected_tool_count=5,
    )
    text = s.render()
    assert text.startswith("---\n")
    assert "name: Quicksort" in text
    assert "tags:" in text
    assert "expected_tool_count: 5" in text

    parsed = Skill.parse("quicksort", text)
    assert parsed.name == "Quicksort"
    assert parsed.description == "Implement Python quicksort"
    assert parsed.tags == ["coding", "algorithms"]
    assert parsed.expected_tool_count == 5
    assert "Partition" in parsed.body


def test_skill_parse_minimal():
    text = (
        "---\n"
        "name: t\n"
        "description: a test\n"
        "---\n"
        "\nbody\n"
    )
    s = Skill.parse("t", text)
    assert s.version == 1
    assert s.tags == []
    assert s.expected_tool_count is None
    assert s.body.strip() == "body"


def test_skill_parse_missing_frontmatter_raises():
    with pytest.raises(SkillFormatError, match="missing YAML frontmatter"):
        Skill.parse("x", "no frontmatter here")


def test_skill_parse_missing_name_raises():
    text = "---\ndescription: d\n---\nbody"
    with pytest.raises(SkillFormatError, match="missing 'name'"):
        Skill.parse("x", text)


def test_skill_parse_missing_description_raises():
    text = "---\nname: n\n---\nbody"
    with pytest.raises(SkillFormatError, match="missing 'description'"):
        Skill.parse("x", text)


def test_skill_parse_preserves_extra_frontmatter():
    text = (
        "---\n"
        "name: t\n"
        "description: d\n"
        "custom_field: preserved\n"
        "---\n"
        "body\n"
    )
    s = Skill.parse("t", text)
    assert s.extra.get("custom_field") == "preserved"
    # And it survives a round-trip
    re_parsed = Skill.parse("t", s.render())
    assert re_parsed.extra.get("custom_field") == "preserved"


def test_tokenize_strips_stopwords_and_punct():
    tokens = tokenize("The quick, brown fox — jumps over!")
    assert "quick" in tokens
    assert "brown" in tokens
    assert "the" not in tokens   # stopword
    assert "" not in tokens


def test_jaccard_empty_sets():
    assert jaccard(set(), set()) == 1.0
    assert jaccard({"a"}, set()) == 0.0


def test_jaccard_partial_overlap():
    a = {"x", "y", "z"}
    b = {"y", "z", "w"}
    # intersection = 2, union = 4
    assert jaccard(a, b) == 0.5


# ---------------------------------------------------------------------------
# SkillIndex — integration with filesystem
# ---------------------------------------------------------------------------


def _make_skill(slug: str, description: str, tags: list[str] | None = None) -> Skill:
    return Skill(
        slug=slug,
        name=slug.replace("-", " ").title(),
        description=description,
        body=f"# {slug}\n\n## Steps\n1. foo\n",
        tags=tags or [],
    )


def test_index_rebuild_on_empty_dir(tmp_path: Path):
    idx = SkillIndex(tmp_path / "skills")
    idx.rebuild()
    assert len(idx) == 0
    assert idx.search("anything") == []


def test_index_rebuild_loads_skills(tmp_path: Path):
    idx = SkillIndex(tmp_path)
    idx.register(_make_skill("quicksort-tests", "quicksort with tests", ["coding"]))
    idx.register(_make_skill("csv-anomaly", "find anomalies in CSV", ["data"]))

    # Fresh index, same dir
    idx2 = SkillIndex(tmp_path)
    idx2.rebuild()
    assert len(idx2) == 2
    assert "quicksort-tests" in idx2


def test_index_register_writes_file(tmp_path: Path):
    idx = SkillIndex(tmp_path)
    skill = _make_skill("refactor-loop", "refactor a for-loop", ["coding"])
    path = idx.register(skill)
    assert path == tmp_path / "refactor-loop.md"
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "refactor-loop" in text or "Refactor Loop" in text


def test_index_search_ranks_by_relevance(tmp_path: Path):
    idx = SkillIndex(tmp_path)
    idx.register(_make_skill("quicksort", "quicksort python tests", ["coding"]))
    idx.register(_make_skill("csv-anom", "csv anomaly detection", ["data"]))
    idx.register(_make_skill("mergesort", "mergesort with tests", ["coding"]))

    results = idx.search("write quicksort tests", k=3)
    assert len(results) > 0
    # quicksort must rank above csv-anom
    slugs = [r.slug for r in results]
    assert slugs.index("quicksort") < slugs.index("mergesort") \
           if "mergesort" in slugs else True
    assert "csv-anom" not in slugs or slugs.index("quicksort") < slugs.index("csv-anom")


def test_index_search_filters_zero_score(tmp_path: Path):
    idx = SkillIndex(tmp_path)
    idx.register(_make_skill("x", "completely unrelated description", ["bucket"]))
    results = idx.search("quicksort tests")
    assert results == []


def test_index_find_similar_below_threshold(tmp_path: Path):
    idx = SkillIndex(tmp_path)
    idx.register(_make_skill("x", "implement quicksort python tests", ["coding"]))
    similar = idx.find_similar("completely different stuff here", threshold=0.6)
    assert similar is None


def test_index_find_similar_above_threshold(tmp_path: Path):
    idx = SkillIndex(tmp_path)
    idx.register(_make_skill("qs", "quicksort python tests algorithm", ["coding"]))
    similar = idx.find_similar("quicksort python tests algorithm", threshold=0.6)
    assert similar is not None
    assert similar.slug == "qs"


def test_index_unregister_removes_file_and_cache(tmp_path: Path):
    idx = SkillIndex(tmp_path)
    idx.register(_make_skill("tmp", "temp skill", ["x"]))
    assert "tmp" in idx
    removed = idx.unregister("tmp")
    assert removed is True
    assert "tmp" not in idx
    assert not (tmp_path / "tmp.md").exists()


def test_index_skips_malformed_skill(tmp_path: Path, caplog):
    idx = SkillIndex(tmp_path)
    good = _make_skill("good", "valid skill", ["x"])
    idx.register(good)
    # Write a malformed file directly
    (tmp_path / "broken.md").write_text("no frontmatter", encoding="utf-8")
    idx.rebuild()
    assert "good" in idx
    assert "broken" not in idx
