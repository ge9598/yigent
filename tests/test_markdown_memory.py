"""Tests for MarkdownMemoryStore — CC-style MEMORY.md + per-topic files."""

from __future__ import annotations

from pathlib import Path

from src.memory.markdown_store import (
    INDEX_FILENAME, MAX_INDEX_LINES, MarkdownMemoryStore, Topic,
    _project_hash, _slugify, default_root,
)


# ---------------------------------------------------------------------------
# Slugging + project hashing
# ---------------------------------------------------------------------------

def test_slugify_basic() -> None:
    assert _slugify("Debugging Notes") == "debugging-notes"
    assert _slugify("API / Conventions") == "api-conventions"
    assert _slugify("   Whitespace   Galore   ") == "whitespace-galore"
    assert _slugify("---weird---chars!!!") == "weird-chars"
    assert _slugify("") == "untitled"
    assert _slugify("???") == "untitled"


def test_project_hash_stable_for_same_cwd(tmp_path: Path) -> None:
    h1 = _project_hash(tmp_path)
    h2 = _project_hash(tmp_path)
    assert h1 == h2
    assert len(h1) == 8


def test_project_hash_differs_for_different_cwds(tmp_path: Path) -> None:
    other = tmp_path.parent
    assert _project_hash(tmp_path) != _project_hash(other)


def test_default_root_includes_project_hash(tmp_path: Path) -> None:
    root = default_root(tmp_path)
    assert _project_hash(tmp_path) in str(root)
    assert ".yigent" in str(root)
    assert "memory" in str(root)


# ---------------------------------------------------------------------------
# Topic frontmatter round-trip
# ---------------------------------------------------------------------------

def test_topic_render_and_parse_roundtrip() -> None:
    t = Topic(
        slug="dbg", title="Debugging",
        created="2026-04-20T10:00:00", updated="2026-04-20T14:00:00",
        body="Found that X causes Y.",
    )
    rendered = t.render()
    parsed = Topic.parse("dbg", rendered)
    assert parsed.slug == "dbg"
    assert parsed.title == "Debugging"
    assert parsed.created == "2026-04-20T10:00:00"
    assert parsed.updated == "2026-04-20T14:00:00"
    assert parsed.body == "Found that X causes Y."


def test_topic_parse_tolerates_missing_frontmatter() -> None:
    t = Topic.parse("dbg", "Just plain body with no frontmatter.")
    assert t.slug == "dbg"
    assert t.title == "Dbg"
    assert "plain body" in t.body


# ---------------------------------------------------------------------------
# write + read + list + delete
# ---------------------------------------------------------------------------

def test_write_and_read_topic(tmp_path: Path) -> None:
    store = MarkdownMemoryStore(tmp_path)
    store.write_topic("Debugging", "Always check the logs first.")
    t = store.read_topic("Debugging")
    assert t is not None
    assert t.title == "Debugging"
    assert "check the logs" in t.body


def test_write_topic_persists_to_disk_as_readable_markdown(tmp_path: Path) -> None:
    store = MarkdownMemoryStore(tmp_path)
    store.write_topic("API Conventions", "Use snake_case for tool names.")
    path = tmp_path / "api-conventions.md"
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "title: API Conventions" in text
    assert "snake_case" in text


def test_write_topic_preserves_created_on_rewrite(tmp_path: Path) -> None:
    store = MarkdownMemoryStore(tmp_path)
    first = store.write_topic("note", "v1")
    second = store.write_topic("note", "v2")
    assert second.created == first.created
    assert second.updated >= first.updated  # monotonic non-decreasing
    assert second.body.startswith("v2")


def test_list_topics_excludes_index(tmp_path: Path) -> None:
    store = MarkdownMemoryStore(tmp_path)
    store.write_topic("alpha", "a")
    store.write_topic("beta", "b")
    store.record_index_entry("alpha", "alpha", "first")  # creates MEMORY.md
    slugs = store.list_topics()
    assert "alpha" in slugs
    assert "beta" in slugs
    assert INDEX_FILENAME.removesuffix(".md").lower() not in slugs
    assert "memory" not in slugs


def test_delete_topic_removes_file_and_index_entry(tmp_path: Path) -> None:
    store = MarkdownMemoryStore(tmp_path)
    store.write_topic("temp", "temporary content")
    store.record_index_entry("temp", "temp", "will be deleted")
    assert store.delete_topic("temp") is True
    assert store.read_topic("temp") is None
    index = store.read_index()
    assert "temp.md" not in index


def test_delete_missing_topic_returns_false(tmp_path: Path) -> None:
    store = MarkdownMemoryStore(tmp_path)
    assert store.delete_topic("ghost") is False


# ---------------------------------------------------------------------------
# MEMORY.md index behaviour
# ---------------------------------------------------------------------------

def test_record_index_entry_creates_header_on_first_call(tmp_path: Path) -> None:
    store = MarkdownMemoryStore(tmp_path)
    store.record_index_entry("debugging", "Debugging", "FTS trigger bug")
    index = store.read_index()
    assert index.startswith("# Memory index")
    assert "[Debugging](debugging.md)" in index
    assert "FTS trigger bug" in index


def test_record_index_entry_dedupes_by_slug(tmp_path: Path) -> None:
    """Rewriting the same topic should NOT leave two lines in MEMORY.md."""
    store = MarkdownMemoryStore(tmp_path)
    store.record_index_entry("dbg", "Debugging", "first hook")
    store.record_index_entry("dbg", "Debugging v2", "updated hook")
    index = store.read_index()
    # Only one pointer line should reference dbg.md.
    assert index.count("](dbg.md)") == 1
    assert "updated hook" in index
    assert "first hook" not in index


def test_read_index_caps_lines(tmp_path: Path) -> None:
    store = MarkdownMemoryStore(tmp_path)
    store.ensure_root()
    # Write 300 lines to MEMORY.md manually.
    path = tmp_path / INDEX_FILENAME
    path.write_text("\n".join(f"line {i}" for i in range(300)), encoding="utf-8")
    capped = store.read_index()
    assert capped.count("\n") <= MAX_INDEX_LINES + 1


def test_read_index_returns_empty_when_missing(tmp_path: Path) -> None:
    store = MarkdownMemoryStore(tmp_path)
    assert store.read_index() == ""


# ---------------------------------------------------------------------------
# Cross-session persistence (the user-facing feature)
# ---------------------------------------------------------------------------

def test_data_survives_across_store_instances(tmp_path: Path) -> None:
    first = MarkdownMemoryStore(tmp_path)
    first.write_topic("pi", "pi is approximately 3.14159")
    first.record_index_entry("pi", "Pi", "math constant")

    # Fresh instance, same directory.
    second = MarkdownMemoryStore(tmp_path)
    t = second.read_topic("pi")
    assert t is not None
    assert "3.14159" in t.body
    assert "pi" in second.list_topics()
    assert "Pi" in second.read_index()
