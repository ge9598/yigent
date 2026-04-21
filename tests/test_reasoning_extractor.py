"""Tests for reasoning extraction from OpenAI-protocol streams."""

from __future__ import annotations

from src.providers.reasoning_extractor import (
    ThinkTagStripper,
    extract_reasoning_content,
)


# ---------------------------------------------------------------------------
# Form #2 — delta.reasoning_content field
# ---------------------------------------------------------------------------

def test_extract_reasoning_content_returns_empty_for_none() -> None:
    assert extract_reasoning_content(None) == ""


def test_extract_reasoning_content_passes_through() -> None:
    assert extract_reasoning_content("step 1...") == "step 1..."


# ---------------------------------------------------------------------------
# Form #3 — <think>...</think> FSM
# ---------------------------------------------------------------------------

def test_stripper_pure_user_text_no_tags() -> None:
    s = ThinkTagStripper()
    user, reason = s.feed("hello world")
    assert user == "hello world"
    assert reason == ""


def test_stripper_single_chunk_with_full_tag_pair() -> None:
    s = ThinkTagStripper()
    user, reason = s.feed("prefix<think>thinking</think>answer")
    assert user == "prefixanswer"
    assert reason == "thinking"


def test_stripper_tag_split_across_chunks() -> None:
    """The <think> tag arrives split — FSM must buffer."""
    s = ThinkTagStripper()
    u1, r1 = s.feed("hi <thi")
    u2, r2 = s.feed("nk>inner</think>end")
    # First chunk holds back "<thi" as a possible tag prefix.
    assert u1 == "hi "
    assert r1 == ""
    # Second chunk completes the open tag, sees full close tag, emits end.
    assert u2 == "end"
    assert r2 == "inner"


def test_stripper_close_tag_split_across_chunks() -> None:
    s = ThinkTagStripper()
    u1, r1 = s.feed("<think>reasoning </th")
    u2, r2 = s.feed("ink>final")
    assert u1 == ""
    assert r1 == "reasoning "
    assert u2 == "final"
    assert r2 == ""


def test_stripper_multiple_think_blocks() -> None:
    s = ThinkTagStripper()
    user, reason = s.feed("a<think>one</think>b<think>two</think>c")
    assert user == "abc"
    assert reason == "onetwo"


def test_stripper_case_insensitive_tag_detection() -> None:
    s = ThinkTagStripper()
    user, reason = s.feed("x<THINK>upper</THINK>y")
    assert user == "xy"
    assert reason == "upper"


def test_stripper_finish_flushes_dangling_prefix() -> None:
    """Ambiguous trailing chars (could-be-a-tag) surface on finish()."""
    s = ThinkTagStripper()
    user, reason = s.feed("text<thi")
    assert user == "text"  # "<thi" held back
    assert reason == ""
    flush_user, flush_reason = s.finish()
    # Stream ended mid-carry — emit as user text (no tag completed).
    assert flush_user == "<thi"
    assert flush_reason == ""


def test_stripper_finish_flushes_unterminated_thinking() -> None:
    """Malformed: <think> opened but never closed. Don't lose the reasoning."""
    s = ThinkTagStripper()
    u1, r1 = s.feed("<think>never closed")
    assert u1 == ""
    assert r1 == "never closed"
    # Nothing in carry; finish is a no-op.
    flush_user, flush_reason = s.finish()
    assert flush_user == ""
    assert flush_reason == ""


def test_stripper_does_not_carry_unambiguous_tail() -> None:
    """Tail "hello" cannot be a tag prefix — emit it, don't carry."""
    s = ThinkTagStripper()
    user, reason = s.feed("hello")
    assert user == "hello"
    assert reason == ""
    # No carry remains.
    flush_user, flush_reason = s.finish()
    assert flush_user == ""
    assert flush_reason == ""
