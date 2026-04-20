"""Tests for L0 WorkingMemory."""

from __future__ import annotations

from src.memory.working import WorkingMemory


def test_starts_empty() -> None:
    wm = WorkingMemory()
    assert wm.conversation == []
    assert wm.todo == []
    assert wm.turn_count == 0


def test_append_and_extend() -> None:
    wm = WorkingMemory()
    wm.append({"role": "user", "content": "hi"})
    wm.extend([
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "again"},
    ])
    assert len(wm.conversation) == 3
    assert wm.turn_count == 2  # two user msgs


def test_last_user_text_returns_most_recent() -> None:
    wm = WorkingMemory()
    wm.append({"role": "user", "content": "first"})
    wm.append({"role": "assistant", "content": "ack"})
    wm.append({"role": "user", "content": "second"})
    assert wm.last_user_text() == "second"


def test_last_user_text_empty_when_no_user_msgs() -> None:
    wm = WorkingMemory()
    wm.append({"role": "assistant", "content": "hello"})
    assert wm.last_user_text() == ""


def test_todo_lifecycle() -> None:
    wm = WorkingMemory()
    wm.add_todo("a")
    wm.add_todo("b")
    assert wm.todo == ["a", "b"]
    assert wm.complete_todo("a") is True
    assert wm.todo == ["b"]
    assert wm.complete_todo("missing") is False
    wm.clear_todo()
    assert wm.todo == []
