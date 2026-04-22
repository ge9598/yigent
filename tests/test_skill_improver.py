"""Tests for SkillImprover (Unit 3b — Phase 3)."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.core.types import Message, StreamChunk, ToolCall, ToolResult
from src.learning.skill_format import Skill
from src.learning.skill_improver import (
    SkillImprover, _extract_steps_block, _rewrite_body,
)
from src.learning.trajectory import TrajectoryRecorder
from src.memory.skill_index import SkillIndex


# ---------------------------------------------------------------------------
# Body helpers
# ---------------------------------------------------------------------------


def test_extract_steps_block():
    body = """# Skill

## When to use
do a thing

## Steps
1. first
2. second

## Example
x"""
    block = _extract_steps_block(body)
    assert "1. first" in block
    assert "## Example" not in block


def test_rewrite_body_replaces_steps():
    body = "# X\n\n## Steps\n1. old1\n2. old2\n\n## Example\nxyz"
    new = _rewrite_body(body, ["new1", "new2", "new3"])
    assert "1. new1" in new
    assert "2. new2" in new
    assert "3. new3" in new
    assert "old1" not in new
    assert "## Example" in new  # preserved


def test_rewrite_body_appends_when_missing():
    body = "# X\n\nbody text only\n"
    new = _rewrite_body(body, ["s1"])
    assert "## Steps" in new
    assert "1. s1" in new


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------


def _trajectory_with_calls(n: int) -> list:
    rec = TrajectoryRecorder(session_id="s")
    for i in range(n):
        rec.record_turn(
            assistant_msg=Message(role="assistant", content=None),
            tool_calls=[ToolCall(id=f"c{i}", name="t", arguments={})],
            tool_results=[ToolResult(tool_call_id=f"c{i}", name="t", content="ok")],
        )
    return rec.turns


def _existing_skill(expected: int = 10) -> Skill:
    return Skill(
        slug="greet",
        name="Greet",
        description="greet someone",
        body="# Greet\n\n## Steps\n1. call greet\n2. read response\n",
        version=1,
        tags=["coding"],
        expected_tool_count=expected,
    )


async def test_gate_blocks_on_non_success(tmp_path: Path):
    idx = SkillIndex(tmp_path)
    improver = SkillImprover(MagicMock(), idx)
    result = await improver.maybe_improve(
        _existing_skill(), _trajectory_with_calls(3), outcome="error",
    )
    assert result is None


async def test_gate_blocks_when_no_expected_count(tmp_path: Path):
    idx = SkillIndex(tmp_path)
    improver = SkillImprover(MagicMock(), idx)
    skill = _existing_skill()
    skill.expected_tool_count = None
    result = await improver.maybe_improve(
        skill, _trajectory_with_calls(3), outcome="success",
    )
    assert result is None


async def test_gate_blocks_on_not_enough_improvement(tmp_path: Path):
    idx = SkillIndex(tmp_path)
    improver = SkillImprover(MagicMock(), idx, improvement_ratio=0.8)
    # Skill expects 10, actual is 9 (0.9 ratio, not below 0.8 threshold)
    result = await improver.maybe_improve(
        _existing_skill(10), _trajectory_with_calls(9), outcome="success",
    )
    assert result is None


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def _mock_provider(response: str):
    provider = MagicMock()

    async def stream_message(**kwargs) -> AsyncGenerator:
        yield StreamChunk(type="token", data=response)
        yield StreamChunk(type="done", data="stop")

    provider.stream_message = stream_message
    return provider


async def test_improve_writes_new_version_and_archives_old(tmp_path: Path):
    idx = SkillIndex(tmp_path)
    old = _existing_skill(expected=10)
    idx.register(old)

    provider = _mock_provider(
        '{"steps": ["shorter1", "shorter2"], "reason": "batched calls"}'
    )
    improver = SkillImprover(provider, idx)

    new = await improver.maybe_improve(
        old, _trajectory_with_calls(5), outcome="success",
    )
    assert new is not None
    assert new.version == 2
    assert new.expected_tool_count == 5
    assert "1. shorter1" in new.body

    # Archive exists
    archive = tmp_path / ".history" / "greet_v1.md"
    assert archive.exists()
    # Live file is v2
    live = (tmp_path / "greet.md").read_text(encoding="utf-8")
    assert "version: 2" in live
    assert "shorter1" in live


async def test_aux_null_response_skips(tmp_path: Path):
    idx = SkillIndex(tmp_path)
    old = _existing_skill(expected=10)
    idx.register(old)

    improver = SkillImprover(_mock_provider("null"), idx)
    new = await improver.maybe_improve(
        old, _trajectory_with_calls(5), outcome="success",
    )
    assert new is None
    # No archive, live file unchanged
    assert not (tmp_path / ".history").exists() or not list(
        (tmp_path / ".history").glob("*")
    )


async def test_aux_failure_is_swallowed(tmp_path: Path):
    idx = SkillIndex(tmp_path)
    old = _existing_skill(expected=10)
    idx.register(old)

    provider = MagicMock()

    async def stream_message(**kwargs):
        raise RuntimeError("boom")
        yield  # pragma: no cover

    provider.stream_message = stream_message
    improver = SkillImprover(provider, idx)

    new = await improver.maybe_improve(
        old, _trajectory_with_calls(5), outcome="success",
    )
    assert new is None


# ---------------------------------------------------------------------------
# Rollback
# ---------------------------------------------------------------------------


async def test_rollback_restores_previous_version(tmp_path: Path):
    idx = SkillIndex(tmp_path)
    old = _existing_skill(expected=10)
    idx.register(old)

    # Improve to v2
    provider = _mock_provider(
        '{"steps": ["new-step"], "reason": "shorter"}'
    )
    improver = SkillImprover(provider, idx)
    new = await improver.maybe_improve(
        old, _trajectory_with_calls(5), outcome="success",
    )
    assert new is not None and new.version == 2

    # Roll back
    restored = improver.rollback_to_previous("greet")
    assert restored is not None
    assert restored.version == 1
    # Live file is now v1 content
    live = (tmp_path / "greet.md").read_text(encoding="utf-8")
    assert "version: 1" in live
    assert "call greet" in live
    # Index reflects v1
    assert idx.load("greet").version == 1


async def test_rollback_without_history_returns_none(tmp_path: Path):
    idx = SkillIndex(tmp_path)
    improver = SkillImprover(MagicMock(), idx)
    result = improver.rollback_to_previous("nonexistent")
    assert result is None


async def test_rollback_picks_highest_version(tmp_path: Path):
    """If multiple history entries exist, the latest archived version wins."""
    idx = SkillIndex(tmp_path)
    old = _existing_skill(expected=10)
    idx.register(old)
    improver = SkillImprover(MagicMock(), idx)

    # Manually write two archive entries
    (tmp_path / ".history").mkdir()
    v1 = Skill(slug="greet", name="G1", description="d", body="# v1", version=1,
               expected_tool_count=10)
    v2 = Skill(slug="greet", name="G2", description="d", body="# v2", version=2,
               expected_tool_count=5)
    (tmp_path / ".history" / "greet_v1.md").write_text(v1.render(), encoding="utf-8")
    (tmp_path / ".history" / "greet_v2.md").write_text(v2.render(), encoding="utf-8")

    restored = improver.rollback_to_previous("greet")
    assert restored is not None
    assert restored.version == 2  # Highest archived


async def test_no_provider_skips_gracefully(tmp_path: Path):
    idx = SkillIndex(tmp_path)
    old = _existing_skill(expected=10)
    idx.register(old)
    improver = SkillImprover(None, idx)
    result = await improver.maybe_improve(
        old, _trajectory_with_calls(5), outcome="success",
    )
    assert result is None
