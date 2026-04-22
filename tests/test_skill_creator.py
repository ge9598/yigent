"""Tests for SkillCreator (Unit 3 — Phase 3)."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.core.types import Message, StreamChunk, ToolCall, ToolResult
from src.learning.skill_creator import (
    SkillCreator, _build_skill, _parse_response, _sanitize_slug,
)
from src.learning.skill_format import Skill
from src.learning.trajectory import TrajectoryRecorder
from src.memory.skill_index import SkillIndex


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_sanitize_slug_lowercases_and_hyphenates():
    assert _sanitize_slug("Refactor Loop!") == "refactor-loop"
    assert _sanitize_slug("  --x--  ") == "x"
    assert _sanitize_slug("") == "skill"
    assert len(_sanitize_slug("a" * 100)) <= 40


def test_parse_response_valid():
    text = (
        '{"slug": "qs", "name": "Quicksort", "description": "d", '
        '"steps": ["s1", "s2"], "tags": ["coding"]}'
    )
    parsed = _parse_response(text)
    assert parsed is not None
    assert parsed["slug"] == "qs"
    assert parsed["steps"] == ["s1", "s2"]


def test_parse_response_missing_steps():
    text = '{"slug": "x", "name": "n", "description": "d"}'
    assert _parse_response(text) is None


def test_parse_response_empty_steps():
    text = '{"slug": "x", "name": "n", "description": "d", "steps": []}'
    assert _parse_response(text) is None


def test_parse_response_null():
    assert _parse_response("null") is None


def test_build_skill_filters_invalid_tags():
    parsed = {
        "slug": "x",
        "name": "X",
        "description": "d",
        "steps": ["a", "b"],
        "tags": ["coding", "invented-tag", "research"],
    }
    skill = _build_skill(parsed, expected_tool_count=5)
    assert skill.tags == ["coding", "research"]
    assert skill.expected_tool_count == 5


def test_build_skill_body_includes_sections():
    parsed = {
        "slug": "x", "name": "X Thing", "description": "d",
        "steps": ["first", "second"],
        "when_to_use": "when asked",
        "example_input": "do x",
    }
    skill = _build_skill(parsed, expected_tool_count=3)
    assert "# X Thing" in skill.body
    assert "## When to use" in skill.body
    assert "1. first" in skill.body
    assert "## Example" in skill.body


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------


def _trajectory_with(n_calls: int, distinct_tools: int) -> list:
    """Build a fake trajectory with exactly n_calls tool calls across
    distinct_tools distinct tools."""
    rec = TrajectoryRecorder(session_id="s")
    rec.record_turn(
        user_msg=Message(role="user", content="please do stuff"),
        assistant_msg=Message(role="assistant", content=None),
    )
    for i in range(n_calls):
        tool_name = f"tool{i % distinct_tools}"
        rec.record_turn(
            assistant_msg=Message(role="assistant", content=None),
            tool_calls=[ToolCall(id=f"c{i}", name=tool_name, arguments={})],
            tool_results=[ToolResult(tool_call_id=f"c{i}", name=tool_name, content="ok")],
        )
    return rec.turns


async def test_gate_blocks_on_not_success(tmp_path: Path):
    idx = SkillIndex(tmp_path)
    creator = SkillCreator(aux_provider=MagicMock(), skill_index=idx)
    result = await creator.maybe_create_skill(
        _trajectory_with(5, 3), outcome="error",
    )
    assert result is None


async def test_gate_blocks_on_too_few_tool_calls(tmp_path: Path):
    idx = SkillIndex(tmp_path)
    creator = SkillCreator(aux_provider=MagicMock(), skill_index=idx)
    result = await creator.maybe_create_skill(
        _trajectory_with(2, 2), outcome="success",
    )
    assert result is None


async def test_gate_blocks_on_single_tool(tmp_path: Path):
    idx = SkillIndex(tmp_path)
    creator = SkillCreator(aux_provider=MagicMock(), skill_index=idx)
    result = await creator.maybe_create_skill(
        _trajectory_with(6, 1), outcome="success",
    )
    assert result is None


# ---------------------------------------------------------------------------
# Creation — happy path
# ---------------------------------------------------------------------------


def _mock_provider(response: str):
    provider = MagicMock()

    async def stream_message(**kwargs) -> AsyncGenerator:
        yield StreamChunk(type="token", data=response)
        yield StreamChunk(type="done", data="stop")

    provider.stream_message = stream_message
    return provider


async def test_happy_path_creates_and_registers(tmp_path: Path):
    idx = SkillIndex(tmp_path)
    provider = _mock_provider(
        '{"slug": "Greet-Workflow", "name": "Greet", '
        '"description": "Greet someone using the greet tool", '
        '"steps": ["call greet", "read response"], '
        '"tags": ["coding"], "when_to_use": "greet someone"}'
    )
    creator = SkillCreator(provider, idx)

    skill = await creator.maybe_create_skill(
        _trajectory_with(4, 2), outcome="success",
    )
    assert skill is not None
    assert skill.slug == "greet-workflow"
    assert skill.expected_tool_count == 4
    # File on disk
    assert (tmp_path / "greet-workflow.md").exists()
    # Registered in index
    assert "greet-workflow" in idx


async def test_dedup_skips_similar_existing_skill(tmp_path: Path):
    idx = SkillIndex(tmp_path)
    # Seed the index with an existing skill
    idx.register(Skill(
        slug="greet-existing",
        name="Greet",
        description="please do stuff",  # matches user request
        body="body", tags=["coding"],
    ))
    provider = _mock_provider(
        '{"slug": "new-skill", "name": "New", '
        '"description": "d", "steps": ["s1", "s2"]}'
    )
    creator = SkillCreator(provider, idx, dedup_threshold=0.3)

    result = await creator.maybe_create_skill(
        _trajectory_with(4, 2), outcome="success",
    )
    assert result is None  # dedup'd before aux call


async def test_aux_null_response_skips(tmp_path: Path):
    idx = SkillIndex(tmp_path)
    provider = _mock_provider("null")
    creator = SkillCreator(provider, idx)

    result = await creator.maybe_create_skill(
        _trajectory_with(4, 2), outcome="success",
    )
    assert result is None
    assert len(idx) == 0


async def test_aux_failure_is_swallowed(tmp_path: Path):
    idx = SkillIndex(tmp_path)
    provider = MagicMock()

    async def stream_message(**kwargs):
        raise RuntimeError("boom")
        yield  # pragma: no cover

    provider.stream_message = stream_message
    creator = SkillCreator(provider, idx)

    result = await creator.maybe_create_skill(
        _trajectory_with(4, 2), outcome="success",
    )
    assert result is None
    assert len(idx) == 0


async def test_no_provider_skips_gracefully(tmp_path: Path):
    idx = SkillIndex(tmp_path)
    creator = SkillCreator(aux_provider=None, skill_index=idx)
    result = await creator.maybe_create_skill(
        _trajectory_with(4, 2), outcome="success",
    )
    assert result is None
