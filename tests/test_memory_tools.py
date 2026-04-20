"""Tests for memory tools (list / read / write / delete)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.core.config import AgentConfig
from src.core.plan_mode import PlanMode
from src.core.types import ToolContext
from src.memory.markdown_store import MarkdownMemoryStore
from src.tools.registry import ToolRegistry

# Importing triggers registration.
import src.tools  # noqa: F401


def _ctx(tmp_path: Path) -> ToolContext:
    store = MarkdownMemoryStore(tmp_path / "mem")
    return ToolContext(
        plan_mode=PlanMode(save_dir=str(tmp_path / "plans")),
        registry=ToolRegistry(),
        config=AgentConfig(),
        working_dir=tmp_path,
        memory_store=store,
    )


@pytest.mark.asyncio
async def test_list_memory_empty(tmp_path) -> None:
    from src.tools.memory_tools import _list_memory_handler
    ctx = _ctx(tmp_path)
    result = await _list_memory_handler(ctx)
    assert "no memory topics" in result.lower()


@pytest.mark.asyncio
async def test_write_then_list_then_read(tmp_path) -> None:
    from src.tools.memory_tools import (
        _list_memory_handler, _read_memory_handler, _write_memory_handler,
    )
    ctx = _ctx(tmp_path)

    w = await _write_memory_handler(
        ctx, topic="pi", content="pi is roughly 3.14159",
        hook="math constant",
    )
    assert "Saved memory topic 'pi'" in w

    l = await _list_memory_handler(ctx)
    assert "pi" in l

    r = await _read_memory_handler(ctx, topic="pi")
    assert "3.14159" in r


@pytest.mark.asyncio
async def test_read_missing_topic(tmp_path) -> None:
    from src.tools.memory_tools import _read_memory_handler
    ctx = _ctx(tmp_path)
    result = await _read_memory_handler(ctx, topic="ghost")
    assert "Error" in result
    assert "no memory topic" in result.lower()


@pytest.mark.asyncio
async def test_delete_memory(tmp_path) -> None:
    from src.tools.memory_tools import (
        _delete_memory_handler, _write_memory_handler, _list_memory_handler,
    )
    ctx = _ctx(tmp_path)
    await _write_memory_handler(
        ctx, topic="temp", content="ephemeral", hook="will delete",
    )
    d = await _delete_memory_handler(ctx, topic="temp")
    assert "Deleted" in d
    l = await _list_memory_handler(ctx)
    assert "temp" not in l


@pytest.mark.asyncio
async def test_write_memory_requires_topic_and_content(tmp_path) -> None:
    from src.tools.memory_tools import _write_memory_handler
    ctx = _ctx(tmp_path)
    out = await _write_memory_handler(ctx, topic="", content="body")
    assert "Error" in out
    out = await _write_memory_handler(ctx, topic="t", content="")
    assert "Error" in out


@pytest.mark.asyncio
async def test_memory_tools_degrade_when_store_none(tmp_path) -> None:
    from src.tools.memory_tools import (
        _list_memory_handler, _read_memory_handler,
        _write_memory_handler, _delete_memory_handler,
    )
    ctx = ToolContext(
        plan_mode=PlanMode(save_dir=str(tmp_path / "plans")),
        registry=ToolRegistry(),
        config=AgentConfig(),
        working_dir=tmp_path,
        memory_store=None,
    )
    assert "not configured" in await _list_memory_handler(ctx)
    assert "not configured" in await _read_memory_handler(ctx, topic="x")
    assert "not configured" in await _write_memory_handler(
        ctx, topic="x", content="y",
    )
    assert "not configured" in await _delete_memory_handler(ctx, topic="x")


def test_memory_tools_registered() -> None:
    """Importing src.tools should self-register all four memory tools."""
    from src.tools.registry import get_registry
    reg = get_registry()
    for name in ("list_memory", "read_memory", "write_memory", "delete_memory"):
        assert reg.get_definition(name) is not None, f"{name} not registered"
