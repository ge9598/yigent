"""Tests for write_file containment under working_dir (audit Top10 #7 / A1)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

import src.tools.file_ops as file_ops  # noqa: F401  — registers tools at import
from src.core.config import load_config
from src.core.plan_mode import PlanMode
from src.core.types import ToolContext
from src.tools.registry import ToolRegistry


@pytest.fixture
def ctx_factory(tmp_path: Path):
    """Build a ToolContext rooted at tmp_path with a real working_dir."""
    def _factory(working_dir: Path | None = None) -> ToolContext:
        return ToolContext(
            plan_mode=PlanMode(),
            registry=ToolRegistry(),
            config=load_config("configs/default.yaml"),
            working_dir=working_dir if working_dir is not None else tmp_path,
            session_id="t",
        )
    return _factory


# Direct handler references avoid the registry entirely (the file_ops module
# registers into a global registry; we test the handler in isolation).
WRITE_HANDLER = file_ops._write_file_handler


@pytest.mark.asyncio
async def test_write_inside_working_dir_succeeds(ctx_factory, tmp_path):
    ctx = ctx_factory()
    out = await WRITE_HANDLER(ctx, path="hello.txt", content="hi")
    assert "Wrote" in out
    assert (tmp_path / "hello.txt").read_text() == "hi"


@pytest.mark.asyncio
async def test_write_dotdot_escape_blocked(ctx_factory, tmp_path):
    ctx = ctx_factory()
    out = await WRITE_HANDLER(ctx, path="../escaped.txt", content="hi")
    assert "refusing to write outside working directory" in out
    assert not (tmp_path.parent / "escaped.txt").exists()


@pytest.mark.asyncio
async def test_write_absolute_outside_blocked(ctx_factory, tmp_path):
    ctx = ctx_factory()
    target = tmp_path.parent / "outside.txt"
    out = await WRITE_HANDLER(ctx, path=str(target), content="hi")
    assert "refusing to write outside working directory" in out
    assert not target.exists()


@pytest.mark.asyncio
@pytest.mark.skipif(sys.platform == "win32", reason="symlink creation may need privileges on Windows")
async def test_write_via_symlink_to_outside_blocked(ctx_factory, tmp_path):
    ctx = ctx_factory()
    outside_dir = tmp_path.parent / "outside_sym_target"
    outside_dir.mkdir(exist_ok=True)
    sym = tmp_path / "trap"
    sym.symlink_to(outside_dir)
    out = await WRITE_HANDLER(ctx, path="trap/inside.txt", content="hi")
    assert "refusing to write outside working directory" in out
    assert not (outside_dir / "inside.txt").exists()


@pytest.mark.asyncio
async def test_write_no_working_dir_unrestricted(tmp_path, ctx_factory):
    # Backwards-compat path: when ctx.working_dir is None, no containment check
    ctx = ctx_factory(working_dir=None)
    # Must use absolute target so the test is hermetic regardless of os.getcwd()
    target = tmp_path / "compat.txt"
    out = await WRITE_HANDLER(ctx, path=str(target), content="hi")
    assert "Wrote" in out
    assert target.read_text() == "hi"
