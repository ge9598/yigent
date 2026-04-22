"""Regression tests for bash/python_repl honoring ctx.working_dir.

Previously the bash tool ran subprocess without a cwd argument, which on
Windows git-bash means `$PWD` is inherited from the parent shell (the
repo root). Agents running in an eval workspace would instead operate
on the repo root — at one point a file_management/easy benchmark task
ran `mv *.md md/` in the repo root and moved CLAUDE.md + README.md
into a stray directory. These tests ensure both shell tools chdir into
ctx.working_dir before exec.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.core.config import load_config
from src.core.types import ToolContext
from src.tools.coding import _bash_handler
from src.tools.interpreter import _python_repl_handler


def _ctx(working_dir: Path) -> ToolContext:
    return ToolContext(
        plan_mode=MagicMock(is_active=False),
        registry=MagicMock(),
        config=load_config(),
        working_dir=working_dir,
    )


async def test_bash_runs_in_ctx_working_dir(tmp_path: Path):
    (tmp_path / "marker.txt").write_text("hello", encoding="utf-8")
    ctx = _ctx(tmp_path)

    # `ls` will list the marker only if bash actually cd'd into tmp_path
    out = await _bash_handler(ctx, command="ls")
    assert "marker.txt" in out
    assert "exit=0" in out


async def test_bash_refuses_stale_cwd(tmp_path: Path):
    missing = tmp_path / "nonexistent"
    ctx = _ctx(missing)
    out = await _bash_handler(ctx, command="echo hi")
    assert "does not exist" in out
    # Must NOT fall back to some parent cwd and run the command anyway
    assert "hi" not in out.splitlines()[-1] if out.strip() else True


async def test_python_repl_runs_in_ctx_working_dir(tmp_path: Path):
    (tmp_path / "hello.txt").write_text("world", encoding="utf-8")
    ctx = _ctx(tmp_path)
    code = "import os; print('EXISTS' if os.path.exists('hello.txt') else 'MISSING')"
    out = await _python_repl_handler(ctx, code=code)
    assert "EXISTS" in out


async def test_python_repl_refuses_stale_cwd(tmp_path: Path):
    ctx = _ctx(tmp_path / "nonexistent")
    out = await _python_repl_handler(ctx, code="print(1)")
    assert "does not exist" in out


async def test_bash_cwd_does_not_leak_across_calls(tmp_path: Path):
    """Each call should independently honor its own ctx — no residual state."""
    ws_a = tmp_path / "a"
    ws_a.mkdir()
    (ws_a / "only_in_a.txt").write_text("", encoding="utf-8")
    ws_b = tmp_path / "b"
    ws_b.mkdir()
    (ws_b / "only_in_b.txt").write_text("", encoding="utf-8")

    out_a = await _bash_handler(_ctx(ws_a), command="ls")
    out_b = await _bash_handler(_ctx(ws_b), command="ls")
    assert "only_in_a.txt" in out_a and "only_in_b.txt" not in out_a
    assert "only_in_b.txt" in out_b and "only_in_a.txt" not in out_b
