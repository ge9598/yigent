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


# ---------------------------------------------------------------------------
# file_ops.py: read/write/list_dir/search_files must honor ctx.working_dir
# for relative paths, or relative writes end up in the CLI's cwd (repo root).
# Observed in Phase 3 benchmark runs: agent wrote "test_workspace/" and
# "buggy.py" that appeared at the repo root instead of in the per-task
# workspace. Regression covers all four file-ops handlers.
# ---------------------------------------------------------------------------


async def test_write_file_relative_path_lands_in_working_dir(tmp_path: Path):
    from src.tools.file_ops import _write_file_handler

    ctx = _ctx(tmp_path)
    result = await _write_file_handler(ctx, path="hello.txt", content="hi")

    # File must be in tmp_path, NOT in the test process's cwd
    assert (tmp_path / "hello.txt").exists()
    assert (tmp_path / "hello.txt").read_text(encoding="utf-8") == "hi"
    assert "Wrote" in result


async def test_write_file_absolute_path_inside_wd_succeeds(tmp_path: Path):
    """Absolute paths that resolve under working_dir are still honored."""
    from src.tools.file_ops import _write_file_handler

    ws = tmp_path / "a"
    ws.mkdir()
    ctx = _ctx(ws)
    # Absolute target that lives inside ws
    target = ws / "sub" / "out.txt"
    result = await _write_file_handler(ctx, path=str(target), content="x")

    assert target.exists()
    assert "Wrote" in result


async def test_write_file_absolute_path_outside_wd_blocked(tmp_path: Path):
    """Absolute paths that escape working_dir are now refused (audit Top10 #7)."""
    from src.tools.file_ops import _write_file_handler

    ws = tmp_path / "a"
    ws.mkdir()
    ctx = _ctx(ws)  # working_dir is a subdir
    target = tmp_path / "b" / "out.txt"
    result = await _write_file_handler(ctx, path=str(target), content="x")

    assert not target.exists()
    assert list(ws.iterdir()) == []
    assert "refusing to write outside working directory" in result


async def test_read_file_relative_path_resolves_under_working_dir(tmp_path: Path):
    from src.tools.file_ops import _read_file_handler

    (tmp_path / "note.md").write_text("# header\nbody\n", encoding="utf-8")
    ctx = _ctx(tmp_path)

    result = await _read_file_handler(ctx, path="note.md")
    assert "header" in result and "body" in result


async def test_list_dir_relative_resolves_under_working_dir(tmp_path: Path):
    from src.tools.file_ops import _list_dir_handler

    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "marker.txt").write_text("", encoding="utf-8")
    ctx = _ctx(tmp_path)

    result = await _list_dir_handler(ctx, path="sub")
    assert "marker.txt" in result


async def test_search_files_relative_resolves_under_working_dir(tmp_path: Path):
    from src.tools.file_ops import _search_files_handler

    (tmp_path / "app.py").write_text("TOKEN = 'secret'\n", encoding="utf-8")
    ctx = _ctx(tmp_path)

    result = await _search_files_handler(ctx, pattern="TOKEN", path=".")
    assert "TOKEN" in result
    assert "app.py" in result
