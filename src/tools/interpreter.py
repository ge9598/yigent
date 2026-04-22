"""Python REPL tool — stateless per-call subprocess."""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

from src.core.types import PermissionLevel, ToolContext, ToolDefinition, ToolSchema

from .registry import register

_MAX_OUTPUT_CHARS = 10_000
_TRUNCATED_TAIL = 5_000


def _truncate(text: str) -> str:
    if len(text) <= _MAX_OUTPUT_CHARS:
        return text
    return (
        f"[output truncated — {len(text)} chars total, showing last {_TRUNCATED_TAIL}]\n"
        + text[-_TRUNCATED_TAIL:]
    )


async def _python_repl_handler(
    ctx: ToolContext, code: str, timeout: int = 30,
) -> str:
    """Execute Python code in a fresh subprocess. Stateless per call."""
    # Validate cwd BEFORE creating the temp file so we don't leak it on
    # the early-return path.
    cwd_str: str | None = None
    if ctx.working_dir is not None:
        wd = Path(ctx.working_dir)
        if wd.is_dir():
            cwd_str = str(wd)
        else:
            return (
                f"Error: working_dir {wd} does not exist — "
                "python_repl will not run in a stale cwd"
            )

    # Write to temp .py (avoids quoting issues with -c for multi-line code)
    fd, tmp_path = tempfile.mkstemp(suffix=".py", text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(code)

        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, tmp_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=cwd_str,
            )
        except FileNotFoundError as e:
            return f"Error: python executable not found: {e}"

        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            await proc.wait()
            return f"Error: python code timed out after {timeout}s"

        output = stdout.decode("utf-8", errors="replace") if stdout else ""
        output = _truncate(output)
        return f"[python exit={proc.returncode}]\n{output}"

    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


register(ToolDefinition(
    name="python_repl",
    description="Execute Python code in a fresh subprocess. Stateless — each call is isolated.",
    handler=_python_repl_handler,
    needs_context=True,
    schema=ToolSchema(
        name="python_repl",
        description=(
            "Run Python code in a fresh subprocess, with cwd set to the "
            "agent's current working directory (relative paths like "
            "'data.csv' resolve there). Returns combined stdout+stderr."
        ),
        parameters={
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute."},
                "timeout": {"type": "integer", "description": "Max seconds.", "default": 30},
            },
            "required": ["code"],
        },
        permission_level=PermissionLevel.EXECUTE,
        timeout=30,
    ),
))
