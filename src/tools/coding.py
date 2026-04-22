"""Bash execution tool — git-bash on Windows, bash on Unix."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import sys
from pathlib import Path

from src.core.types import PermissionLevel, ToolContext, ToolDefinition, ToolSchema

from .registry import register

logger = logging.getLogger(__name__)

_MAX_OUTPUT_CHARS = 10_000
_TRUNCATED_TAIL = 5_000


def _find_bash() -> str | None:
    """Locate bash executable. Returns None if not available."""
    if sys.platform == "win32":
        # 1. In PATH (git-bash adds itself)
        bash = shutil.which("bash")
        if bash and "System32" not in bash:  # skip Windows WSL bash.exe (unreliable)
            return bash
        # 2. Common Git install paths
        for candidate in (
            r"C:\Program Files\Git\bin\bash.exe",
            r"C:\Program Files (x86)\Git\bin\bash.exe",
        ):
            if os.path.exists(candidate):
                return candidate
        return None
    # Unix: bash should be on PATH
    return shutil.which("bash")


_BASH_PATH = _find_bash()
if _BASH_PATH is None and sys.platform == "win32":
    logger.warning("bash not found on Windows; bash tool will fall back to cmd.exe")


def _build_argv(command: str) -> tuple[list[str], str]:
    """Returns (argv, shell_label)."""
    if _BASH_PATH:
        return ([_BASH_PATH, "-c", command], "bash")
    # Fallback: cmd on Windows, sh on Unix
    if sys.platform == "win32":
        return (["cmd", "/c", command], "cmd")
    return (["sh", "-c", command], "sh")


def _truncate_output(text: str) -> str:
    if len(text) <= _MAX_OUTPUT_CHARS:
        return text
    return (
        f"[output truncated — {len(text)} chars total, showing last {_TRUNCATED_TAIL}]\n"
        + text[-_TRUNCATED_TAIL:]
    )


async def _bash_handler(
    ctx: ToolContext, command: str, timeout: int = 60,
) -> str:
    argv, shell_label = _build_argv(command)

    # Honor ctx.working_dir so the command runs where the agent thinks it
    # is. Without this on Windows, git-bash's $PWD inherits the parent
    # shell's cwd (typically the repo root) — the agent ends up running
    # mv/rm in the wrong directory and corrupts the repo. Pre-flight
    # check that the dir actually exists so we fail fast with a clear
    # error instead of silently falling back to the parent cwd.
    cwd_str: str | None = None
    if ctx.working_dir is not None:
        wd = Path(ctx.working_dir)
        if wd.is_dir():
            cwd_str = str(wd)
        else:
            return (
                f"Error: working_dir {wd} does not exist — "
                "bash will not run in a stale cwd"
            )

    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=cwd_str,
        )
    except FileNotFoundError as e:
        return f"Error: shell executable not found: {e}"

    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        await proc.wait()
        return f"Error: command timed out after {timeout}s"

    output = stdout.decode("utf-8", errors="replace") if stdout else ""
    output = _truncate_output(output)
    exit_code = proc.returncode

    header = f"[{shell_label} exit={exit_code}]\n"
    return header + output


register(ToolDefinition(
    name="bash",
    description="Execute a shell command. Uses git-bash on Windows, bash on Unix. Returns combined stdout+stderr.",
    handler=_bash_handler,
    needs_context=True,
    schema=ToolSchema(
        name="bash",
        description=(
            "Execute a shell command in the agent's current working "
            "directory (no `cd` needed — cwd is already set for you). "
            "Output combined stdout+stderr. Timeout default 60s."
        ),
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to run."},
                "timeout": {"type": "integer", "description": "Max seconds.", "default": 60},
            },
            "required": ["command"],
        },
        permission_level=PermissionLevel.EXECUTE,
        timeout=60,
    ),
))
