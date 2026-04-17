"""Bash execution tool — git-bash on Windows, bash on Unix."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import sys

from src.core.types import PermissionLevel, ToolDefinition, ToolSchema

from .registry import register

logger = logging.getLogger(__name__)

_MAX_OUTPUT_CHARS = 10_000
_TRUNCATED_TAIL = 5_000

# Dangerous patterns — logged as warnings, not blocked (Phase 2 self-check).
_DANGEROUS_PATTERNS = [
    re.compile(r"\brm\s+-rf?\s+/(?:\s|$)"),
    re.compile(r":\(\)\s*\{.*\|\s*:.*\}"),  # fork bomb
    re.compile(r"\bdd\s+if=/dev/(?:zero|random|urandom)\s+of=/dev/(?:sd|hd|nvme)"),
    re.compile(r"\bmkfs\."),
    re.compile(r">\s*/dev/(?:sd|hd|nvme)"),
]


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


def _scan_dangerous(command: str) -> str | None:
    for pat in _DANGEROUS_PATTERNS:
        if pat.search(command):
            return pat.pattern
    return None


def _truncate_output(text: str) -> str:
    if len(text) <= _MAX_OUTPUT_CHARS:
        return text
    return (
        f"[output truncated — {len(text)} chars total, showing last {_TRUNCATED_TAIL}]\n"
        + text[-_TRUNCATED_TAIL:]
    )


async def _bash_handler(command: str, timeout: int = 60) -> str:
    argv, shell_label = _build_argv(command)

    danger = _scan_dangerous(command)
    warning = f"[warning: dangerous pattern detected: {danger}]\n" if danger else ""

    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
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
        return f"Error: command timed out after {timeout}s\n" + warning

    output = stdout.decode("utf-8", errors="replace") if stdout else ""
    output = _truncate_output(output)
    exit_code = proc.returncode

    header = f"[{shell_label} exit={exit_code}]\n"
    return warning + header + output


register(ToolDefinition(
    name="bash",
    description="Execute a shell command. Uses git-bash on Windows, bash on Unix. Returns combined stdout+stderr.",
    handler=_bash_handler,
    schema=ToolSchema(
        name="bash",
        description="Execute a shell command. Output combined stdout+stderr. Timeout default 60s.",
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
