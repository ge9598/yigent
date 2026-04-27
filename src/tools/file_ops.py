"""File operation tools: read_file, write_file, list_dir, search_files.

All four handlers take ``ctx`` and resolve relative paths against
``ctx.working_dir`` (not the Python process's cwd). Absolute paths are
left untouched. Without this, tools like ``write_file`` would silently
write relative paths into whatever dir the CLI was launched from — a
bug that previously caused benchmark runs to create stray
``test_workspace/`` and ``buggy.py`` in the repo root.
"""

from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path

from src.core.types import PermissionLevel, ToolContext, ToolDefinition, ToolSchema

from .registry import register


def _resolve_under(ctx: ToolContext, raw_path: str) -> Path:
    """Resolve ``raw_path`` relative to ``ctx.working_dir``.

    Absolute paths pass through untouched. Relative paths are joined
    with the working dir. If no working dir is configured, falls back
    to Python's cwd (Phase 1 behavior, preserved for tests that don't
    bother setting working_dir).
    """
    p = Path(raw_path)
    if p.is_absolute():
        return p
    wd = ctx.working_dir if ctx.working_dir is not None else Path.cwd()
    return Path(wd) / p


def _enforce_within_working_dir(ctx: ToolContext, resolved: Path, raw_path: str) -> str | None:
    """Block writes that escape ctx.working_dir via symlink or ``..``.

    Returns an error string when the path escapes containment, ``None`` when safe.
    Only enforced for WRITE-level tools (write_file). Reads are deliberately
    permissive — the audit (Top10 #7) flagged write-side escape as the practical
    risk; read-side restrictions break legitimate "read /etc/hosts" workflows.

    No-op when ``ctx.working_dir`` is not set (Phase 1 / test compatibility).
    """
    if ctx.working_dir is None:
        return None
    try:
        wd_real = Path(ctx.working_dir).resolve(strict=False)
    except OSError:
        return None
    try:
        # strict=False so we can validate paths to files that don't exist yet
        target_real = resolved.resolve(strict=False)
    except OSError as e:
        return f"Error: cannot resolve {raw_path}: {type(e).__name__}"
    try:
        target_real.relative_to(wd_real)
    except ValueError:
        return (
            f"Error: refusing to write outside working directory. "
            f"Target {raw_path!r} resolves to {target_real}, "
            f"which is not under {wd_real}."
        )
    return None

# Directories to skip in traversal (both list_dir and search_files).
_SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", "dist", "build",
    ".idea", ".vscode",
}

_BINARY_CHECK_BYTES = 1024  # first N bytes to sample for binary detection


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_binary_file(path: Path) -> bool:
    """Heuristic: file is binary if first 1KB contains null byte or fails UTF-8."""
    try:
        with open(path, "rb") as f:
            chunk = f.read(_BINARY_CHECK_BYTES)
    except OSError:
        return False
    if b"\x00" in chunk:
        return True
    try:
        chunk.decode("utf-8")
    except UnicodeDecodeError:
        return True
    return False


# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------

async def _read_file_handler(
    ctx: ToolContext, path: str, offset: int = 0, limit: int = 2000,
) -> str:
    def _sync() -> str:
        p = _resolve_under(ctx, path)
        if not p.exists():
            return f"Error: file not found: {path}"
        if not p.is_file():
            return f"Error: not a regular file: {path}"
        if _is_binary_file(p):
            return f"Error: cannot read binary file: {path}"
        try:
            with open(p, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
        except OSError as e:
            return f"Error reading {path}: {e}"
        total = len(lines)
        end = min(total, offset + limit)
        selected = lines[offset:end]
        numbered = [f"{i + offset + 1:>6}\t{line.rstrip(chr(10))}" for i, line in enumerate(selected)]
        header = f"# {path} (lines {offset + 1}-{end} of {total})"
        return header + "\n" + "\n".join(numbered)

    return await asyncio.to_thread(_sync)


register(ToolDefinition(
    name="read_file",
    description="Read a text file and return its contents with line numbers. Rejects binary files.",
    handler=_read_file_handler,
    needs_context=True,
    schema=ToolSchema(
        name="read_file",
        description="Read a text file and return its contents with line numbers.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path (absolute or relative)."},
                "offset": {"type": "integer", "description": "Starting line (0-indexed).", "default": 0},
                "limit": {"type": "integer", "description": "Max lines to return.", "default": 2000},
            },
            "required": ["path"],
        },
        permission_level=PermissionLevel.READ_ONLY,
        timeout=10,
    ),
))


# ---------------------------------------------------------------------------
# write_file
# ---------------------------------------------------------------------------

async def _write_file_handler(
    ctx: ToolContext, path: str, content: str,
) -> str:
    def _sync() -> str:
        p = _resolve_under(ctx, path)
        # Containment check: refuse to follow symlinks or `..` out of working_dir.
        # No-op when working_dir is not set (legacy / tests).
        escape_error = _enforce_within_working_dir(ctx, p, path)
        if escape_error is not None:
            return escape_error
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(p.suffix + ".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(content)
            os.replace(tmp, p)
        except OSError as e:
            return f"Error writing {path}: {e}"
        return f"Wrote {len(content.encode('utf-8'))} bytes to {path}"

    return await asyncio.to_thread(_sync)


register(ToolDefinition(
    name="write_file",
    description="Write content to a file, creating parent directories as needed. Overwrites existing files.",
    handler=_write_file_handler,
    needs_context=True,
    schema=ToolSchema(
        name="write_file",
        description="Write content to a file. Creates parent dirs. Atomic via temp-file + rename.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Target file path."},
                "content": {"type": "string", "description": "Full file content to write."},
            },
            "required": ["path", "content"],
        },
        permission_level=PermissionLevel.WRITE,
        timeout=15,
    ),
))


# ---------------------------------------------------------------------------
# list_dir
# ---------------------------------------------------------------------------

async def _list_dir_handler(
    ctx: ToolContext, path: str = ".", depth: int = 2,
) -> str:
    def _sync() -> str:
        root = _resolve_under(ctx, path)
        if not root.exists():
            return f"Error: path not found: {path}"
        if not root.is_dir():
            return f"Error: not a directory: {path}"

        lines: list[str] = [f"{root}/"]
        root_abs = root.resolve()

        def walk(current: Path, level: int) -> None:
            if level > depth:
                return
            try:
                entries = sorted(current.iterdir(), key=lambda p: (p.is_file(), p.name))
            except PermissionError:
                return
            for entry in entries:
                if entry.name in _SKIP_DIRS:
                    continue
                rel_depth = len(entry.resolve().relative_to(root_abs).parts)
                indent = "  " * rel_depth
                if entry.is_dir():
                    lines.append(f"{indent}{entry.name}/")
                    walk(entry, level + 1)
                else:
                    lines.append(f"{indent}{entry.name}")

        walk(root, 1)
        return "\n".join(lines)

    return await asyncio.to_thread(_sync)


register(ToolDefinition(
    name="list_dir",
    description="List directory contents recursively as a tree, up to a given depth.",
    handler=_list_dir_handler,
    needs_context=True,
    schema=ToolSchema(
        name="list_dir",
        description="Tree listing of a directory. Skips .git, __pycache__, node_modules, .venv, etc.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory path.", "default": "."},
                "depth": {"type": "integer", "description": "Max recursion depth.", "default": 2},
            },
        },
        permission_level=PermissionLevel.READ_ONLY,
        timeout=10,
    ),
))


# ---------------------------------------------------------------------------
# search_files
# ---------------------------------------------------------------------------

_MAX_MATCHES = 50


async def _search_files_handler(
    ctx: ToolContext,
    pattern: str,
    path: str = ".",
    glob: str = "**/*",
) -> str:
    def _sync() -> str:
        try:
            regex = re.compile(pattern)
        except re.error as e:
            return f"Error: invalid regex '{pattern}': {e}"

        root = _resolve_under(ctx, path)
        if not root.exists():
            return f"Error: path not found: {path}"

        matches: list[str] = []
        count = 0
        for file in root.rglob(glob):
            if count >= _MAX_MATCHES:
                break
            if not file.is_file():
                continue
            if any(skip in file.parts for skip in _SKIP_DIRS):
                continue
            if _is_binary_file(file):
                continue
            try:
                with open(file, "r", encoding="utf-8", errors="replace") as f:
                    for i, line in enumerate(f, 1):
                        if count >= _MAX_MATCHES:
                            break
                        if regex.search(line):
                            matches.append(f"{file}:{i}:{line.rstrip()}")
                            count += 1
            except OSError:
                continue

        if not matches:
            return f"No matches for pattern '{pattern}' in {path}"
        header = f"Found {len(matches)} match(es)"
        if count >= _MAX_MATCHES:
            header += f" (truncated at {_MAX_MATCHES})"
        return header + ":\n" + "\n".join(matches)

    return await asyncio.to_thread(_sync)


register(ToolDefinition(
    name="search_files",
    description="Search for a regex pattern across files in a directory tree.",
    handler=_search_files_handler,
    needs_context=True,
    schema=ToolSchema(
        name="search_files",
        description="Regex search across files. Returns 'file:line:content' matches (max 50).",
        parameters={
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regular expression to search for."},
                "path": {"type": "string", "description": "Root directory.", "default": "."},
                "glob": {"type": "string", "description": "Glob filter for files.", "default": "**/*"},
            },
            "required": ["pattern"],
        },
        permission_level=PermissionLevel.READ_ONLY,
        timeout=30,
    ),
))
