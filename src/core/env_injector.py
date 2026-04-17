"""Task-type-aware environment context injection."""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

_MAX_CHARS = 2000

_TASK_KEYWORDS: dict[str, list[str]] = {
    "coding": ["code", "bug", "fix", "implement", "function", "class", "refactor",
               "test", "debug", "compile", "build", "import", "error", "traceback",
               "git", "commit", "branch", "merge", "lint", "type"],
    "data_analysis": ["csv", "json", "data", "analyze", "column", "dataframe",
                      "statistics", "mean", "median", "plot", "chart", "pandas"],
    "file_ops": ["file", "directory", "folder", "organize", "move", "copy",
                 "rename", "delete", "search", "find", "list", "tree"],
    "research": ["search", "look up", "find out", "summarize", "compare",
                 "documentation", "article", "paper"],
}

_SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv",
              ".pytest_cache", "dist", "build"}


class EnvironmentInjector:
    """Injects task-relevant context before each LLM call."""

    def __init__(self, working_dir: Path | None = None) -> None:
        self._cwd = working_dir or Path.cwd()

    def detect_task_type(self, text: str) -> str:
        """Heuristic task-type detection from conversation text."""
        text_lower = text.lower()
        scores: dict[str, int] = {t: 0 for t in _TASK_KEYWORDS}
        for task_type, keywords in _TASK_KEYWORDS.items():
            for kw in keywords:
                if kw in text_lower:
                    scores[task_type] += 1
        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else "coding"

    async def get_context(self, task_type: str) -> str:
        """Return task-type-aware environment context string, max ~2000 chars."""
        dispatch = {
            "coding": self._coding_context,
            "data_analysis": self._data_context,
            "file_ops": self._file_ops_context,
            "research": self._research_context,
        }
        fn = dispatch.get(task_type, self._coding_context)
        try:
            ctx = await fn()
        except Exception:
            ctx = ""
        return ctx[:_MAX_CHARS]

    async def _coding_context(self) -> str:
        """Git branch + recent commits + working tree status."""
        parts: list[str] = []
        for cmd, label in [
            (["git", "branch", "--show-current"], "Branch"),
            (["git", "log", "--oneline", "-5"], "Recent commits"),
            (["git", "status", "--short"], "Working tree"),
        ]:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL, cwd=str(self._cwd),
                )
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
                output = stdout.decode("utf-8", errors="replace").strip()
                if output:
                    parts.append(f"[{label}]\n{output}")
            except (FileNotFoundError, asyncio.TimeoutError):
                pass
        return "\n\n".join(parts) if parts else ""

    async def _data_context(self) -> str:
        """Scan for CSV/JSON in working dir, show first line (headers)."""
        lines: list[str] = ["[Data files in working directory]"]
        for ext in ("*.csv", "*.json", "*.jsonl"):
            for f in sorted(self._cwd.glob(ext))[:5]:
                try:
                    with open(f, "r", encoding="utf-8") as fh:
                        head = fh.readline().strip()
                    lines.append(f"  {f.name}: {head[:200]}")
                except OSError:
                    pass
        return "\n".join(lines) if len(lines) > 1 else ""

    async def _file_ops_context(self) -> str:
        """2-level directory tree."""
        lines: list[str] = [f"[Directory: {self._cwd}]"]
        for entry in sorted(self._cwd.iterdir()):
            if entry.name in _SKIP_DIRS or entry.name.startswith("."):
                continue
            if entry.is_dir():
                lines.append(f"  {entry.name}/")
                try:
                    for child in sorted(entry.iterdir())[:10]:
                        if child.name not in _SKIP_DIRS:
                            suffix = "/" if child.is_dir() else ""
                            lines.append(f"    {child.name}{suffix}")
                except PermissionError:
                    pass
            else:
                lines.append(f"  {entry.name}")
        return "\n".join(lines)

    async def _research_context(self) -> str:
        """Phase 1: empty. Phase 2 will add search history."""
        return ""
