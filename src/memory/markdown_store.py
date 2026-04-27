"""Markdown-based memory store (Claude Code style).

Design goals (follows Claude Code's auto-memory v2.1.59+):
    - One plain ``MEMORY.md`` **index** file per project, always loaded into
      the assembler's Zone 3 at session start. Hard cap on size so the
      assembler never sees more than ~25 KB of memory.
    - Per-topic markdown files alongside the index. The LLM decides when to
      read a topic file by seeing its name in the index.
    - Everything is human-editable with ``vim`` and ``git diff``-able. No
      binary store, no hidden magic.
    - Per-project scope via a hash of the working directory, so switching
      projects gives a fresh memory slate automatically.

Why we chose this over SQLite+FTS5 (2026-04-20 decision):
    Claude Code ships at scale with exactly this design and explicitly
    rejects embeddings and FTS. The upside is LLM-legibility (the model can
    open MEMORY.md, read the pointer table, and decide which topic file to
    open) and git-friendliness (users can commit their agent's memory).

File layout::

    ~/.yigent/memory/
    └── <sha256(cwd)[:8]>/
        ├── MEMORY.md             # index, auto-loaded every session
        ├── debugging.md          # topic files, read on demand
        ├── api-conventions.md
        └── ...

Topic file schema::

    ---
    title: Debugging notes
    created: 2026-04-20T10:15:00
    updated: 2026-04-20T14:02:00
    ---
    Free-form markdown content.

MEMORY.md schema::

    # Memory index

    - [Debugging notes](debugging.md) — how we traced the FTS5 trigger bug
    - [API conventions](api-conventions.md) — naming rules for handlers
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


# Claude Code caps MEMORY.md at 200 lines / 25 KB. Match those so our
# assembler cost is bounded regardless of how much the user accumulates.
MAX_INDEX_LINES = 200
MAX_INDEX_BYTES = 25 * 1024

INDEX_FILENAME = "MEMORY.md"
INDEX_HEADER = "# Memory index\n\n"


def _slugify(name: str) -> str:
    """Convert a free-form topic name to a filesystem-safe slug.

    Lowercase, replace non-alphanumerics with dashes, collapse runs, strip
    leading/trailing dashes. Empty inputs produce ``"untitled"`` so we
    always get a valid filename.
    """
    s = name.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = s.strip("-")
    return s or "untitled"


def _project_hash(cwd: Path) -> str:
    """Stable 8-char SHA-256 prefix of the absolute working directory."""
    resolved = str(cwd.resolve())
    return hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:8]


def default_root(cwd: Path | None = None) -> Path:
    """Resolve the default memory root: ``~/.yigent/memory/<project_hash>/``."""
    base = Path.home() / ".yigent" / "memory"
    cwd = cwd or Path.cwd()
    return base / _project_hash(cwd)


# ---------------------------------------------------------------------------
# Topic file — frontmatter + body
# ---------------------------------------------------------------------------

_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.DOTALL)


@dataclass
class Topic:
    """A topic file: metadata + markdown body."""
    slug: str
    title: str
    body: str
    created: str = ""
    updated: str = ""

    def render(self) -> str:
        """Serialise back to frontmatter + body form."""
        return (
            "---\n"
            f"title: {self.title}\n"
            f"created: {self.created}\n"
            f"updated: {self.updated}\n"
            "---\n"
            f"{self.body.rstrip()}\n"
        )

    @classmethod
    def parse(cls, slug: str, text: str) -> "Topic":
        match = _FRONTMATTER_RE.match(text)
        if not match:
            # No frontmatter — treat entire file as body, infer title from slug.
            return cls(
                slug=slug,
                title=slug.replace("-", " ").title(),
                body=text.rstrip(),
            )
        fm_block, body = match.group(1), match.group(2)
        meta: dict[str, str] = {}
        for line in fm_block.splitlines():
            key, _, value = line.partition(":")
            if key.strip():
                meta[key.strip()] = value.strip()
        return cls(
            slug=slug,
            title=meta.get("title", slug),
            created=meta.get("created", ""),
            updated=meta.get("updated", ""),
            body=body.rstrip(),
        )


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class MarkdownMemoryStore:
    """Read/write access to ``MEMORY.md`` + topic files under a project root.

    All methods are synchronous — file IO is fast and we're not expecting
    contention the way an SQLite WAL store would.
    """

    def __init__(self, root: Path | str | None = None) -> None:
        self._root = Path(root) if root is not None else default_root()

    # -- lifecycle -----------------------------------------------------------

    @property
    def root(self) -> Path:
        return self._root

    def ensure_root(self) -> None:
        self._root.mkdir(parents=True, exist_ok=True)

    # -- index ---------------------------------------------------------------

    def read_index(self) -> str:
        """Return MEMORY.md contents, capped to MAX_INDEX_{LINES,BYTES}."""
        path = self._root / INDEX_FILENAME
        if not path.exists():
            return ""
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("Failed to read %s: %s", path, exc)
            return ""
        # Cap by lines first, then by bytes (bytes cap wins if both hit).
        lines = text.splitlines()
        if len(lines) > MAX_INDEX_LINES:
            lines = lines[:MAX_INDEX_LINES]
            text = "\n".join(lines) + "\n"
        if len(text.encode("utf-8")) > MAX_INDEX_BYTES:
            # Truncate to the last newline within the byte budget.
            encoded = text.encode("utf-8")[:MAX_INDEX_BYTES]
            text = encoded.decode("utf-8", errors="ignore")
            nl = text.rfind("\n")
            if nl > 0:
                text = text[: nl + 1]
        return text

    def record_index_entry(self, slug: str, title: str, hook: str) -> None:
        """Append ``- [title](slug.md) — hook`` to MEMORY.md (dedup by slug).

        Why dedup: without it, multiple writes to the same topic leave
        stale pointer lines littering the index.
        """
        self.ensure_root()
        path = self._root / INDEX_FILENAME
        entry_line = f"- [{title}]({slug}.md) — {hook}"

        if path.exists():
            existing = path.read_text(encoding="utf-8")
        else:
            existing = INDEX_HEADER

        # Strip any prior line referencing this slug.
        marker = f"]({slug}.md)"
        kept_lines: list[str] = []
        for line in existing.splitlines():
            if marker not in line:
                kept_lines.append(line)

        # Ensure header is present and well-formed.
        if not kept_lines or not kept_lines[0].startswith("# "):
            kept_lines = ["# Memory index", ""] + [
                ln for ln in kept_lines if ln.strip()
            ]

        kept_lines.append(entry_line)
        path.write_text("\n".join(kept_lines).rstrip() + "\n", encoding="utf-8")

    # -- topics --------------------------------------------------------------

    def list_topics(self) -> list[str]:
        """Return slugs of all topic files (not the index itself)."""
        if not self._root.exists():
            return []
        return sorted(
            p.stem for p in self._root.glob("*.md") if p.name != INDEX_FILENAME
        )

    def read_topic(self, slug: str) -> Topic | None:
        path = self._root / f"{_slugify(slug)}.md"
        if not path.exists():
            return None
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("Failed to read topic %s: %s", path, exc)
            return None
        return Topic.parse(path.stem, text)

    def write_topic(
        self, name: str, body: str, title: str | None = None,
    ) -> Topic:
        """Create or update a topic file. Returns the resulting Topic.

        - ``name`` is slugified; existing files with the same slug are
          **replaced** (preserving original ``created`` timestamp).
        - ``title`` defaults to ``name``.
        """
        self.ensure_root()
        slug = _slugify(name)
        path = self._root / f"{slug}.md"
        now = datetime.now().isoformat(timespec="seconds")

        existing = self.read_topic(slug) if path.exists() else None
        created = existing.created if (existing and existing.created) else now

        topic = Topic(
            slug=slug,
            title=(title or name).strip() or slug,
            body=body.rstrip() + "\n",
            created=created,
            updated=now,
        )
        path.write_text(topic.render(), encoding="utf-8")
        return topic

    def delete_topic(self, name: str) -> bool:
        slug = _slugify(name)
        path = self._root / f"{slug}.md"
        if not path.exists():
            return False
        path.unlink()
        # Also strip the index entry.
        index_path = self._root / INDEX_FILENAME
        if index_path.exists():
            marker = f"]({slug}.md)"
            kept = [
                ln for ln in index_path.read_text(encoding="utf-8").splitlines()
                if marker not in ln
            ]
            index_path.write_text("\n".join(kept).rstrip() + "\n", encoding="utf-8")
        return True

    # -- async wrappers ------------------------------------------------------
    # Audit B5 / Top10 #10: ContextAssembler reads the memory index on every
    # turn; tools mutate topic files inside an async event loop. Wrapping
    # disk I/O in asyncio.to_thread keeps a slow filesystem (network mount,
    # antivirus, contention) from stalling the loop. Sync methods above are
    # preserved for back-compat and tests that don't need async.

    async def aread_index(self) -> str:
        return await asyncio.to_thread(self.read_index)

    async def aread_topic(self, slug: str) -> Topic | None:
        return await asyncio.to_thread(self.read_topic, slug)

    async def alist_topics(self) -> list[str]:
        return await asyncio.to_thread(self.list_topics)

    async def awrite_topic(
        self, name: str, body: str, title: str | None = None,
    ) -> Topic:
        return await asyncio.to_thread(self.write_topic, name, body, title)

    async def adelete_topic(self, name: str) -> bool:
        return await asyncio.to_thread(self.delete_topic, name)

    async def arecord_index_entry(self, slug: str, title: str, hook: str) -> None:
        await asyncio.to_thread(self.record_index_entry, slug, title, hook)
