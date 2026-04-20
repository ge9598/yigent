"""Memory tools: list_memory, read_memory, write_memory, delete_memory.

These expose the MarkdownMemoryStore to the agent. The model is instructed
in the system prompt to call ``write_memory`` when it learns something
reusable across sessions, and ``read_memory`` when the MEMORY.md index (which
it sees automatically at session start) points at a topic relevant to the
current task.

Design note: ``write_memory`` is classified WRITE-level so the permission
gate asks the user first by default. In YOLO mode it auto-allows, which
matches Claude Code's auto-memory behaviour. Users who want mandatory
confirmation can also add a ``pre_tool_use`` hook returning ``ask``.
"""

from __future__ import annotations

from src.core.types import (
    PermissionLevel, ToolContext, ToolDefinition, ToolSchema,
)

from .registry import register


# ---------------------------------------------------------------------------
# list_memory
# ---------------------------------------------------------------------------

async def _list_memory_handler(ctx: ToolContext) -> str:
    if ctx.memory_store is None:
        return "Error: memory store is not configured."
    slugs = ctx.memory_store.list_topics()
    if not slugs:
        return "(no memory topics)"
    return "\n".join(slugs)


register(ToolDefinition(
    name="list_memory",
    description=(
        "List the slugs of all memory topics saved for this project. "
        "Memory is stored as markdown files under ~/.yigent/memory/."
    ),
    handler=_list_memory_handler,
    schema=ToolSchema(
        name="list_memory",
        description="List all saved memory topic slugs for this project.",
        parameters={"type": "object", "properties": {}, "required": []},
        permission_level=PermissionLevel.READ_ONLY,
        timeout=5,
    ),
    needs_context=True,
))


# ---------------------------------------------------------------------------
# read_memory
# ---------------------------------------------------------------------------

async def _read_memory_handler(ctx: ToolContext, topic: str) -> str:
    if ctx.memory_store is None:
        return "Error: memory store is not configured."
    t = ctx.memory_store.read_topic(topic)
    if t is None:
        return f"Error: no memory topic named '{topic}'. Use list_memory to see available topics."
    return (
        f"# {t.title}\n"
        f"(updated: {t.updated})\n\n"
        f"{t.body}"
    )


register(ToolDefinition(
    name="read_memory",
    description=(
        "Read a saved memory topic by name. Use this when the MEMORY.md "
        "index points at a topic relevant to the current task."
    ),
    handler=_read_memory_handler,
    schema=ToolSchema(
        name="read_memory",
        description=(
            "Read a memory topic. Pass the topic slug or title. "
            "Returns the full markdown body of the topic file."
        ),
        parameters={
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "Slug or free-form name of the topic to read.",
                },
            },
            "required": ["topic"],
        },
        permission_level=PermissionLevel.READ_ONLY,
        timeout=5,
    ),
    needs_context=True,
))


# ---------------------------------------------------------------------------
# write_memory
# ---------------------------------------------------------------------------

async def _write_memory_handler(
    ctx: ToolContext,
    topic: str,
    content: str,
    hook: str = "",
    title: str = "",
) -> str:
    if ctx.memory_store is None:
        return "Error: memory store is not configured."
    if not topic or not content:
        return "Error: both 'topic' and 'content' are required."
    t = ctx.memory_store.write_topic(
        topic, content, title=(title or topic),
    )
    ctx.memory_store.record_index_entry(
        t.slug, t.title, hook or "(no hook)",
    )
    return f"Saved memory topic '{t.slug}' ({len(t.body)} chars)."


register(ToolDefinition(
    name="write_memory",
    description=(
        "Save a reusable fact or pattern to long-term memory. Call this when "
        "you discover something that would help a future session — user "
        "preferences, project conventions, gotchas, architectural decisions."
    ),
    handler=_write_memory_handler,
    schema=ToolSchema(
        name="write_memory",
        description=(
            "Save a topic to long-term memory. Existing topics with the same "
            "name are overwritten (created timestamp is preserved). The index "
            "entry 'hook' is a one-line description shown in MEMORY.md."
        ),
        parameters={
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "Short name (will be slugified for the filename).",
                },
                "content": {
                    "type": "string",
                    "description": "Full markdown body to save as this topic's file.",
                },
                "hook": {
                    "type": "string",
                    "description": "One-line description for the MEMORY.md index entry.",
                },
                "title": {
                    "type": "string",
                    "description": "Human-readable title. Defaults to 'topic'.",
                },
            },
            "required": ["topic", "content"],
        },
        permission_level=PermissionLevel.WRITE,
        timeout=5,
    ),
    needs_context=True,
))


# ---------------------------------------------------------------------------
# delete_memory
# ---------------------------------------------------------------------------

async def _delete_memory_handler(ctx: ToolContext, topic: str) -> str:
    if ctx.memory_store is None:
        return "Error: memory store is not configured."
    ok = ctx.memory_store.delete_topic(topic)
    return (
        f"Deleted memory topic '{topic}'." if ok
        else f"No memory topic named '{topic}' found."
    )


register(ToolDefinition(
    name="delete_memory",
    description="Delete a memory topic by name. Also removes its MEMORY.md entry.",
    handler=_delete_memory_handler,
    schema=ToolSchema(
        name="delete_memory",
        description="Delete a memory topic.",
        parameters={
            "type": "object",
            "properties": {
                "topic": {"type": "string",
                          "description": "Slug or name to delete."},
            },
            "required": ["topic"],
        },
        permission_level=PermissionLevel.WRITE,
        timeout=5,
    ),
    needs_context=True,
))
