"""Tool registry — self-registration + deferred loading + ToolSearch.

Three patterns combined:

1. Self-registration: each tool module calls ``register()`` at import time.
2. Deferred loading: tools marked ``deferred=True`` are hidden from initial
   system prompt; discoverable only via ``tool_search()``.
3. ToolSearch: a tool itself (registered below). LLM calls it to activate
   tools and receive their full JSON schemas.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from src.core.types import (
    PermissionLevel,
    ToolContext,
    ToolDefinition,
    ToolSchema,
)

if TYPE_CHECKING:
    pass


class ToolRegistry:
    """Registry of all tools. Tracks registration + activation state."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}
        self._activated: set[str] = set()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, tool: ToolDefinition) -> None:
        """Register a tool. Called by tool modules at import time.

        Non-deferred tools are auto-activated (schema available immediately).
        Deferred tools require ``tool_search()`` to activate.
        """
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' already registered")
        self._tools[tool.name] = tool
        if not tool.schema.deferred:
            self._activated.add(tool.name)

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def get_initial_tools(self) -> list[dict[str, str]]:
        """Names + descriptions for all non-deferred tools.

        Used to populate the system prompt's tool listing.
        Deferred tools are NOT included (they need tool_search to surface).
        """
        return [
            {"name": t.name, "description": t.description}
            for t in self._tools.values()
            if not t.schema.deferred
        ]

    def get_active_schemas(self) -> list[ToolSchema]:
        """Full schemas for activated tools only.

        This is what gets passed to the LLM provider's ``tools`` parameter.
        """
        return [self._tools[n].schema for n in self._activated]

    def get_all_schemas(self) -> list[ToolSchema]:
        """All registered schemas (including deferred + unactivated). Debug use."""
        return [t.schema for t in self._tools.values()]

    def get_definition(self, name: str) -> ToolDefinition | None:
        return self._tools.get(name)

    def get_handler(self, name: str) -> tuple | None:
        """Returns ``(handler, needs_context)`` or None if not found."""
        t = self._tools.get(name)
        if t is None:
            return None
        return (t.handler, t.needs_context)

    def is_activated(self, name: str) -> bool:
        return name in self._activated

    def activate(self, name: str) -> ToolSchema | None:
        """Mark a registered tool as activated. Returns its schema."""
        t = self._tools.get(name)
        if t is None:
            return None
        self._activated.add(name)
        return t.schema

    # ------------------------------------------------------------------
    # ToolSearch — the discovery mechanism
    # ------------------------------------------------------------------

    def tool_search(self, query: str, limit: int = 10) -> list[ToolSchema]:
        """Fuzzy-match ``query`` against tool names + descriptions.

        Matched tools are activated (schemas become available to the LLM
        via ``get_active_schemas()``). Case-insensitive substring match.
        """
        q = query.lower().strip()
        if not q:
            return []

        matches: list[ToolSchema] = []
        for t in self._tools.values():
            if q in t.name.lower() or q in t.description.lower():
                self._activated.add(t.name)
                matches.append(t.schema)
                if len(matches) >= limit:
                    break
        return matches


# ---------------------------------------------------------------------------
# Module-level singleton + convenience API
# ---------------------------------------------------------------------------

_default_registry = ToolRegistry()


def get_registry() -> ToolRegistry:
    return _default_registry


def register(tool: ToolDefinition) -> None:
    """Register a tool to the default registry. Called at tool-module import."""
    _default_registry.register(tool)


# ---------------------------------------------------------------------------
# ToolSearch itself as a tool — registered in the default registry
# ---------------------------------------------------------------------------

async def _tool_search_handler(ctx: ToolContext, query: str) -> str:
    schemas = ctx.registry.tool_search(query)
    if not schemas:
        return f"No tools matched query: {query!r}"
    lines = [f"Matched {len(schemas)} tool(s):\n"]
    for s in schemas:
        lines.append(f"• {s.name} ({s.permission_level.value})")
        lines.append(f"  {s.description}")
        lines.append(f"  parameters: {json.dumps(s.parameters, ensure_ascii=False)}")
        lines.append("")
    return "\n".join(lines)


_TOOL_SEARCH_SCHEMA = ToolSchema(
    name="tool_search",
    description=(
        "Search and activate tools by keyword. Use this to discover tools "
        "that are not in your initial tool list. Returns matching tool "
        "schemas and makes them callable."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Keyword to match against tool names and descriptions.",
            },
        },
        "required": ["query"],
    },
    permission_level=PermissionLevel.READ_ONLY,
    timeout=5,
    deferred=False,
)

register(ToolDefinition(
    name="tool_search",
    description=_TOOL_SEARCH_SCHEMA.description,
    handler=_tool_search_handler,
    schema=_TOOL_SEARCH_SCHEMA,
    needs_context=True,
))
