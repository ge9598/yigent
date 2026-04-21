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

    def all(self) -> list[ToolDefinition]:
        """All registered tool definitions. Used by PlanMode (Unit 10) to
        compute the dynamic READ_ONLY allowlist at enter() time."""
        return list(self._tools.values())

    # Unit 10 — capability → tool-name mapping. Hardcoded for built-in tools;
    # MCP / skill tools fall outside this map (they go through ToolSearch
    # instead).
    _CAPABILITY_GROUPS: dict[str, list[str]] = {
        "search": ["web_search", "search_files"],
        "coding": ["read_file", "write_file", "bash"],
        "interpreter": ["python_repl", "bash"],
        "file_ops": ["read_file", "write_file", "list_dir", "search_files"],
    }

    def activate_capability_group(self, capability: str) -> list[str]:
        """Pre-activate all tools belonging to a capability group.

        Returns the list of tools that were newly activated (already-active
        tools are silently skipped). Unknown capability returns [].

        Used by the agent loop after CapabilityRouter classifies the user's
        intent — pre-loading the obvious tools spares a ToolSearch round-trip.
        """
        names = self._CAPABILITY_GROUPS.get(capability, [])
        activated: list[str] = []
        for name in names:
            if name in self._tools and name not in self._activated:
                self._activated.add(name)
                activated.append(name)
        return activated

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

    def tool_search(self, query: str, limit: int = 5) -> list[ToolSchema]:
        """Search tools by keyword with weighted scoring or exact ``select:`` syntax.

        Two modes:

        **select: syntax** — ``"select:write_file,enter_plan_mode"``
            Exact name lookup for the comma-separated list. Unknown names are
            silently skipped. Activates all matched tools.

        **Weighted scoring** — any other query string
            Scores each tool and returns the top ``limit`` (default 5):

            * +10  exact name match (case-insensitive)
            * +5   query is a substring of the tool name
            * +2   query is a substring of the tool description

            Results are sorted by score descending. Matched tools are activated.

        Matched tools are activated (schemas become available to the LLM
        via ``get_active_schemas()``).
        """
        q = query.strip()
        if not q:
            return []

        # ------------------------------------------------------------------
        # select: exact-name path
        # ------------------------------------------------------------------
        if q.startswith("select:"):
            names = [n.strip() for n in q[len("select:"):].split(",") if n.strip()]
            results: list[ToolSchema] = []
            for name in names:
                t = self._tools.get(name)
                if t is not None:
                    self._activated.add(name)
                    results.append(t.schema)
            return results

        # ------------------------------------------------------------------
        # Weighted scoring path
        # ------------------------------------------------------------------
        ql = q.lower()
        scored: list[tuple[int, ToolSchema]] = []
        for t in self._tools.values():
            name_lower = t.name.lower()
            score = 0
            if ql == name_lower:
                score += 10
            elif ql in name_lower:
                score += 5
            if ql in t.description.lower():
                score += 2
            if score > 0:
                scored.append((score, t.schema))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:limit]

        for _, schema in top:
            self._activated.add(schema.name)

        return [schema for _, schema in top]


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
