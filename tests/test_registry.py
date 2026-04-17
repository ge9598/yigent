import pytest
from src.core.types import ToolDefinition, ToolSchema, PermissionLevel
from src.tools.registry import ToolRegistry


def _make_tool(name: str, desc: str = "", deferred: bool = False) -> ToolDefinition:
    async def _noop() -> str:
        return ""

    return ToolDefinition(
        name=name,
        description=desc,
        handler=_noop,
        schema=ToolSchema(
            name=name,
            description=desc,
            parameters={"type": "object", "properties": {}},
            permission_level=PermissionLevel.READ_ONLY,
            deferred=deferred,
        ),
    )


class TestToolSearch:
    def setup_method(self):
        self.reg = ToolRegistry()
        self.reg.register(_make_tool("read_file", "Read a text file with line numbers"))
        self.reg.register(_make_tool("write_file", "Write content to a file"))
        self.reg.register(_make_tool("search_files", "Regex search across files"))
        self.reg.register(_make_tool("enter_plan_mode", "Enter plan mode", deferred=True))

    def test_exact_name_scores_highest(self):
        results = self.reg.tool_search("read_file")
        assert results[0].name == "read_file"

    def test_substring_in_name_beats_description(self):
        results = self.reg.tool_search("file")
        names = [r.name for r in results]
        assert "read_file" in names

    def test_select_syntax_exact_lookup(self):
        results = self.reg.tool_search("select:write_file,enter_plan_mode")
        names = {r.name for r in results}
        assert names == {"write_file", "enter_plan_mode"}

    def test_select_unknown_tool_skipped(self):
        results = self.reg.tool_search("select:write_file,nonexistent")
        assert len(results) == 1
        assert results[0].name == "write_file"

    def test_default_limit_is_5(self):
        for i in range(10):
            self.reg.register(_make_tool(f"tool_{i}", f"desc {i}"))
        results = self.reg.tool_search("tool")
        assert len(results) <= 5

    def test_deferred_tool_activated_by_search(self):
        assert not self.reg.is_activated("enter_plan_mode")
        self.reg.tool_search("plan")
        assert self.reg.is_activated("enter_plan_mode")

    def test_empty_query_returns_empty(self):
        assert self.reg.tool_search("") == []
