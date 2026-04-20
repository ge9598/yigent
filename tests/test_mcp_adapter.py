"""Tests for MCP adapter — schema conversion + client registration."""

from __future__ import annotations

import pytest

from src.tools.mcp_adapter import mcp_tool_to_definition


def test_converts_basic_mcp_tool():
    """An MCP tool dict must produce a ToolDefinition with prefixed name."""
    mcp_tool = {
        "name": "read_repo_file",
        "description": "Read a file from a git repo",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path"},
            },
            "required": ["path"],
        },
    }
    definition = mcp_tool_to_definition(mcp_tool, server_name="gitmcp", call_tool=None)
    assert definition.name == "gitmcp__read_repo_file"
    assert "Read a file" in definition.description
    assert definition.schema.name == "gitmcp__read_repo_file"
    assert definition.schema.parameters == mcp_tool["inputSchema"]


def test_name_collision_prevention_via_prefix():
    """Same tool name from different servers must produce distinct definitions."""
    def _call(*args, **kwargs):
        raise NotImplementedError

    a = mcp_tool_to_definition(
        {"name": "search", "description": "x", "inputSchema": {"type": "object"}},
        server_name="server_a",
        call_tool=_call,
    )
    b = mcp_tool_to_definition(
        {"name": "search", "description": "y", "inputSchema": {"type": "object"}},
        server_name="server_b",
        call_tool=_call,
    )
    assert a.name != b.name
    assert a.name == "server_a__search"
    assert b.name == "server_b__search"


def test_missing_description_fills_default():
    """Tool without a description should get a non-empty placeholder."""
    definition = mcp_tool_to_definition(
        {"name": "x", "inputSchema": {"type": "object"}},
        server_name="s",
        call_tool=None,
    )
    assert definition.description  # non-empty
    assert definition.schema.description  # non-empty


def test_missing_input_schema_gets_object_default():
    """Tool without inputSchema should default to an empty-object schema."""
    definition = mcp_tool_to_definition(
        {"name": "x", "description": "d"},
        server_name="s",
        call_tool=None,
    )
    assert definition.schema.parameters == {"type": "object", "properties": {}}


@pytest.mark.asyncio
async def test_proxy_handler_calls_mcp_session():
    """The generated handler should forward invocations to the MCP session."""
    forwarded_calls = []

    async def _fake_call(raw_name: str, arguments: dict):
        forwarded_calls.append((raw_name, arguments))
        # Mimic MCP response shape
        return type("R", (), {
            "content": [type("C", (), {"type": "text", "text": "echoed: " + arguments.get("msg", "")})()],
            "isError": False,
        })()

    definition = mcp_tool_to_definition(
        {
            "name": "echo",
            "description": "Echo a message",
            "inputSchema": {
                "type": "object",
                "properties": {"msg": {"type": "string"}},
            },
        },
        server_name="test",
        call_tool=_fake_call,
    )

    result = await definition.handler(msg="hello")
    assert "echoed: hello" in result
    assert forwarded_calls == [("echo", {"msg": "hello"})]


@pytest.mark.asyncio
async def test_proxy_handler_handles_error_result():
    """MCP error results should surface as plain string with error marker."""
    async def _fake_call(raw_name: str, arguments: dict):
        return type("R", (), {
            "content": [type("C", (), {"type": "text", "text": "boom"})()],
            "isError": True,
        })()

    definition = mcp_tool_to_definition(
        {"name": "fail", "description": "d", "inputSchema": {"type": "object"}},
        server_name="s",
        call_tool=_fake_call,
    )
    result = await definition.handler()
    assert "boom" in result
    # We stringify the error — the content reaches the model either way.


def test_mcp_server_config_model():
    """AgentConfig must accept a mcp_servers: list entry."""
    from src.core.config import AgentConfig

    cfg = AgentConfig.model_validate({
        "mcp_servers": [
            {
                "name": "git",
                "transport": "stdio",
                "command": ["uvx", "mcp-server-git"],
                "env": {"FOO": "bar"},
            },
        ]
    })
    assert len(cfg.mcp_servers) == 1
    assert cfg.mcp_servers[0].name == "git"
    assert cfg.mcp_servers[0].transport == "stdio"
    assert cfg.mcp_servers[0].command == ["uvx", "mcp-server-git"]
    assert cfg.mcp_servers[0].env == {"FOO": "bar"}


def test_mcp_server_config_defaults():
    """Optional fields should default sensibly."""
    from src.core.config import MCPServerConfig

    m = MCPServerConfig.model_validate({"name": "x"})
    assert m.transport == "stdio"
    assert m.command == []
    assert m.url == ""
    assert m.env == {}


@pytest.mark.asyncio
async def test_mcp_client_registers_tools_via_session_factory():
    """Inject a fake session → its tools should land in the registry."""
    from src.tools.mcp_adapter import MCPClient
    from src.tools.registry import ToolRegistry

    class _FakeSession:
        async def list_tools(self):
            class _T:
                pass
            t = _T()
            t.tools = [
                type("Tool", (), {
                    "name": "echo",
                    "description": "Echo input",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"msg": {"type": "string"}},
                    },
                })(),
            ]
            return t

        async def call_tool(self, name, arguments):
            return type("R", (), {
                "content": [type("C", (), {"type": "text", "text": f"echoed: {arguments.get('msg')}"})()],
                "isError": False,
            })()

    class _FakeSessionContext:
        async def __aenter__(self):
            return _FakeSession()

        async def __aexit__(self, *args):
            return None

    registry = ToolRegistry()
    client = MCPClient(
        server_name="echo_server",
        session_factory=lambda: _FakeSessionContext(),
    )
    await client.connect(registry)

    definition = registry.get_definition("echo_server__echo")
    assert definition is not None
    assert definition.schema.parameters["properties"]["msg"]["type"] == "string"

    result = await definition.handler(msg="hi")
    assert "echoed: hi" in result

    await client.close()


@pytest.mark.asyncio
async def test_mcp_client_sse_transport_not_implemented():
    from src.tools.mcp_adapter import MCPClient
    from src.tools.registry import ToolRegistry

    client = MCPClient(server_name="x", transport="sse", url="http://example")
    with pytest.raises(NotImplementedError):
        await client.connect(ToolRegistry())


@pytest.mark.asyncio
async def test_mcp_client_unknown_transport_raises():
    from src.tools.mcp_adapter import MCPClient
    from src.tools.registry import ToolRegistry

    client = MCPClient(server_name="x", transport="bogus")
    with pytest.raises(ValueError, match="Unknown MCP transport"):
        await client.connect(ToolRegistry())


def test_default_permission_is_read_only():
    """Without an explicit override, MCP tools default to read_only."""
    from src.core.types import PermissionLevel

    definition = mcp_tool_to_definition(
        {"name": "x", "description": "d", "inputSchema": {"type": "object"}},
        server_name="s",
        call_tool=None,
    )
    assert definition.schema.permission_level == PermissionLevel.READ_ONLY


def test_explicit_permission_level_propagates_to_schema():
    """Caller can raise the permission level for write/execute MCP tools."""
    from src.core.types import PermissionLevel

    definition = mcp_tool_to_definition(
        {"name": "x", "description": "d", "inputSchema": {"type": "object"}},
        server_name="s",
        call_tool=None,
        permission_level=PermissionLevel.EXECUTE,
    )
    assert definition.schema.permission_level == PermissionLevel.EXECUTE


@pytest.mark.asyncio
async def test_mcp_client_applies_default_permission_to_all_tools():
    """MCPClient(default_permission="write") must tag every registered tool."""
    from src.core.types import PermissionLevel
    from src.tools.mcp_adapter import MCPClient
    from src.tools.registry import ToolRegistry

    class _FakeSession:
        async def list_tools(self):
            class _T:
                pass
            t = _T()
            t.tools = [
                type("Tool", (), {
                    "name": "write_something",
                    "description": "x",
                    "inputSchema": {"type": "object"},
                })(),
                type("Tool", (), {
                    "name": "read_something",
                    "description": "y",
                    "inputSchema": {"type": "object"},
                })(),
            ]
            return t

        async def call_tool(self, name, arguments):
            return None

    class _FakeCM:
        async def __aenter__(self):
            return _FakeSession()

        async def __aexit__(self, *a):
            return None

    registry = ToolRegistry()
    client = MCPClient(
        server_name="fs",
        session_factory=lambda: _FakeCM(),
        default_permission="write",
    )
    await client.connect(registry)

    assert registry.get_definition("fs__write_something").schema.permission_level == PermissionLevel.WRITE
    assert registry.get_definition("fs__read_something").schema.permission_level == PermissionLevel.WRITE
    await client.close()


def test_mcp_client_rejects_invalid_permission_string():
    """Unknown permission level string must raise at construction time."""
    from src.tools.mcp_adapter import MCPClient

    with pytest.raises(ValueError, match="Unknown default_permission"):
        MCPClient(server_name="x", default_permission="bogus")


def test_mcp_server_config_default_permission_field():
    """Pydantic model accepts default_permission field with read_only default."""
    from src.core.config import MCPServerConfig

    m = MCPServerConfig.model_validate({"name": "x"})
    assert m.default_permission == "read_only"

    m2 = MCPServerConfig.model_validate({"name": "y", "default_permission": "execute"})
    assert m2.default_permission == "execute"
