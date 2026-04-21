"""MCP (Model Context Protocol) adapter.

Converts external MCP server tools into internal ToolDefinitions so they
look and behave like native tools. Two transports are supported:

  - stdio: spawn a subprocess speaking the MCP stdio protocol
  - sse:   connect to a Server-Sent Events endpoint (requires URL)

MCP specification: https://modelcontextprotocol.io/
Reference implementations: https://github.com/modelcontextprotocol/servers
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from src.core.types import PermissionLevel, ToolDefinition, ToolSchema

if TYPE_CHECKING:
    from src.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

# A callable that executes a tool on the MCP server: (raw_name, arguments) -> result
# The result is expected to have a ``.content`` list where each item has ``.type``
# and ``.text`` attributes, and an ``.isError`` bool (matching mcp.types.CallToolResult).
MCPToolCallable = Callable[[str, dict[str, Any]], Awaitable[Any]]


def mcp_tool_to_definition(
    mcp_tool: dict[str, Any],
    server_name: str,
    call_tool: MCPToolCallable | None,
    permission_level: PermissionLevel = PermissionLevel.READ_ONLY,
) -> ToolDefinition:
    """Convert an MCP tool dict into an internal ToolDefinition.

    Names are prefixed ``{server_name}__{tool_name}`` to prevent collisions
    when multiple servers expose tools with the same name.

    ``call_tool`` is the async function that invokes the tool on the server;
    it's closed over by the generated handler. For schema-only tests, pass None.

    ``permission_level`` is the security classification applied to every tool
    from this server. MCP has no per-tool permission metadata, so the caller
    must declare the risk level explicitly (via ``MCPServerConfig.default_permission``).
    """
    raw_name = mcp_tool["name"]
    prefixed = f"{server_name}__{raw_name}"
    description = mcp_tool.get("description") or f"MCP tool {raw_name} from {server_name}"
    parameters = mcp_tool.get("inputSchema") or {"type": "object", "properties": {}}

    async def _handler(**kwargs: Any) -> str:
        if call_tool is None:
            raise RuntimeError(
                f"MCP tool {prefixed} has no active session (call_tool is None)"
            )
        try:
            result = await call_tool(raw_name, kwargs)
        except Exception as exc:
            logger.warning("MCP call %s failed: %s", prefixed, exc)
            return f"MCP tool error: {exc}"
        return _format_mcp_result(result)

    schema = ToolSchema(
        name=prefixed,
        description=description,
        parameters=parameters,
        permission_level=permission_level,
        timeout=60,
        deferred=False,
    )
    return ToolDefinition(
        name=prefixed,
        description=description,
        handler=_handler,
        schema=schema,
        validate=None,
        needs_context=False,
    )


def _format_mcp_result(result: Any) -> str:
    """Extract text content from an MCP CallToolResult-shaped response."""
    content = getattr(result, "content", None)
    if content is None:
        return str(result)
    parts: list[str] = []
    for item in content:
        itype = getattr(item, "type", None)
        if itype == "text":
            parts.append(getattr(item, "text", ""))
        else:
            parts.append(str(item))
    text = "\n".join(p for p in parts if p)
    is_error = bool(getattr(result, "isError", False))
    if is_error and text:
        return f"[error] {text}"
    return text or ""


class MCPClient:
    """Owns one MCP session and registers its tools into a ToolRegistry.

    Transports:
      - ``stdio``: spawn a subprocess speaking the MCP stdio protocol
      - ``sse``:  connect to an SSE endpoint at ``url`` with optional ``headers``

    The lifecycle is::

        client = MCPClient(name="git", command=["uvx", "mcp-server-git"])
        await client.connect(registry)
        # ... use the registry; tools from this server are now active ...
        await client.close()
    """

    def __init__(
        self,
        server_name: str,
        transport: str = "stdio",
        command: list[str] | None = None,
        url: str = "",
        env: dict[str, str] | None = None,
        headers: dict[str, str] | None = None,
        session_factory: Callable[[], Any] | None = None,
        default_permission: str | PermissionLevel = PermissionLevel.READ_ONLY,
    ) -> None:
        self._server_name = server_name
        self._transport = transport
        self._command = list(command or [])
        self._url = url
        self._env = dict(env or {})
        self._headers = dict(headers or {})
        # Classify every tool from this server at this permission level.
        # Accept either the enum or its string value (for YAML config ergonomics).
        if isinstance(default_permission, str):
            try:
                self._default_permission = PermissionLevel(default_permission)
            except ValueError as exc:
                valid = [p.value for p in PermissionLevel]
                raise ValueError(
                    f"Unknown default_permission {default_permission!r} for MCP "
                    f"server {server_name!r}. Valid: {valid}"
                ) from exc
        else:
            self._default_permission = default_permission
        # Test seam: allow tests to inject a fake session context manager
        self._session_factory = session_factory
        self._session: Any | None = None
        self._session_cm: Any | None = None  # the async context returned by factory

    async def connect(self, registry: "ToolRegistry") -> None:
        """Open the session and register all tools into ``registry``.

        If any step after ``__aenter__`` raises (e.g. ``list_tools`` fails
        with a transient network error), we call ``__aexit__`` on the
        half-opened session before re-raising so the subprocess / SSE
        connection doesn't leak.
        """
        if self._session_cm is not None:
            raise RuntimeError(
                f"MCP client for {self._server_name!r} already connected"
            )
        if self._session_factory is not None:
            self._session_cm = self._session_factory()
        elif self._transport == "stdio":
            self._session_cm = self._build_stdio_session()
        elif self._transport == "sse":
            self._session_cm = self._build_sse_session()
        else:
            raise ValueError(f"Unknown MCP transport: {self._transport!r}")

        try:
            self._session = await self._session_cm.__aenter__()

            tools_response = await self._session.list_tools()
            raw_tools = getattr(tools_response, "tools", [])
            for mcp_tool in raw_tools:
                tool_dict = {
                    "name": getattr(mcp_tool, "name", None),
                    "description": getattr(mcp_tool, "description", None),
                    "inputSchema": getattr(mcp_tool, "inputSchema", None),
                }
                if not tool_dict["name"]:
                    logger.warning("Skipping MCP tool without a name on %s", self._server_name)
                    continue
                definition = mcp_tool_to_definition(
                    tool_dict,
                    server_name=self._server_name,
                    call_tool=self._session.call_tool,
                    permission_level=self._default_permission,
                )
                try:
                    registry.register(definition)
                except ValueError as exc:
                    logger.warning(
                        "Skipping duplicate MCP tool %s: %s", definition.name, exc
                    )
        except BaseException:
            # Partial open — release what we opened before the caller retries.
            try:
                import sys
                await self._session_cm.__aexit__(*sys.exc_info())
            except Exception as cleanup_exc:
                logger.debug(
                    "MCP cleanup on failed connect raised: %s", cleanup_exc,
                )
            self._session_cm = None
            self._session = None
            raise

    async def close(self) -> None:
        """Close the session. Safe to call multiple times."""
        if self._session_cm is None:
            return
        try:
            await self._session_cm.__aexit__(None, None, None)
        except Exception as exc:
            logger.debug("MCP session close raised: %s", exc)
        finally:
            self._session = None
            self._session_cm = None

    def _build_stdio_session(self) -> Any:
        """Construct a stdio session via the mcp SDK.

        Deferred to runtime so the mcp import is only paid when actually used.
        """
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError as exc:
            raise RuntimeError(
                "mcp SDK not installed. Run: pip install 'mcp>=1.0.0'"
            ) from exc

        if not self._command:
            raise ValueError("stdio transport requires 'command' (list of args)")

        params = StdioServerParameters(
            command=self._command[0],
            args=self._command[1:],
            env=self._env or None,
        )

        class _StdioSessionContext:
            def __init__(self, params):
                self._params = params
                self._stdio_ctx = None
                self._session_ctx = None

            async def __aenter__(self):
                self._stdio_ctx = stdio_client(self._params)
                read, write = await self._stdio_ctx.__aenter__()
                self._session_ctx = ClientSession(read, write)
                session = await self._session_ctx.__aenter__()
                await session.initialize()
                return session

            async def __aexit__(self, exc_type, exc, tb):
                try:
                    if self._session_ctx is not None:
                        await self._session_ctx.__aexit__(exc_type, exc, tb)
                finally:
                    if self._stdio_ctx is not None:
                        await self._stdio_ctx.__aexit__(exc_type, exc, tb)

        return _StdioSessionContext(params)

    def _build_sse_session(self) -> Any:
        """Construct an SSE session via the mcp SDK.

        Mirrors ``_build_stdio_session`` — open the SSE transport, wrap in a
        ``ClientSession``, ``initialize()``. Headers are forwarded so users can
        pass auth tokens (``Authorization: Bearer ...``).
        """
        if not self._url:
            raise ValueError("sse transport requires 'url'")

        try:
            from mcp import ClientSession
            from mcp.client.sse import sse_client
        except ImportError as exc:
            raise RuntimeError(
                "mcp SDK not installed or too old for SSE. "
                "Run: pip install 'mcp>=1.0.0'"
            ) from exc

        url = self._url
        headers = dict(self._headers)

        class _SseSessionContext:
            def __init__(self):
                self._sse_ctx = None
                self._session_ctx = None

            async def __aenter__(self):
                self._sse_ctx = sse_client(url, headers=headers or None)
                read, write = await self._sse_ctx.__aenter__()
                self._session_ctx = ClientSession(read, write)
                session = await self._session_ctx.__aenter__()
                await session.initialize()
                return session

            async def __aexit__(self, exc_type, exc, tb):
                try:
                    if self._session_ctx is not None:
                        await self._session_ctx.__aexit__(exc_type, exc, tb)
                finally:
                    if self._sse_ctx is not None:
                        await self._sse_ctx.__aexit__(exc_type, exc, tb)

        return _SseSessionContext()
