"""Regression tests for Fix 5 — MCP connect() cleanup on partial failure.

Prior to the fix, if __aenter__ succeeded but a subsequent step (typically
list_tools) raised, the context manager was never __aexit__'d. The next
connect() silently overwrote _session_cm, leaking the subprocess / SSE
connection until process exit.
"""

from __future__ import annotations

import pytest

from src.tools.mcp_adapter import MCPClient
from src.tools.registry import ToolRegistry


class _ListToolsFails:
    """Session whose list_tools() always raises."""
    async def list_tools(self):
        raise RuntimeError("transient network error")

    async def call_tool(self, name, args):  # pragma: no cover
        raise NotImplementedError


class _FakeCMWithExitTracking:
    """Fake context manager that records whether __aexit__ ran and with what exc."""
    def __init__(self, session):
        self._session = session
        self.entered = False
        self.exited = False
        self.exit_exc_type = None

    async def __aenter__(self):
        self.entered = True
        return self._session

    async def __aexit__(self, exc_type, exc, tb):
        self.exited = True
        self.exit_exc_type = exc_type
        return None


@pytest.mark.asyncio
async def test_connect_failure_releases_session():
    cm = _FakeCMWithExitTracking(_ListToolsFails())
    client = MCPClient(server_name="x", session_factory=lambda: cm)

    with pytest.raises(RuntimeError, match="transient"):
        await client.connect(ToolRegistry())

    assert cm.entered is True
    assert cm.exited is True, "must call __aexit__ to release the session"
    assert cm.exit_exc_type is RuntimeError
    # Client state is cleared so retries are safe.
    assert client._session is None
    assert client._session_cm is None


@pytest.mark.asyncio
async def test_retry_after_failed_connect_succeeds():
    """After a failed connect, a second connect with a good factory works."""
    bad_cm = _FakeCMWithExitTracking(_ListToolsFails())

    class _GoodSession:
        async def list_tools(self):
            class _T: pass
            class _Tool:
                name = "ping"
                description = "d"
                inputSchema = {"type": "object", "properties": {}}
            r = _T()
            r.tools = [_Tool()]
            return r
        async def call_tool(self, *a, **kw):  # pragma: no cover
            raise NotImplementedError

    class _GoodCM:
        async def __aenter__(self): return _GoodSession()
        async def __aexit__(self, *a): return None

    # Use a factory that returns bad_cm once, then good on retry.
    state = {"attempt": 0}
    def factory():
        state["attempt"] += 1
        return bad_cm if state["attempt"] == 1 else _GoodCM()

    client = MCPClient(server_name="srv", session_factory=factory)
    registry = ToolRegistry()

    with pytest.raises(RuntimeError):
        await client.connect(registry)

    # Retry: must not raise "already connected" because state was cleared.
    await client.connect(registry)
    assert registry.get_definition("srv__ping") is not None


@pytest.mark.asyncio
async def test_double_connect_raises_instead_of_silent_overwrite():
    """Calling connect() twice without close() in between is a programming
    error — prefer a loud failure over silent leak."""

    class _OKSession:
        async def list_tools(self):
            class _T: pass
            r = _T(); r.tools = []
            return r
        async def call_tool(self, *a, **kw):  # pragma: no cover
            raise NotImplementedError

    class _OKCM:
        async def __aenter__(self): return _OKSession()
        async def __aexit__(self, *a): return None

    client = MCPClient(server_name="srv", session_factory=lambda: _OKCM())
    await client.connect(ToolRegistry())

    with pytest.raises(RuntimeError, match="already connected"):
        await client.connect(ToolRegistry())
