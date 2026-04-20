"""Tests for HookSystem — registration, firing, isolation, deny semantics."""

from __future__ import annotations

import pytest

from src.safety.hook_system import HookSystem, load_hooks_from_config


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def test_register_unknown_event_raises() -> None:
    h = HookSystem()
    with pytest.raises(ValueError, match="Unknown hook event"):
        h.register("nonsense", lambda **kw: None)  # type: ignore[arg-type]


def test_count_after_registration() -> None:
    h = HookSystem()
    h.register("session_start", lambda **kw: None)
    h.register("session_start", lambda **kw: None)
    assert h.count("session_start") == 2
    assert h.count("session_end") == 0


def test_clear_one_event() -> None:
    h = HookSystem()
    h.register("session_start", lambda **kw: None)
    h.register("session_end", lambda **kw: None)
    h.clear("session_start")
    assert h.count("session_start") == 0
    assert h.count("session_end") == 1


def test_clear_all() -> None:
    h = HookSystem()
    h.register("session_start", lambda **kw: None)
    h.register("session_end", lambda **kw: None)
    h.clear()
    assert h.count("session_start") == 0
    assert h.count("session_end") == 0


# ---------------------------------------------------------------------------
# Firing
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fires_sync_hook() -> None:
    h = HookSystem()
    seen: list = []
    h.register("session_start", lambda **kw: seen.append(kw))
    await h.fire("session_start", x=1)
    assert seen == [{"x": 1}]


@pytest.mark.asyncio
async def test_fires_async_hook() -> None:
    h = HookSystem()
    seen: list = []

    async def hook(**kw):
        seen.append(kw)

    h.register("session_start", hook)
    await h.fire("session_start", x=2)
    assert seen == [{"x": 2}]


@pytest.mark.asyncio
async def test_returns_allow_when_no_hooks() -> None:
    h = HookSystem()
    result = await h.fire("pre_tool_use", tool_call=None)
    assert result == "allow"


@pytest.mark.asyncio
async def test_deny_short_circuits_outcome() -> None:
    h = HookSystem()
    h.register("pre_tool_use", lambda **kw: "deny")
    result = await h.fire("pre_tool_use", tool_call=None)
    assert result == "deny"


@pytest.mark.asyncio
async def test_one_deny_among_allows_still_denies() -> None:
    h = HookSystem()
    h.register("pre_tool_use", lambda **kw: None)
    h.register("pre_tool_use", lambda **kw: "deny")
    h.register("pre_tool_use", lambda **kw: "allow")
    result = await h.fire("pre_tool_use", tool_call=None)
    assert result == "deny"


@pytest.mark.asyncio
async def test_failing_hook_is_isolated() -> None:
    """One broken hook must not break the chain or leak its exception."""
    h = HookSystem()
    seen: list = []

    def boom(**kw):
        raise RuntimeError("boom")

    h.register("session_start", boom)
    h.register("session_start", lambda **kw: seen.append("ok"))
    result = await h.fire("session_start")
    assert result == "allow"
    assert seen == ["ok"]


@pytest.mark.asyncio
async def test_unknown_event_fires_warns_returns_none() -> None:
    h = HookSystem()
    result = await h.fire("not_real")  # type: ignore[arg-type]
    assert result is None


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def test_loader_handles_empty_config() -> None:
    h = load_hooks_from_config({})
    assert h.count("session_start") == 0


def test_loader_skips_bad_event_keys() -> None:
    # No exception, but no registration either.
    h = load_hooks_from_config({"not_real": ["x.y"]})
    assert h.count("session_start") == 0


def test_loader_skips_non_list_values() -> None:
    h = load_hooks_from_config({"session_start": "not a list"})
    assert h.count("session_start") == 0


def test_loader_skips_unimportable_refs() -> None:
    h = load_hooks_from_config(
        {"session_start": ["nonexistent_module.nonexistent_func"]},
    )
    assert h.count("session_start") == 0
