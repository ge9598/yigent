"""Tests for CapabilityRouter — aux LLM intent classifier."""

from __future__ import annotations

import pytest

from src.core.capability_router import CapabilityRouter, RoutingDecision
from src.core.types import StreamChunk


class _FakeAuxProvider:
    """Yields a canned token then 'done'."""

    def __init__(self, payload: str):
        self._payload = payload

    async def stream_message(self, messages, model=None, tools=None, temperature=0.0):
        yield StreamChunk(type="token", data=self._payload, model=model or "test")
        yield StreamChunk(type="done", data="stop", model=model or "test")


@pytest.mark.asyncio
async def test_simple_task_routes_direct():
    provider = _FakeAuxProvider('{"strategy": "direct", "reason": "single edit"}')
    router = CapabilityRouter(aux_provider=provider)
    decision = await router.classify("add a comment to foo.py")
    assert decision.strategy == "direct"
    assert "single edit" in decision.reason


@pytest.mark.asyncio
async def test_complex_task_routes_plan_then_execute():
    provider = _FakeAuxProvider(
        '{"strategy": "plan_then_execute", "reason": "multi-file refactor"}'
    )
    router = CapabilityRouter(aux_provider=provider)
    decision = await router.classify("refactor the auth subsystem across 12 files")
    assert decision.strategy == "plan_then_execute"


@pytest.mark.asyncio
async def test_malformed_json_defaults_to_direct():
    provider = _FakeAuxProvider("this is not json at all")
    router = CapabilityRouter(aux_provider=provider)
    decision = await router.classify("anything")
    assert decision.strategy == "direct"
    assert "unparseable" in decision.reason.lower() or "default" in decision.reason.lower()


@pytest.mark.asyncio
async def test_unknown_strategy_value_defaults_to_direct():
    provider = _FakeAuxProvider('{"strategy": "weird_option", "reason": "x"}')
    router = CapabilityRouter(aux_provider=provider)
    decision = await router.classify("anything")
    assert decision.strategy == "direct"


@pytest.mark.asyncio
async def test_none_aux_provider_defaults_to_direct():
    router = CapabilityRouter(aux_provider=None)
    decision = await router.classify("anything")
    assert decision.strategy == "direct"


@pytest.mark.asyncio
async def test_aux_provider_exception_defaults_to_direct():
    class _BoomProvider:
        async def stream_message(self, messages, model=None, tools=None, temperature=0.0):
            raise RuntimeError("network down")
            yield  # make this a generator

    router = CapabilityRouter(aux_provider=_BoomProvider())
    decision = await router.classify("anything")
    assert decision.strategy == "direct"
    assert "error" in decision.reason.lower()


@pytest.mark.asyncio
async def test_strips_code_fences_from_aux_output():
    """Some aux LLMs wrap JSON in markdown code fences."""
    provider = _FakeAuxProvider(
        '```json\n{"strategy": "plan_then_execute", "reason": "fenced"}\n```'
    )
    router = CapabilityRouter(aux_provider=provider)
    decision = await router.classify("anything")
    assert decision.strategy == "plan_then_execute"
    assert decision.reason == "fenced"


@pytest.mark.asyncio
async def test_decision_reason_defaults_to_empty_string():
    """Missing 'reason' field must not crash; reason is optional."""
    provider = _FakeAuxProvider('{"strategy": "direct"}')
    router = CapabilityRouter(aux_provider=provider)
    decision = await router.classify("anything")
    assert decision.strategy == "direct"
    assert decision.reason == ""


def test_routing_decision_dataclass():
    """RoutingDecision must be constructable with just strategy."""
    d = RoutingDecision(strategy="direct")
    assert d.strategy == "direct"
    assert d.reason == ""
    d2 = RoutingDecision(strategy="plan_then_execute", reason="why")
    assert d2.reason == "why"


@pytest.mark.asyncio
async def test_classify_with_valid_strategies_only():
    """Exhaustive strategy validation."""
    for strategy in ("direct", "plan_then_execute"):
        provider = _FakeAuxProvider(f'{{"strategy": "{strategy}", "reason": "ok"}}')
        router = CapabilityRouter(aux_provider=provider)
        d = await router.classify("x")
        assert d.strategy == strategy


def test_plan_mode_triggered_event_exists():
    """PlanModeTriggeredEvent must be a dataclass with a reason field."""
    from src.core.types import PlanModeTriggeredEvent

    ev = PlanModeTriggeredEvent(reason="complex task")
    assert ev.reason == "complex task"
    # Default reason
    ev2 = PlanModeTriggeredEvent()
    assert ev2.reason == ""


# ---------------------------------------------------------------------------
# Unit 10 — capabilities classifier output
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_classifier_returns_capabilities():
    provider = _FakeAuxProvider(
        '{"strategy": "direct", "reason": "ok", '
        '"capabilities": ["coding", "file_ops"]}'
    )
    router = CapabilityRouter(aux_provider=provider)
    d = await router.classify("edit foo.py and bar.py")
    assert d.strategy == "direct"
    assert d.capabilities == ["coding", "file_ops"]


@pytest.mark.asyncio
async def test_classifier_filters_invalid_capabilities():
    provider = _FakeAuxProvider(
        '{"strategy": "direct", "reason": "ok", '
        '"capabilities": ["coding", "telepathy", "search"]}'
    )
    router = CapabilityRouter(aux_provider=provider)
    d = await router.classify("x")
    assert "telepathy" not in d.capabilities
    assert d.capabilities == ["coding", "search"]


@pytest.mark.asyncio
async def test_classifier_missing_capabilities_field_defaults_empty():
    provider = _FakeAuxProvider('{"strategy": "direct", "reason": "ok"}')
    router = CapabilityRouter(aux_provider=provider)
    d = await router.classify("x")
    assert d.capabilities == []


def test_routing_decision_default_capabilities():
    d = RoutingDecision(strategy="direct")
    assert d.capabilities == []


def test_registry_activate_capability_group():
    """Registry pre-loads the obvious deferred tools for a capability group.

    Non-deferred tools auto-activate at register time, so we use deferred=True
    here to verify the capability-group activation path actually fires.
    """
    from src.core.types import (
        PermissionLevel, ToolDefinition, ToolSchema,
    )
    from src.tools.registry import ToolRegistry

    reg = ToolRegistry()

    async def noop(**kw):
        return ""

    for name in ("read_file", "write_file", "bash", "web_search", "search_files"):
        reg.register(ToolDefinition(
            name=name, description=name, handler=noop,
            schema=ToolSchema(
                name=name, description=name,
                parameters={"type": "object", "properties": {}},
                permission_level=PermissionLevel.READ_ONLY,
                deferred=True,  # so capability_group has work to do
            ),
        ))
    # Sanity: nothing activated yet
    assert reg.is_activated("read_file") is False

    activated = reg.activate_capability_group("coding")
    assert "read_file" in activated
    assert "write_file" in activated
    assert "bash" in activated
    # Now they're activated.
    assert reg.is_activated("read_file") is True
    # Calling again should not re-activate (idempotent).
    re_activated = reg.activate_capability_group("coding")
    assert re_activated == []


def test_registry_unknown_capability_returns_empty():
    from src.tools.registry import ToolRegistry
    reg = ToolRegistry()
    assert reg.activate_capability_group("telepathy") == []
