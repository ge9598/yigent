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


# Prompt long enough to bypass the CapabilityRouter fast-path heuristic
# (default: skip aux for prompts ≤ 20 words with no plan keywords).
_AUX_PROMPT = (
    "please evaluate the following situation carefully and return your "
    "classification so that I can see whether the aux provider is being "
    "called as expected in this unit test scenario here"
)


@pytest.mark.asyncio
async def test_simple_task_routes_direct():
    provider = _FakeAuxProvider('{"strategy": "direct", "reason": "single edit"}')
    router = CapabilityRouter(aux_provider=provider)
    decision = await router.classify(
        "add a comment to foo.py explaining why we handle the retry loop this way "
        "and what the edge cases are that actually matter in production today"
    )
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
    decision = await router.classify(_AUX_PROMPT)
    assert decision.strategy == "direct"
    assert "unparseable" in decision.reason.lower() or "default" in decision.reason.lower()


@pytest.mark.asyncio
async def test_unknown_strategy_value_defaults_to_direct():
    provider = _FakeAuxProvider('{"strategy": "weird_option", "reason": "x"}')
    router = CapabilityRouter(aux_provider=provider)
    decision = await router.classify(_AUX_PROMPT)
    assert decision.strategy == "direct"


@pytest.mark.asyncio
async def test_none_aux_provider_defaults_to_direct():
    router = CapabilityRouter(aux_provider=None)
    decision = await router.classify(_AUX_PROMPT)
    assert decision.strategy == "direct"


@pytest.mark.asyncio
async def test_aux_provider_exception_defaults_to_direct():
    class _BoomProvider:
        async def stream_message(self, messages, model=None, tools=None, temperature=0.0):
            raise RuntimeError("network down")
            yield  # make this a generator

    router = CapabilityRouter(aux_provider=_BoomProvider())
    decision = await router.classify(_AUX_PROMPT)
    assert decision.strategy == "direct"
    assert "error" in decision.reason.lower()


@pytest.mark.asyncio
async def test_strips_code_fences_from_aux_output():
    """Some aux LLMs wrap JSON in markdown code fences."""
    provider = _FakeAuxProvider(
        '```json\n{"strategy": "plan_then_execute", "reason": "fenced"}\n```'
    )
    router = CapabilityRouter(aux_provider=provider)
    decision = await router.classify(_AUX_PROMPT)
    assert decision.strategy == "plan_then_execute"
    assert decision.reason == "fenced"


@pytest.mark.asyncio
async def test_decision_reason_defaults_to_empty_string():
    """Missing 'reason' field must not crash; reason is optional."""
    provider = _FakeAuxProvider('{"strategy": "direct"}')
    router = CapabilityRouter(aux_provider=provider)
    decision = await router.classify(_AUX_PROMPT)
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
        d = await router.classify(_AUX_PROMPT)
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
    d = await router.classify(
        "please edit foo.py and bar.py so that they share the new helper "
        "function we just discussed with whatever signature you think is best"
    )
    assert d.strategy == "direct"
    assert d.capabilities == ["coding", "file_ops"]


@pytest.mark.asyncio
async def test_classifier_filters_invalid_capabilities():
    provider = _FakeAuxProvider(
        '{"strategy": "direct", "reason": "ok", '
        '"capabilities": ["coding", "telepathy", "search"]}'
    )
    router = CapabilityRouter(aux_provider=provider)
    d = await router.classify(_AUX_PROMPT)
    assert "telepathy" not in d.capabilities
    assert d.capabilities == ["coding", "search"]


@pytest.mark.asyncio
async def test_classifier_missing_capabilities_field_defaults_empty():
    provider = _FakeAuxProvider('{"strategy": "direct", "reason": "ok"}')
    router = CapabilityRouter(aux_provider=provider)
    d = await router.classify(_AUX_PROMPT)
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


# ---------------------------------------------------------------------------
# Fast-path heuristic (audit Top-10 #2) — short prompts skip aux-LLM entirely
# ---------------------------------------------------------------------------

class _TrackingAuxProvider:
    """Raises if called — used to verify the fast-path doesn't invoke aux."""
    def __init__(self):
        self.called = False

    async def stream_message(self, messages, model=None, tools=None, temperature=0.0):
        self.called = True
        raise AssertionError("fast-path should have bypassed aux provider")
        yield  # make it a generator


@pytest.mark.asyncio
async def test_fast_path_short_prompt_skips_aux():
    """A short prompt with no planning keywords must not call the aux LLM."""
    aux = _TrackingAuxProvider()
    router = CapabilityRouter(aux_provider=aux)
    d = await router.classify("list memory")
    assert d.strategy == "direct"
    assert aux.called is False
    assert "fast-path" in d.reason


@pytest.mark.asyncio
async def test_fast_path_empty_prompt_skips_aux():
    aux = _TrackingAuxProvider()
    router = CapabilityRouter(aux_provider=aux)
    d = await router.classify("")
    assert d.strategy == "direct"
    assert aux.called is False


@pytest.mark.asyncio
async def test_fast_path_skipped_on_plan_keyword():
    """'refactor' is a plan-trigger keyword even in a short prompt →
    fast-path bows out and the aux LLM is consulted."""
    provider = _FakeAuxProvider(
        '{"strategy": "plan_then_execute", "reason": "keyword"}'
    )
    router = CapabilityRouter(aux_provider=provider)
    d = await router.classify("refactor auth")  # only 2 words
    assert d.strategy == "plan_then_execute"


@pytest.mark.asyncio
async def test_fast_path_skipped_on_long_prompt():
    provider = _FakeAuxProvider('{"strategy": "direct", "reason": "long"}')
    router = CapabilityRouter(aux_provider=provider)
    d = await router.classify(_AUX_PROMPT)  # > 20 words
    assert d.strategy == "direct"
    assert d.reason == "long"  # came from aux, not fast-path


def test_fast_path_method_returns_none_for_long_prompt():
    router = CapabilityRouter(aux_provider=None)
    result = router.fast_path(_AUX_PROMPT)
    assert result is None


def test_fast_path_method_returns_none_on_chinese_plan_keyword():
    router = CapabilityRouter(aux_provider=None)
    # 重构 is in _PLAN_TRIGGER_KEYWORDS
    result = router.fast_path("帮我重构一下")
    assert result is None
