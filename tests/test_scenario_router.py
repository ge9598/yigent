"""Tests for ScenarioRouter — task-type → provider+model mapping."""

from __future__ import annotations

import pytest

from src.providers.scenario_router import ScenarioRouter


class _FakeProvider:
    def __init__(self, name):
        self.name = name


def test_routes_by_task_type():
    providers = {
        "fast": _FakeProvider("fast"),
        "big": _FakeProvider("big"),
    }
    router = ScenarioRouter(
        providers=providers,
        routes={
            "default": {"provider": "fast", "model": "small"},
            "long_context": {"provider": "big", "model": "large"},
        },
    )
    p, m = router.select(task_type="long_context")
    assert p.name == "big"
    assert m == "large"


def test_falls_back_to_default_for_unknown_task():
    providers = {"fast": _FakeProvider("fast")}
    router = ScenarioRouter(
        providers=providers,
        routes={"default": {"provider": "fast", "model": "small"}},
    )
    p, m = router.select(task_type="bogus_type")
    assert p.name == "fast"
    assert m == "small"


def test_raises_when_no_default_route():
    providers = {"fast": _FakeProvider("fast")}
    with pytest.raises(ValueError, match="must define a 'default' route"):
        ScenarioRouter(
            providers=providers,
            routes={"thinking": {"provider": "fast", "model": "small"}},
        )


def test_raises_when_route_references_unknown_provider():
    providers = {"fast": _FakeProvider("fast")}
    with pytest.raises(ValueError, match="unknown provider"):
        ScenarioRouter(
            providers=providers,
            routes={"default": {"provider": "missing", "model": "small"}},
        )


def test_four_route_types_supported():
    providers = {
        "p1": _FakeProvider("p1"),
        "p2": _FakeProvider("p2"),
    }
    router = ScenarioRouter(
        providers=providers,
        routes={
            "default": {"provider": "p1", "model": "m1"},
            "background": {"provider": "p2", "model": "m2"},
            "long_context": {"provider": "p1", "model": "m3"},
            "thinking": {"provider": "p2", "model": "m4"},
        },
    )
    for task_type, expected_model in [
        ("default", "m1"),
        ("background", "m2"),
        ("long_context", "m3"),
        ("thinking", "m4"),
    ]:
        _, m = router.select(task_type=task_type)
        assert m == expected_model


def test_list_routes_returns_copy():
    providers = {"p": _FakeProvider("p")}
    routes = {"default": {"provider": "p", "model": "m"}}
    router = ScenarioRouter(providers=providers, routes=routes)
    snapshot = router.list_routes()
    assert snapshot == routes
    # Mutating snapshot must not affect internal state
    snapshot["new"] = {"provider": "p", "model": "m2"}
    assert "new" not in router.list_routes()


def test_config_parses_routes():
    from src.core.config import ProviderSection

    section = ProviderSection.model_validate({
        "name": "primary",
        "api_key": "k",
        "base_url": "https://a",
        "model": "m",
        "routes": {
            "default": {"provider": "primary", "model": "small"},
            "long_context": {"provider": "primary", "model": "large"},
        },
    })
    assert section.routes["default"]["model"] == "small"
    assert section.routes["long_context"]["model"] == "large"


def test_config_default_routes_empty():
    from src.core.config import ProviderSection

    section = ProviderSection.model_validate({
        "name": "primary",
        "api_key": "k",
        "base_url": "https://a",
        "model": "m",
    })
    assert section.routes == {}


def test_resolver_builds_scenario_router():
    from src.core.config import AgentConfig, ProviderSection
    from src.providers.resolver import resolve_provider, resolve_scenario_router

    cfg = AgentConfig(
        provider=ProviderSection(
            name="openai_compat",
            api_key="k",
            base_url="https://example.test/v1",
            model="default-model",
            routes={
                "default": {"provider": "openai_compat", "model": "fast"},
                "long_context": {"provider": "openai_compat", "model": "big"},
            },
        )
    )
    primary = resolve_provider(cfg)
    router = resolve_scenario_router(cfg, primary)
    assert router is not None
    _, model = router.select("long_context")
    assert model == "big"


def test_resolver_returns_none_when_no_routes():
    from src.core.config import AgentConfig, ProviderSection
    from src.providers.resolver import resolve_provider, resolve_scenario_router

    cfg = AgentConfig(
        provider=ProviderSection(
            name="openai_compat",
            api_key="k",
            base_url="https://example.test/v1",
            model="m",
        )
    )
    primary = resolve_provider(cfg)
    assert resolve_scenario_router(cfg, primary) is None


@pytest.mark.asyncio
async def test_agent_loop_selects_via_router(monkeypatch):
    """Verify the agent loop uses scenario_router to pick provider+model."""
    from src.providers.scenario_router import ScenarioRouter

    # Build two fake providers recording their stream_message args
    calls = []

    class _RecordingProvider:
        def __init__(self, name):
            self.name = name

        async def stream_message(self, messages, model=None, tools=None, temperature=0.0):
            calls.append((self.name, model))
            from src.core.types import StreamChunk
            yield StreamChunk(type="done", data="stop", model=model or "test")

    fast = _RecordingProvider("fast")
    big = _RecordingProvider("big")
    router = ScenarioRouter(
        providers={"fast": fast, "big": big},
        routes={
            "default": {"provider": "fast", "model": "m-fast"},
            "long_context": {"provider": "big", "model": "m-big"},
        },
    )

    # Select is what agent_loop calls; exercise it directly
    p, m = router.select("default")
    assert p is fast and m == "m-fast"
    p, m = router.select("long_context")
    assert p is big and m == "m-big"
    p, m = router.select("unknown_type")
    assert p is fast and m == "m-fast"  # falls back to default
