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


def test_task_type_to_route_mapping_covers_injector_vocabulary():
    """agent_loop maps env_injector task types into CCR route keys.

    Without this mapping only the 'default' route would ever fire because
    EnvironmentInjector.detect_task_type returns coding/data_analysis/
    file_ops/research which don't overlap with the CCR route names.
    """
    from src.core.agent_loop import _TASK_TYPE_TO_ROUTE
    from src.core.env_injector import _TASK_KEYWORDS

    # Every injector task type must have an explicit route mapping.
    for injector_type in _TASK_KEYWORDS:
        assert injector_type in _TASK_TYPE_TO_ROUTE, (
            f"injector task type {injector_type!r} has no route mapping"
        )

    # Mapped values must be valid CCR route keys.
    valid_route_keys = {"default", "background", "long_context", "thinking"}
    for injector_type, route_key in _TASK_TYPE_TO_ROUTE.items():
        assert route_key in valid_route_keys, (
            f"{injector_type!r} maps to invalid route {route_key!r}"
        )


@pytest.mark.asyncio
async def test_agent_loop_maps_task_type_to_route_key():
    """Integration: data_analysis task type should select long_context route."""
    from src.core.agent_loop import _TASK_TYPE_TO_ROUTE

    calls = []

    class _RecordingProvider:
        async def stream_message(self, messages, model=None, tools=None, temperature=0.0):
            calls.append(model)
            from src.core.types import StreamChunk
            yield StreamChunk(type="done", data="stop", model=model or "test")

    fast = _RecordingProvider()
    big = _RecordingProvider()
    router = ScenarioRouter(
        providers={"fast": fast, "big": big},
        routes={
            "default": {"provider": "fast", "model": "m-fast"},
            "long_context": {"provider": "big", "model": "m-big"},
            "background": {"provider": "fast", "model": "m-cheap"},
        },
    )

    # Simulate the agent_loop translation step for a data_analysis task
    injector_task_type = "data_analysis"
    route_key = _TASK_TYPE_TO_ROUTE.get(injector_task_type, injector_task_type)
    assert route_key == "long_context"
    _, model = router.select(route_key)
    assert model == "m-big"

    # research → background
    route_key = _TASK_TYPE_TO_ROUTE.get("research", "research")
    assert route_key == "background"
    _, model = router.select(route_key)
    assert model == "m-cheap"

    # Unmapped type passes through unchanged; router falls back to default
    route_key = _TASK_TYPE_TO_ROUTE.get("unknown_type", "unknown_type")
    assert route_key == "unknown_type"
    _, model = router.select(route_key)
    assert model == "m-fast"  # default fallback


# ---------------------------------------------------------------------------
# Unit 4 — cross-provider routing via providers: alias map
# ---------------------------------------------------------------------------

def test_resolver_builds_cross_provider_router():
    """Routes can reference provider aliases declared under `providers:`."""
    from src.core.config import AgentConfig, ProviderConfig, ProviderSection
    from src.providers.resolver import resolve_provider, resolve_scenario_router

    cfg = AgentConfig(
        provider=ProviderSection(
            name="openai_compat",
            api_key="k",
            base_url="https://primary.test/v1",
            model="primary-model",
            providers={
                # Alias "reasoner" points at a different provider class + base.
                "reasoner": ProviderConfig(
                    name="anthropic_compat",
                    api_key="ak",
                    base_url="https://anthropic.test",
                    model="claude-x",
                ),
            },
            routes={
                "default": {"provider": "openai_compat", "model": "primary-model"},
                "thinking": {"provider": "reasoner", "model": "claude-x"},
            },
        )
    )
    primary = resolve_provider(cfg)
    router = resolve_scenario_router(cfg, primary)
    assert router is not None
    thinking_provider, thinking_model = router.select("thinking")
    default_provider, _ = router.select("default")
    # Different provider instances for different routes
    assert thinking_provider is not default_provider
    assert thinking_model == "claude-x"


def test_resolver_alias_colliding_with_primary_name_is_skipped(caplog):
    """Aliases colliding with the primary's name should log a warning and drop
    the alias — the primary wins."""
    import logging
    from src.core.config import AgentConfig, ProviderConfig, ProviderSection
    from src.providers.resolver import resolve_provider, resolve_scenario_router

    cfg = AgentConfig(
        provider=ProviderSection(
            name="openai_compat",
            api_key="k",
            base_url="https://primary.test/v1",
            model="primary-model",
            providers={
                # Same name as primary — must be skipped.
                "openai_compat": ProviderConfig(
                    name="openai_compat",
                    api_key="other",
                    base_url="https://other.test/v1",
                    model="other-model",
                ),
            },
            routes={
                "default": {"provider": "openai_compat", "model": "primary-model"},
            },
        )
    )
    primary = resolve_provider(cfg)
    with caplog.at_level(logging.WARNING):
        router = resolve_scenario_router(cfg, primary)
    assert router is not None
    # The primary instance (not the collided alias) is what's used.
    provider_for_default, _ = router.select("default")
    assert provider_for_default is primary
    assert any("collides with primary" in r.message for r in caplog.records)


def test_resolver_route_to_missing_alias_raises_at_build():
    """Route that references neither primary nor any known alias must fail
    loudly at router construction."""
    from src.core.config import AgentConfig, ProviderSection
    from src.providers.resolver import resolve_provider, resolve_scenario_router

    cfg = AgentConfig(
        provider=ProviderSection(
            name="openai_compat",
            api_key="k",
            base_url="https://primary.test/v1",
            model="m",
            routes={
                "default": {"provider": "openai_compat", "model": "m"},
                "thinking": {"provider": "no_such_alias", "model": "m"},
            },
        )
    )
    primary = resolve_provider(cfg)
    with pytest.raises(ValueError, match="unknown provider"):
        resolve_scenario_router(cfg, primary)


def test_resolver_fallback_provider_reachable_via_routes():
    """The fallback provider, if named, should also be routable by alias."""
    from src.core.config import (
        AgentConfig, ProviderConfig, ProviderSection,
    )
    from src.providers.resolver import resolve_provider, resolve_scenario_router

    cfg = AgentConfig(
        provider=ProviderSection(
            name="openai_compat",
            api_key="k",
            base_url="https://primary.test/v1",
            model="primary-model",
            fallback=ProviderConfig(
                name="anthropic_compat",
                api_key="ak",
                base_url="https://anthropic.test",
                model="fb-model",
            ),
            routes={
                "default": {"provider": "openai_compat", "model": "primary-model"},
                "thinking": {"provider": "anthropic_compat", "model": "fb-model"},
            },
        )
    )
    primary = resolve_provider(cfg)
    router = resolve_scenario_router(cfg, primary)
    assert router is not None
    p_thinking, m = router.select("thinking")
    assert m == "fb-model"
    assert p_thinking is not primary
