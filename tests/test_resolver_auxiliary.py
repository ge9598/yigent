"""Regression tests for resolve_auxiliary — Fix 2 from the 2026-04-21 code review.

Prior to the fix, resolve_auxiliary didn't pass credential_pool or debug to
_build_provider. When primary used multi-key `keys: [...]` (so primary.api_key
is "" and keys live in the pool), aux silently fell back to None — disabling
the classifier, compression L3/L4, and any future aux-dependent feature.
"""

from src.core.config import AgentConfig, ProviderConfig, ProviderSection, UISection
from src.providers.resolver import resolve_auxiliary, resolve_provider


def test_aux_inherits_primary_pool_when_primary_is_multi_key():
    """Multi-key primary + bare aux section → aux should build with primary's pool."""
    cfg = AgentConfig(
        provider=ProviderSection(
            name="openai_compat",
            keys=["sk-a", "sk-b"],
            base_url="https://example.test/v1",
            model="primary-model",
            auxiliary=ProviderConfig(model="aux-model"),
        )
    )
    aux = resolve_auxiliary(cfg)
    assert aux is not None, "aux should not silently disable when primary has a pool"
    assert aux._credential_pool is not None
    assert aux._credential_pool.list_keys() == ["sk-a", "sk-b"]
    assert aux._default_model == "aux-model"


def test_aux_builds_its_own_pool_when_aux_multi_key():
    """Aux with its own keys list → aux pool is independent of primary."""
    cfg = AgentConfig(
        provider=ProviderSection(
            name="openai_compat",
            api_key="sk-primary",
            base_url="https://example.test/v1",
            model="primary-model",
            auxiliary=ProviderConfig(keys=["sk-x", "sk-y"], model="aux-model"),
        )
    )
    aux = resolve_auxiliary(cfg)
    assert aux is not None
    assert aux._credential_pool is not None
    assert aux._credential_pool.list_keys() == ["sk-x", "sk-y"]


def test_aux_single_key_primary_still_works():
    """Single-key primary + empty aux → aux inherits the primary key (no pool)."""
    cfg = AgentConfig(
        provider=ProviderSection(
            name="openai_compat",
            api_key="sk-primary",
            base_url="https://example.test/v1",
            model="primary-model",
            auxiliary=ProviderConfig(model="aux-model"),
        )
    )
    aux = resolve_auxiliary(cfg)
    assert aux is not None
    assert aux._credential_pool is None
    assert aux._fallback_key == "sk-primary"
    assert aux._default_model == "aux-model"


def test_aux_none_when_auxiliary_unset():
    cfg = AgentConfig(
        provider=ProviderSection(
            name="openai_compat",
            api_key="sk-primary",
            base_url="https://example.test/v1",
            model="primary-model",
        )
    )
    assert resolve_auxiliary(cfg) is None


def test_aux_debug_flag_propagates():
    """ui.debug=True should flow into the aux provider."""
    cfg = AgentConfig(
        provider=ProviderSection(
            name="openai_compat",
            api_key="sk-primary",
            base_url="https://example.test/v1",
            model="primary-model",
            auxiliary=ProviderConfig(model="aux-model"),
        ),
        ui=UISection(debug=True),
    )
    aux = resolve_auxiliary(cfg)
    assert aux is not None
    assert aux._debug is True


def test_fallback_debug_flag_propagates():
    """resolver's fallback branch also gets debug — regression from same fix."""
    cfg = AgentConfig(
        provider=ProviderSection(
            name="does_not_exist_provider",  # force primary build to fail
            api_key="sk-primary",
            base_url="https://example.test/v1",
            model="primary-model",
            fallback=ProviderConfig(
                name="openai_compat", api_key="sk-fb", model="fb-model",
            ),
        ),
        ui=UISection(debug=True),
    )
    provider = resolve_provider(cfg)
    assert provider._debug is True
