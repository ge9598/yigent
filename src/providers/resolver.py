"""Runtime provider resolution from config."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .anthropic_compat import AnthropicCompatProvider
from .deepseek import DeepSeekProvider
from .openai_compat import OpenAICompatProvider

if TYPE_CHECKING:
    from src.core.config import AgentConfig, ProviderConfig, ProviderSection

    from .credential_pool import CredentialPool
    from .scenario_router import ScenarioRouter

from .base import LLMProvider

logger = logging.getLogger(__name__)

_PROVIDER_REGISTRY: dict[str, type[LLMProvider]] = {
    "deepseek": DeepSeekProvider,
    "openai_compat": OpenAICompatProvider,
    "anthropic_compat": AnthropicCompatProvider,
}


def _maybe_build_pool(section) -> "CredentialPool | None":
    """Return a CredentialPool if section has 2+ effective keys, else None."""
    from .credential_pool import CredentialPool

    keys = section.effective_keys()
    if len(keys) > 1:
        return CredentialPool(
            keys=keys,
            strategy=section.strategy,
            cooldown_seconds=section.cooldown_seconds,
        )
    return None


def _build_provider(
    name: str,
    api_key: str,
    base_url: str,
    model: str,
    debug: bool = False,
    credential_pool: "CredentialPool | None" = None,
) -> LLMProvider:
    cls = _PROVIDER_REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown provider '{name}'. Available: {list(_PROVIDER_REGISTRY)}"
        )
    if credential_pool is None and not api_key:
        raise ValueError(f"Provider '{name}' requires an API key or credential pool")
    return cls(
        api_key=api_key or "",
        base_url=base_url,
        model=model,
        debug=debug,
        credential_pool=credential_pool,
    )


def _single_key(section) -> str:
    """Return a single fallback api_key: prefer section.api_key, else keys[0]."""
    if section.api_key:
        return section.api_key
    effective = section.effective_keys()
    return effective[0] if effective else ""


def resolve_provider(config: AgentConfig) -> LLMProvider:
    """Resolve the primary LLM provider from config.

    Falls back to ``config.provider.fallback`` if the primary fails to init.
    """
    section = config.provider
    debug = config.ui.debug
    pool = _maybe_build_pool(section)

    try:
        return _build_provider(
            name=section.name,
            api_key=_single_key(section),
            base_url=section.base_url,
            model=section.model,
            debug=debug,
            credential_pool=pool,
        )
    except Exception as exc:
        if section.fallback is None:
            raise
        logger.warning("Primary provider failed (%s), trying fallback", exc)

    fb = section.fallback
    fb_pool = _maybe_build_pool(fb)
    return _build_provider(
        name=fb.name,
        api_key=_single_key(fb),
        base_url=fb.base_url,
        model=fb.model,
        credential_pool=fb_pool,
    )


def resolve_auxiliary(config: AgentConfig) -> LLMProvider | None:
    """Resolve the auxiliary provider (for compression, nudge, etc.)."""
    aux = config.provider.auxiliary
    if aux is None:
        return None
    try:
        return _build_provider(
            name=aux.name,
            api_key=aux.api_key or config.provider.api_key,
            base_url=aux.base_url or config.provider.base_url,
            model=aux.model,
        )
    except Exception as exc:
        logger.warning("Auxiliary provider failed (%s), skipping", exc)
        return None


def resolve_scenario_router(
    config: "AgentConfig",
    primary_provider: "LLMProvider",
) -> "ScenarioRouter | None":
    """Build a ScenarioRouter if routes are configured, else None.

    For this MVP, only the primary provider's name is available for routing
    — routes may not reference other provider names. (A future extension
    could add a top-level ``providers:`` map.)
    """
    from .scenario_router import ScenarioRouter

    section = config.provider
    if not section.routes:
        return None

    providers = {section.name: primary_provider}
    return ScenarioRouter(providers=providers, routes=section.routes)
