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
            billing_cooldown_seconds=getattr(
                section, "billing_cooldown_seconds", 86_400.0,
            ),
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
        name=fb.name or section.name,
        api_key=_single_key(fb) or _single_key(section),
        base_url=fb.base_url or section.base_url,
        model=fb.model or section.model,
        debug=debug,
        credential_pool=fb_pool,
    )


def resolve_auxiliary(config: AgentConfig) -> LLMProvider | None:
    """Resolve the auxiliary provider (for compression, nudge, etc.).

    Unset fields fall back to the primary provider — auxiliary defaults to a
    clone of the main provider unless explicitly overridden.
    """
    aux = config.provider.auxiliary
    if aux is None:
        return None
    primary = config.provider
    debug = config.ui.debug
    # Prefer aux's own keys/pool; fall back to primary's pool so single-key
    # primary configs keep working without a separate aux key. Without this
    # fallback, multi-key primary configs silently disable aux (classifier,
    # compression, nudge) because aux would hit "requires API key" with no key.
    pool = _maybe_build_pool(aux) or _maybe_build_pool(primary)
    try:
        return _build_provider(
            name=aux.name or primary.name,
            api_key=_single_key(aux) or _single_key(primary),
            base_url=aux.base_url or primary.base_url,
            model=aux.model or primary.model,
            debug=debug,
            credential_pool=pool,
        )
    except Exception as exc:
        logger.warning("Auxiliary provider failed (%s), skipping", exc)
        return None


def resolve_scenario_router(
    config: "AgentConfig",
    primary_provider: "LLMProvider",
) -> "ScenarioRouter | None":
    """Build a ScenarioRouter if routes are configured, else None.

    Builds a complete name→instance map covering:
      - the primary provider (name ``provider.name``)
      - any aliased providers under ``provider.providers`` (Unit 4)
      - the fallback provider (name ``provider.fallback.name``) when set
      - the auxiliary provider (name ``provider.auxiliary.name``) when set

    Aliased providers are instantiated eagerly. Failures during instantiation
    are logged and the alias is dropped from the map — a bad alias should not
    take the whole session down. Routes that reference a dropped alias will
    raise at ``ScenarioRouter`` construction (loud failure preferred over
    silent route fallback).
    """
    from .scenario_router import ScenarioRouter

    section = config.provider
    if not section.routes:
        return None

    providers: dict[str, "LLMProvider"] = {section.name: primary_provider}

    # Eagerly build aliased providers.
    for alias, sub in section.providers.items():
        if alias in providers:
            logger.warning(
                "Provider alias %r collides with primary provider name; "
                "alias entry skipped",
                alias,
            )
            continue
        try:
            sub_pool = _maybe_build_pool(sub)
            providers[alias] = _build_provider(
                name=sub.name or section.name,
                api_key=_single_key(sub) or _single_key(section),
                base_url=sub.base_url or section.base_url,
                model=sub.model or section.model,
                debug=config.ui.debug,
                credential_pool=sub_pool,
            )
        except Exception as exc:
            logger.warning(
                "Aliased provider %r failed to build (%s) — skipping", alias, exc,
            )

    # Surface fallback / auxiliary by their declared names too. This lets a
    # route point at "fallback" (or whatever it's named) without redeclaring.
    for sub_section in (section.fallback, section.auxiliary):
        if sub_section is None or not sub_section.name:
            continue
        if sub_section.name in providers:
            continue
        try:
            sub_pool = _maybe_build_pool(sub_section)
            providers[sub_section.name] = _build_provider(
                name=sub_section.name,
                api_key=_single_key(sub_section) or _single_key(section),
                base_url=sub_section.base_url or section.base_url,
                model=sub_section.model or section.model,
                debug=config.ui.debug,
                credential_pool=sub_pool,
            )
        except Exception as exc:
            logger.warning(
                "Could not bring %r online for routing (%s) — skipping",
                sub_section.name, exc,
            )

    return ScenarioRouter(providers=providers, routes=section.routes)
