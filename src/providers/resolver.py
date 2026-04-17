"""Runtime provider resolution from config."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .deepseek import DeepSeekProvider
from .openai_compat import OpenAICompatProvider

if TYPE_CHECKING:
    from src.core.config import AgentConfig, ProviderConfig, ProviderSection

from .base import LLMProvider

logger = logging.getLogger(__name__)

_PROVIDER_REGISTRY: dict[str, type[LLMProvider]] = {
    "deepseek": DeepSeekProvider,
    "openai_compat": OpenAICompatProvider,
}


def _build_provider(name: str, api_key: str, base_url: str, model: str) -> LLMProvider:
    cls = _PROVIDER_REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown provider '{name}'. Available: {list(_PROVIDER_REGISTRY)}"
        )
    if not api_key:
        raise ValueError(f"Provider '{name}' requires an API key")
    return cls(api_key=api_key, base_url=base_url, model=model)


def resolve_provider(config: AgentConfig) -> LLMProvider:
    """Resolve the primary LLM provider from config.

    Falls back to ``config.provider.fallback`` if the primary fails to init.
    """
    section = config.provider

    try:
        return _build_provider(
            name=section.name,
            api_key=section.api_key,
            base_url=section.base_url,
            model=section.model,
        )
    except (ValueError, Exception) as exc:
        if section.fallback is None:
            raise
        logger.warning("Primary provider failed (%s), trying fallback", exc)

    fb = section.fallback
    return _build_provider(
        name=fb.name,
        api_key=fb.api_key,
        base_url=fb.base_url,
        model=fb.model,
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
    except (ValueError, Exception) as exc:
        logger.warning("Auxiliary provider failed (%s), skipping", exc)
        return None
