"""Per-endpoint quirk detection.

Different LLM endpoints implement the "same" protocol with incompatible
twists. This module centralises the URL-based detection so providers don't
hard-code conditionals throughout their streaming paths.

Design mirrors Hermes Agent's ``_forbids_sampling_params`` / ``_is_third_party_anthropic_endpoint`` helpers
(see ``agent/anthropic_adapter.py``): lightweight predicate functions keyed
on ``base_url``, consumed by provider constructors to build an
:class:`EndpointQuirks` record applied once per request.

Add a new quirk here when a real endpoint breaks without it. Do not add
speculative quirks — they become dead code nobody can prune safely.
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse


@dataclass(frozen=True)
class EndpointQuirks:
    """Quirks to apply when talking to a specific endpoint."""

    # Third-party endpoints that speak the Anthropic protocol but don't honour
    # Anthropic's signed ``thinking`` blocks (MiniMax /anthropic, Bedrock proxies,
    # most self-hosted gateways). Assistant messages must have thinking
    # content stripped before re-sending.
    strip_thinking_signature: bool = False

    # Some endpoints reject temperature=0 (MiniMax /anthropic requires > 0).
    # When True, the provider clamps 0.0 → ``min_temperature``.
    forbids_zero_temperature: bool = False
    min_temperature: float = 0.01

    # DeepSeek caps max_tokens at 8192. OpenAI o1 series has its own caps.
    # None means "use provider default".
    max_tokens_cap: int | None = None

    # OpenAI o1 series rejects temperature / top_p / system role.
    forbids_sampling_params: bool = False


def detect_quirks(base_url: str) -> EndpointQuirks:
    """Return the :class:`EndpointQuirks` record for the given base_url.

    Detection is URL-substring based — we key on host+path rather than
    provider name, because the same provider can serve multiple protocol
    shapes at different URLs (MiniMax /v1 vs /anthropic).
    """
    if not base_url:
        return EndpointQuirks()

    host = (urlparse(base_url).hostname or "").lower()
    path = (urlparse(base_url).path or "").lower()

    # --- Anthropic-protocol endpoints ----------------------------------------

    is_anthropic_official = host.endswith("api.anthropic.com")
    is_anthropic_path = path.endswith("/anthropic") or "/anthropic/" in path

    if is_anthropic_path and not is_anthropic_official:
        # Third-party /anthropic gateway. Most prominent: MiniMax.
        base = _minimax_anthropic_quirks() if "minimax" in host else _generic_third_party_anthropic()
        return base

    # --- OpenAI-protocol endpoints -------------------------------------------

    if "deepseek.com" in host:
        return EndpointQuirks(max_tokens_cap=8192)

    if host.endswith("api.openai.com"):
        # OpenAI-native. o1 restrictions are model-specific, not URL-specific;
        # we don't detect them here — the provider inspects model name.
        return EndpointQuirks()

    # Generic OpenAI-compatible (Qwen, Moonshot, local vLLM, OpenRouter, ...).
    return EndpointQuirks()


def _minimax_anthropic_quirks() -> EndpointQuirks:
    """MiniMax /anthropic: strict ``temperature > 0``, strip signed thinking."""
    return EndpointQuirks(
        strip_thinking_signature=True,
        forbids_zero_temperature=True,
        min_temperature=0.01,
    )


def _generic_third_party_anthropic() -> EndpointQuirks:
    """Conservative defaults for any non-official /anthropic gateway."""
    return EndpointQuirks(strip_thinking_signature=True)


def model_forbids_sampling(model: str) -> bool:
    """Whether a model rejects temperature / top_p / system role.

    OpenAI o1 series: yes. Everything else: no. Detection is by model-name
    prefix because multiple endpoints (Azure, OpenAI, proxies) expose o1.
    """
    name = (model or "").lower()
    return name.startswith("o1") or name.startswith("o3") or name.startswith("o4")
