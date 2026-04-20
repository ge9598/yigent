"""DeepSeek provider — thin subclass of OpenAI-compatible provider."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .openai_compat import OpenAICompatProvider

if TYPE_CHECKING:
    from .credential_pool import CredentialPool


class DeepSeekProvider(OpenAICompatProvider):
    """DeepSeek V3 via their OpenAI-compatible API."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepseek.com/v1",
        model: str = "deepseek-chat",
        default_timeout: float = 120.0,
        debug: bool = False,
        credential_pool: "CredentialPool | None" = None,
    ) -> None:
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            model=model,
            default_timeout=default_timeout,
            debug=debug,
            credential_pool=credential_pool,
        )
