"""Tests for CredentialPool — multi-key rotation with 429 cooldown."""

from __future__ import annotations

import time

import pytest

from src.providers.credential_pool import CredentialPool


def test_round_robin_rotation():
    pool = CredentialPool(keys=["k1", "k2", "k3"], strategy="round_robin")
    assert pool.acquire() == "k1"
    assert pool.acquire() == "k2"
    assert pool.acquire() == "k3"
    assert pool.acquire() == "k1"


def test_fill_first_sticks_until_error():
    pool = CredentialPool(keys=["k1", "k2", "k3"], strategy="fill_first")
    assert pool.acquire() == "k1"
    assert pool.acquire() == "k1"
    pool.mark_error("k1", status=500)
    assert pool.acquire() == "k2"
    assert pool.acquire() == "k2"


def test_least_used_picks_minimum():
    pool = CredentialPool(keys=["k1", "k2", "k3"], strategy="least_used")
    assert pool.acquire() == "k1"
    assert pool.acquire() == "k2"
    assert pool.acquire() == "k3"
    # All have usage=1; tie → k1 (insertion order)
    assert pool.acquire() == "k1"


def test_random_stays_within_pool():
    pool = CredentialPool(keys=["k1", "k2"], strategy="random", seed=42)
    for _ in range(20):
        assert pool.acquire() in {"k1", "k2"}


def test_429_cooldown_excludes_key_temporarily():
    pool = CredentialPool(keys=["k1", "k2"], strategy="round_robin", cooldown_seconds=0.1)
    assert pool.acquire() == "k1"
    pool.mark_error("k1", status=429)
    assert pool.acquire() == "k2"
    assert pool.acquire() == "k2"
    time.sleep(0.15)
    next_keys = {pool.acquire() for _ in range(4)}
    assert "k1" in next_keys


def test_non_429_rotates_but_does_not_cool():
    pool = CredentialPool(keys=["k1", "k2"], strategy="round_robin", cooldown_seconds=10)
    assert pool.acquire() == "k1"
    pool.mark_error("k1", status=500)
    assert pool.acquire() == "k2"
    assert pool.acquire() == "k1"


def test_all_keys_cooling_raises():
    pool = CredentialPool(keys=["k1", "k2"], strategy="round_robin", cooldown_seconds=10)
    pool.mark_error("k1", status=429)
    pool.mark_error("k2", status=429)
    with pytest.raises(RuntimeError, match="all keys in cooldown"):
        pool.acquire()


def test_single_key_works():
    pool = CredentialPool(keys=["k1"], strategy="fill_first")
    assert pool.acquire() == "k1"
    assert pool.acquire() == "k1"


def test_empty_keys_raises_at_init():
    with pytest.raises(ValueError, match="at least one key"):
        CredentialPool(keys=[], strategy="round_robin")


def test_unknown_strategy_raises():
    with pytest.raises(ValueError, match="Unknown strategy"):
        CredentialPool(keys=["k1"], strategy="bogus")


def test_add_remove_key():
    pool = CredentialPool(keys=["k1"], strategy="round_robin")
    pool.add_key("k2")
    acquired = {pool.acquire() for _ in range(4)}
    assert acquired == {"k1", "k2"}
    pool.remove_key("k1")
    assert pool.acquire() == "k2"


def test_remove_last_key_raises():
    pool = CredentialPool(keys=["k1"], strategy="round_robin")
    with pytest.raises(ValueError, match="cannot remove last key"):
        pool.remove_key("k1")


def test_set_strategy_at_runtime():
    pool = CredentialPool(keys=["k1", "k2", "k3"], strategy="round_robin")
    pool.acquire()
    pool.set_strategy("fill_first")
    first = pool.acquire()
    assert pool.acquire() == first


def test_round_robin_cursor_does_not_regress_on_late_error():
    """Out-of-order mark_error must not rewind the cursor."""
    pool = CredentialPool(keys=["k1", "k2", "k3"], strategy="round_robin")
    assert pool.acquire() == "k1"
    assert pool.acquire() == "k2"
    # Late arriving error from the k1 request
    pool.mark_error("k1", status=500)
    # Cursor was at position 2 — next acquire must advance to k3, not rewind to k2
    assert pool.acquire() == "k3"
    assert pool.acquire() == "k1"


def test_mark_error_on_unknown_key_is_noop():
    pool = CredentialPool(keys=["k1"], strategy="round_robin")
    # Should not raise, should not affect state
    pool.mark_error("never_added", status=429)
    assert pool.acquire() == "k1"


def test_config_accepts_keys_list():
    from src.core.config import ProviderConfig

    cfg = ProviderConfig.model_validate({
        "name": "deepseek",
        "keys": ["sk-a", "sk-b"],
        "strategy": "round_robin",
        "cooldown_seconds": 30,
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
    })
    assert cfg.keys == ["sk-a", "sk-b"]
    assert cfg.strategy == "round_robin"
    assert cfg.cooldown_seconds == 30
    assert cfg.effective_keys() == ["sk-a", "sk-b"]


def test_config_api_key_backward_compat():
    from src.core.config import ProviderConfig

    cfg = ProviderConfig.model_validate({
        "name": "deepseek",
        "api_key": "sk-legacy",
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
    })
    assert cfg.api_key == "sk-legacy"
    assert cfg.keys == []
    assert cfg.strategy == "round_robin"
    assert cfg.cooldown_seconds == 60.0
    assert cfg.effective_keys() == ["sk-legacy"]


def test_config_keys_takes_precedence_over_api_key():
    """When both are set, keys: wins."""
    from src.core.config import ProviderConfig

    cfg = ProviderConfig.model_validate({
        "name": "deepseek",
        "api_key": "sk-old",
        "keys": ["sk-new-a", "sk-new-b"],
        "base_url": "x",
        "model": "m",
    })
    assert cfg.effective_keys() == ["sk-new-a", "sk-new-b"]


def test_provider_section_effective_keys():
    from src.core.config import ProviderSection

    section = ProviderSection.model_validate({
        "name": "deepseek",
        "keys": ["sk-a", "sk-b", "sk-c"],
        "strategy": "fill_first",
        "cooldown_seconds": 120,
        "base_url": "x",
        "model": "m",
    })
    assert section.effective_keys() == ["sk-a", "sk-b", "sk-c"]
    assert section.strategy == "fill_first"


@pytest.mark.asyncio
async def test_openai_provider_acquires_key_per_request(monkeypatch):
    """OpenAICompatProvider, given a pool, must acquire a fresh key per call."""
    from src.providers.credential_pool import CredentialPool
    from src.providers.openai_compat import OpenAICompatProvider

    pool = CredentialPool(keys=["key-A", "key-B"], strategy="round_robin")

    acquired: list[str] = []

    class _FakeStream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

    class _FakeCompletions:
        async def create(self, **kwargs):
            return _FakeStream()

    class _FakeChat:
        def __init__(self):
            self.completions = _FakeCompletions()

    class _FakeClient:
        def __init__(self, api_key, **kwargs):
            acquired.append(api_key)
            self.chat = _FakeChat()

    monkeypatch.setattr("src.providers.openai_compat.AsyncOpenAI", _FakeClient)

    provider = OpenAICompatProvider(
        api_key="fallback",
        base_url="https://example.test/v1",
        model="test-model",
        credential_pool=pool,
    )

    async for _ in provider.stream_message(messages=[{"role": "user", "content": "hi"}]):
        pass
    async for _ in provider.stream_message(messages=[{"role": "user", "content": "hi"}]):
        pass

    assert acquired == ["key-A", "key-B"]


@pytest.mark.asyncio
async def test_openai_provider_without_pool_is_unchanged(monkeypatch):
    """When no pool is provided, provider reuses a single client per init."""
    from src.providers.openai_compat import OpenAICompatProvider

    client_init_count = [0]

    class _FakeStream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

    class _FakeCompletions:
        async def create(self, **kwargs):
            return _FakeStream()

    class _FakeChat:
        def __init__(self):
            self.completions = _FakeCompletions()

    class _FakeClient:
        def __init__(self, api_key, **kwargs):
            client_init_count[0] += 1
            self.chat = _FakeChat()

    monkeypatch.setattr("src.providers.openai_compat.AsyncOpenAI", _FakeClient)

    provider = OpenAICompatProvider(
        api_key="sk-only",
        base_url="https://example.test/v1",
        model="test-model",
    )
    async for _ in provider.stream_message(messages=[{"role": "user", "content": "hi"}]):
        pass
    async for _ in provider.stream_message(messages=[{"role": "user", "content": "hi"}]):
        pass

    # Only one client was built (at __init__)
    assert client_init_count[0] == 1


@pytest.mark.asyncio
async def test_openai_provider_marks_429_on_http_error(monkeypatch):
    """A failing request with status 429 should notify the pool."""
    from src.providers.credential_pool import CredentialPool
    from src.providers.openai_compat import OpenAICompatProvider

    pool = CredentialPool(keys=["k1", "k2"], strategy="round_robin", cooldown_seconds=60)

    class _Error(Exception):
        def __init__(self):
            self.status_code = 429

    class _FakeCompletions:
        async def create(self, **kwargs):
            raise _Error()

    class _FakeChat:
        def __init__(self):
            self.completions = _FakeCompletions()

    class _FakeClient:
        def __init__(self, api_key, **kwargs):
            self._api_key = api_key
            self.chat = _FakeChat()

    monkeypatch.setattr("src.providers.openai_compat.AsyncOpenAI", _FakeClient)

    provider = OpenAICompatProvider(
        api_key="fallback",
        base_url="https://example.test/v1",
        model="test-model",
        credential_pool=pool,
    )

    with pytest.raises(_Error):
        async for _ in provider.stream_message(messages=[{"role": "user", "content": "hi"}]):
            pass

    # k1 should be cooling now; next acquire returns k2
    assert pool.acquire() == "k2"


def test_deepseek_provider_accepts_credential_pool():
    """Smoke test: DeepSeekProvider forwards credential_pool to the base."""
    from src.providers.credential_pool import CredentialPool
    from src.providers.deepseek import DeepSeekProvider

    pool = CredentialPool(keys=["k1", "k2"], strategy="round_robin")
    provider = DeepSeekProvider(
        api_key="fallback",
        base_url="https://example.test/v1",
        model="deepseek-chat",
        credential_pool=pool,
    )
    assert provider._credential_pool is pool


def test_anthropic_provider_accepts_credential_pool():
    """Smoke test: AnthropicCompatProvider stores credential_pool."""
    from src.providers.anthropic_compat import AnthropicCompatProvider
    from src.providers.credential_pool import CredentialPool

    pool = CredentialPool(keys=["k1", "k2"], strategy="round_robin")
    provider = AnthropicCompatProvider(
        api_key="fallback",
        base_url="https://example.test",
        model="claude-sonnet-4-5",
        credential_pool=pool,
    )
    assert provider._credential_pool is pool


@pytest.mark.asyncio
async def test_anthropic_provider_without_pool_builds_client_eagerly(monkeypatch):
    """No pool → client is created at __init__ and reused."""
    from src.providers.anthropic_compat import AnthropicCompatProvider

    init_count = [0]

    class _FakeClient:
        def __init__(self, api_key, **kwargs):
            init_count[0] += 1

    monkeypatch.setattr("src.providers.anthropic_compat.AsyncAnthropic", _FakeClient)

    provider = AnthropicCompatProvider(
        api_key="sk-only",
        base_url="https://example.test",
        model="claude-sonnet-4-5",
    )
    assert init_count[0] == 1
    assert provider._client is not None
    assert provider._credential_pool is None


def test_resolver_builds_pool_when_keys_list_present():
    from src.core.config import AgentConfig, ProviderSection
    from src.providers.resolver import resolve_provider

    cfg = AgentConfig(
        provider=ProviderSection(
            name="openai_compat",
            keys=["sk-a", "sk-b", "sk-c"],
            strategy="fill_first",
            cooldown_seconds=30,
            base_url="https://example.test/v1",
            model="test-model",
        )
    )
    provider = resolve_provider(cfg)
    assert provider._credential_pool is not None
    assert provider._credential_pool.list_keys() == ["sk-a", "sk-b", "sk-c"]


def test_resolver_single_key_path_unchanged():
    from src.core.config import AgentConfig, ProviderSection
    from src.providers.resolver import resolve_provider

    cfg = AgentConfig(
        provider=ProviderSection(
            name="openai_compat",
            api_key="sk-only",
            base_url="https://example.test/v1",
            model="test-model",
        )
    )
    provider = resolve_provider(cfg)
    assert provider._credential_pool is None


def test_resolver_api_key_as_single_item_list_does_not_build_pool():
    """If keys: [one] is given, effective_keys returns 1 entry → no pool."""
    from src.core.config import AgentConfig, ProviderSection
    from src.providers.resolver import resolve_provider

    cfg = AgentConfig(
        provider=ProviderSection(
            name="openai_compat",
            keys=["sk-solo"],
            base_url="https://example.test/v1",
            model="test-model",
        )
    )
    provider = resolve_provider(cfg)
    # Only 1 key: no pool needed, fall through to single-key path
    assert provider._credential_pool is None
