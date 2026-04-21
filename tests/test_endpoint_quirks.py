"""Tests for endpoint quirk detection."""

from __future__ import annotations

from src.providers.endpoint_quirks import detect_quirks, model_forbids_sampling


def test_official_anthropic_has_no_quirks() -> None:
    q = detect_quirks("https://api.anthropic.com")
    assert q.strip_thinking_signature is False
    assert q.forbids_zero_temperature is False


def test_minimax_anthropic_strips_signature_and_clamps_temperature() -> None:
    q = detect_quirks("https://api.minimaxi.com/anthropic")
    assert q.strip_thinking_signature is True
    assert q.forbids_zero_temperature is True
    assert q.min_temperature > 0


def test_generic_third_party_anthropic_strips_signature() -> None:
    q = detect_quirks("https://some-proxy.example.com/anthropic")
    assert q.strip_thinking_signature is True
    # But we don't assume zero-temp is forbidden without evidence.
    assert q.forbids_zero_temperature is False


def test_deepseek_caps_max_tokens() -> None:
    q = detect_quirks("https://api.deepseek.com/v1")
    assert q.max_tokens_cap == 8192
    assert q.strip_cache_control is True


def test_openai_native_has_no_max_tokens_cap() -> None:
    q = detect_quirks("https://api.openai.com/v1")
    assert q.max_tokens_cap is None
    assert q.strip_cache_control is False


def test_generic_openai_compat_strips_cache_control() -> None:
    q = detect_quirks("https://dashscope.aliyuncs.com/compatible-mode/v1")
    # Not Anthropic, not OpenAI native — treat as generic compat gateway.
    assert q.strip_cache_control is True


def test_empty_base_url_returns_default_quirks() -> None:
    q = detect_quirks("")
    assert q.strip_thinking_signature is False
    assert q.forbids_zero_temperature is False
    assert q.max_tokens_cap is None


def test_model_forbids_sampling_detects_o_series() -> None:
    assert model_forbids_sampling("o1-preview") is True
    assert model_forbids_sampling("o1-mini") is True
    assert model_forbids_sampling("o3") is True
    assert model_forbids_sampling("o4-mini") is True


def test_model_forbids_sampling_false_for_regular_models() -> None:
    assert model_forbids_sampling("gpt-4o-mini") is False
    assert model_forbids_sampling("deepseek-chat") is False
    assert model_forbids_sampling("claude-sonnet-4-5") is False
    assert model_forbids_sampling("") is False
