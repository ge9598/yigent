"""Tests for the 5-layer CompressionEngine."""

from __future__ import annotations

import pytest

from src.context.circuit_breaker import CircuitBreaker
from src.context.engine import CompressionEngine, estimate_tokens


# ---------------------------------------------------------------------------
# estimate_tokens
# ---------------------------------------------------------------------------

def test_estimate_tokens_counts_string() -> None:
    n = estimate_tokens("hello world")
    assert n > 0


def test_estimate_tokens_counts_messages() -> None:
    msgs = [
        {"role": "user", "content": "the quick brown fox jumps over the lazy dog"},
        {"role": "assistant", "content": "indeed it does"},
    ]
    n = estimate_tokens(msgs)
    # Two msgs = 8 envelope tokens + ~13 content tokens.
    assert n > 15


# ---------------------------------------------------------------------------
# Layer 1 — truncate large tool results
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_layer1_truncates_oversized_tool_result() -> None:
    engine = CompressionEngine(tool_result_cap=100)
    msgs = [
        {"role": "user", "content": "go"},
        {"role": "tool", "tool_call_id": "x", "name": "read_file",
         "content": "X" * 5000},
    ]
    out = engine._layer1_truncate_tool_results(msgs)
    assert len(out[1]["content"]) < 200
    assert "[truncated]" in out[1]["content"]


@pytest.mark.asyncio
async def test_layer1_leaves_short_tool_results_alone() -> None:
    engine = CompressionEngine(tool_result_cap=100)
    msgs = [{"role": "tool", "tool_call_id": "x", "name": "read_file",
             "content": "small"}]
    out = engine._layer1_truncate_tool_results(msgs)
    assert out[0]["content"] == "small"


# ---------------------------------------------------------------------------
# Layer 2 — dedup file reads
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_layer2_dedups_identical_reads() -> None:
    engine = CompressionEngine()
    msgs = [
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "1", "type": "function",
                         "function": {"name": "read_file",
                                      "arguments": '{"path": "foo"}'}}]},
        {"role": "tool", "tool_call_id": "1", "name": "read_file",
         "content": "FILE BODY"},
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "2", "type": "function",
                         "function": {"name": "read_file",
                                      "arguments": '{"path": "foo"}'}}]},
        {"role": "tool", "tool_call_id": "2", "name": "read_file",
         "content": "FILE BODY"},
    ]
    out = engine._layer2_dedup_file_reads(msgs)
    # The earlier one should become a stub; the latest stays.
    tool_msgs = [m for m in out if m.get("role") == "tool"]
    assert tool_msgs[0]["content"] == "[duplicate file read elided]"
    assert tool_msgs[1]["content"] == "FILE BODY"


@pytest.mark.asyncio
async def test_layer2_keeps_distinct_files() -> None:
    engine = CompressionEngine()
    msgs = [
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "1", "type": "function",
                         "function": {"name": "read_file",
                                      "arguments": '{"path": "foo"}'}}]},
        {"role": "tool", "tool_call_id": "1", "name": "read_file",
         "content": "FOO"},
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "2", "type": "function",
                         "function": {"name": "read_file",
                                      "arguments": '{"path": "bar"}'}}]},
        {"role": "tool", "tool_call_id": "2", "name": "read_file",
         "content": "BAR"},
    ]
    out = engine._layer2_dedup_file_reads(msgs)
    tool_msgs = [m for m in out if m.get("role") == "tool"]
    assert tool_msgs[0]["content"] == "FOO"
    assert tool_msgs[1]["content"] == "BAR"


# ---------------------------------------------------------------------------
# Layer 5 — hard truncate
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_layer5_keeps_system_msgs_and_last_n() -> None:
    engine = CompressionEngine(layer5_keep_turns=2)
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "1"},
        {"role": "assistant", "content": "1a"},
        {"role": "user", "content": "2"},
        {"role": "assistant", "content": "2a"},
        {"role": "user", "content": "3"},
    ]
    out = engine._layer5_hard_truncate(msgs)
    assert out[0]["role"] == "system"
    assert len(out) == 3  # 1 system + 2 last
    assert out[-1]["content"] == "3"


# ---------------------------------------------------------------------------
# compress() orchestration — short-circuits when already fits
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_compress_returns_unchanged_if_fits_budget() -> None:
    engine = CompressionEngine()
    msgs = [{"role": "user", "content": "hi"}]
    out = await engine.compress(msgs, target_tokens=100_000)
    assert out == msgs


# ---------------------------------------------------------------------------
# circuit breaker integration — layer 3 skipped when open
# ---------------------------------------------------------------------------

class _FakeAuxProvider:
    """Counts how many times stream_message was called."""
    def __init__(self, fail: bool = False) -> None:
        self.calls = 0
        self.fail = fail

    async def stream_message(self, messages, **kw):
        self.calls += 1
        if self.fail:
            raise RuntimeError("aux LLM down")
        # yield one token then done
        from src.core.types import StreamChunk
        yield StreamChunk(type="token", data="SUMMARY")
        yield StreamChunk(type="done", data="stop")


@pytest.mark.asyncio
async def test_layer3_breaker_opens_after_failures() -> None:
    aux = _FakeAuxProvider(fail=True)
    engine = CompressionEngine(
        auxiliary_provider=aux,
        layer3_breaker=CircuitBreaker(threshold=2),
        layer4_breaker=CircuitBreaker(threshold=2),
    )
    # Force compression: build a conversation bigger than target.
    big_msgs = [
        {"role": "user", "content": "x" * 1000} for _ in range(20)
    ]
    # First compress: layer 1+2 don't shrink (no tool msgs). Layer 3 fails.
    await engine.compress(big_msgs, target_tokens=10)
    assert engine.layer3_breaker.failures == 1
    # Second: layer 3 fails again → breaker opens.
    await engine.compress(big_msgs, target_tokens=10)
    assert engine.layer3_breaker.is_open
    # Third: layer 3 should be skipped (no new call attempts)
    calls_before = aux.calls
    await engine.compress(big_msgs, target_tokens=10)
    # layer4 still tries (independent breaker), so calls may grow. The check
    # is that layer3 was skipped — its breaker count didn't increment.
    assert engine.layer3_breaker.failures == 2  # didn't grow past threshold


@pytest.mark.asyncio
async def test_layer3_success_resets_breaker() -> None:
    aux = _FakeAuxProvider(fail=False)
    engine = CompressionEngine(
        auxiliary_provider=aux,
        layer3_breaker=CircuitBreaker(threshold=2),
    )
    engine.layer3_breaker.failures = 1
    big_msgs = [
        {"role": "user", "content": "alpha bravo charlie"},
        {"role": "assistant", "content": "ack"},
        {"role": "user", "content": "delta echo foxtrot"},
        {"role": "assistant", "content": "ack"},
        {"role": "user", "content": "golf hotel india"},
        {"role": "assistant", "content": "ack"},
        {"role": "user", "content": "juliet kilo lima"},
        {"role": "assistant", "content": "ack"},
    ]
    await engine.compress(big_msgs, target_tokens=20)
    assert engine.layer3_breaker.failures == 0


# ---------------------------------------------------------------------------
# Unit 1 — pre_compression / post_compression hooks
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_compression_hooks_fire_when_compression_runs() -> None:
    from src.safety.hook_system import HookSystem
    hooks = HookSystem()
    pre_events: list[dict] = []
    post_events: list[dict] = []

    async def _pre(**data):
        pre_events.append(data)

    async def _post(**data):
        post_events.append(data)

    hooks.register("pre_compression", _pre)
    hooks.register("post_compression", _post)

    engine = CompressionEngine(tool_result_cap=100, hook_system=hooks)
    msgs = [
        {"role": "user", "content": "go"},
        {"role": "tool", "tool_call_id": "x", "name": "read_file",
         "content": "X" * 5000},
    ]
    await engine.compress(msgs, target_tokens=50)
    assert len(pre_events) == 1
    assert pre_events[0]["before_tokens"] > 50
    assert pre_events[0]["target_tokens"] == 50
    assert len(post_events) == 1
    assert 1 in post_events[0]["layers_run"]
    assert post_events[0]["after_tokens"] <= post_events[0]["before_tokens"]


@pytest.mark.asyncio
async def test_compression_hooks_skip_when_no_compression_needed() -> None:
    from src.safety.hook_system import HookSystem
    hooks = HookSystem()
    pre_events: list[dict] = []

    async def _pre(**data):
        pre_events.append(data)

    hooks.register("pre_compression", _pre)

    engine = CompressionEngine(hook_system=hooks)
    msgs = [{"role": "user", "content": "short"}]
    await engine.compress(msgs, target_tokens=10_000)
    assert pre_events == []  # no hook fires when target already satisfied
