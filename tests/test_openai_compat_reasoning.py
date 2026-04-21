"""Tests for OpenAICompatProvider reasoning handling and endpoint quirks."""

from __future__ import annotations

import pytest

from src.providers.openai_compat import (
    OpenAICompatProvider,
    _strip_reasoning_for_request,
)


# ---------------------------------------------------------------------------
# Fake OpenAI SDK stream — minimal surface to drive _stream_message
# ---------------------------------------------------------------------------

class _D:  # dynamic bag
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _delta(content: str = "", reasoning_content: str | None = None,
           tool_calls=None) -> _D:
    return _D(
        content=content,
        reasoning_content=reasoning_content,
        tool_calls=tool_calls,
    )


def _chunk(delta_obj: _D, finish_reason: str | None = None) -> _D:
    return _D(choices=[_D(delta=delta_obj, finish_reason=finish_reason)])


class _FakeStream:
    def __init__(self, chunks: list):
        self._chunks = chunks

    def __aiter__(self):
        return self._agen()

    async def _agen(self):
        for c in self._chunks:
            yield c


class _FakeCompletions:
    def __init__(self, chunks: list):
        self._chunks = chunks
        self.last_kwargs: dict | None = None

    async def create(self, **kwargs):
        self.last_kwargs = kwargs
        return _FakeStream(self._chunks)


class _FakeChat:
    def __init__(self, chunks: list):
        self.completions = _FakeCompletions(chunks)


class _FakeClient:
    def __init__(self, chunks: list):
        self.chat = _FakeChat(chunks)


def _make_provider(chunks: list, base_url: str = "https://api.openai.com/v1",
                   model: str = "gpt-4o-mini") -> OpenAICompatProvider:
    p = OpenAICompatProvider(api_key="test", base_url=base_url, model=model)
    p._client = _FakeClient(chunks)  # type: ignore[assignment]
    return p


# ---------------------------------------------------------------------------
# Form #2: delta.reasoning_content (DeepSeek-R1 / o1 style)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_reasoning_content_emits_live_delta_fragments() -> None:
    """Each reasoning_content delta emits a reasoning_delta chunk."""
    chunks = [
        _chunk(_delta(reasoning_content="alpha ")),
        _chunk(_delta(reasoning_content="beta")),
        _chunk(_delta(), finish_reason="stop"),
    ]
    p = _make_provider(chunks)
    events = [e async for e in p.stream_message(
        [{"role": "user", "content": "q"}],
    )]
    deltas = [e for e in events if e.type == "reasoning_delta"]
    assert [d.data for d in deltas] == ["alpha ", "beta"]


@pytest.mark.asyncio
async def test_think_tags_emit_live_reasoning_deltas() -> None:
    """<think> content split by FSM emits reasoning_delta chunks."""
    chunks = [
        _chunk(_delta(content="<think>x</think>")),
        _chunk(_delta(content="visible")),
        _chunk(_delta(), finish_reason="stop"),
    ]
    p = _make_provider(chunks)
    events = [e async for e in p.stream_message(
        [{"role": "user", "content": "q"}],
    )]
    deltas = [e for e in events if e.type == "reasoning_delta"]
    assert deltas
    assert "".join(d.data for d in deltas) == "x"


@pytest.mark.asyncio
async def test_reasoning_content_field_accumulated_into_reasoning_chunk() -> None:
    chunks = [
        _chunk(_delta(reasoning_content="Let me think. ")),
        _chunk(_delta(reasoning_content="2+2=4.")),
        _chunk(_delta(content="4")),
        _chunk(_delta(), finish_reason="stop"),
    ]
    p = _make_provider(chunks)
    events = [e async for e in p.stream_message(
        [{"role": "user", "content": "2+2?"}],
    )]
    reasoning = [e for e in events if e.type == "reasoning"]
    assert len(reasoning) == 1
    assert reasoning[0].data["text"] == "Let me think. 2+2=4."
    assert reasoning[0].data["details"] is None  # OpenAI protocol → no signed details
    tokens = [e.data for e in events if e.type == "token"]
    assert "".join(tokens) == "4"


# ---------------------------------------------------------------------------
# Form #3: <think> tags inline in delta.content (MiniMax M2 /v1 / Qwen QwQ)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_think_tags_in_content_split_into_reasoning_and_text() -> None:
    chunks = [
        _chunk(_delta(content="<think>internal reasoning</think>")),
        _chunk(_delta(content="visible answer")),
        _chunk(_delta(), finish_reason="stop"),
    ]
    p = _make_provider(chunks)
    events = [e async for e in p.stream_message(
        [{"role": "user", "content": "hi"}],
    )]
    tokens = "".join(e.data for e in events if e.type == "token")
    assert tokens == "visible answer"
    reasoning = [e for e in events if e.type == "reasoning"]
    assert len(reasoning) == 1
    assert reasoning[0].data["text"] == "internal reasoning"


@pytest.mark.asyncio
async def test_think_tag_split_across_chunks_still_reconstructed() -> None:
    """Tag boundary falls mid-chunk; FSM state crosses chunk boundary."""
    chunks = [
        _chunk(_delta(content="pre<thi")),
        _chunk(_delta(content="nk>thought</think>final")),
        _chunk(_delta(), finish_reason="stop"),
    ]
    p = _make_provider(chunks)
    events = [e async for e in p.stream_message(
        [{"role": "user", "content": "hi"}],
    )]
    tokens = "".join(e.data for e in events if e.type == "token")
    assert tokens == "prefinal"
    reasoning = [e for e in events if e.type == "reasoning"]
    assert reasoning[0].data["text"] == "thought"


# ---------------------------------------------------------------------------
# No reasoning → no reasoning chunk
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_no_reasoning_means_no_reasoning_chunk() -> None:
    chunks = [
        _chunk(_delta(content="plain")),
        _chunk(_delta(), finish_reason="stop"),
    ]
    p = _make_provider(chunks)
    events = [e async for e in p.stream_message(
        [{"role": "user", "content": "hi"}],
    )]
    assert not any(e.type == "reasoning" for e in events)


# ---------------------------------------------------------------------------
# Endpoint quirks: DeepSeek max_tokens cap, o1-style no sampling
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_deepseek_caps_max_tokens_at_8192() -> None:
    chunks = [_chunk(_delta(), finish_reason="stop")]
    p = _make_provider(chunks, base_url="https://api.deepseek.com/v1",
                       model="deepseek-chat")
    _ = [e async for e in p.stream_message([{"role": "user", "content": "hi"}])]
    kwargs = p._client.chat.completions.last_kwargs  # type: ignore[attr-defined]
    assert kwargs["max_tokens"] <= 8192


@pytest.mark.asyncio
async def test_o1_model_omits_temperature() -> None:
    chunks = [_chunk(_delta(), finish_reason="stop")]
    p = _make_provider(chunks, base_url="https://api.openai.com/v1",
                       model="o1-preview")
    _ = [e async for e in p.stream_message([{"role": "user", "content": "hi"}])]
    kwargs = p._client.chat.completions.last_kwargs  # type: ignore[attr-defined]
    assert "temperature" not in kwargs


@pytest.mark.asyncio
async def test_non_o1_model_keeps_temperature() -> None:
    chunks = [_chunk(_delta(), finish_reason="stop")]
    p = _make_provider(chunks, model="gpt-4o-mini")
    _ = [e async for e in p.stream_message(
        [{"role": "user", "content": "hi"}], temperature=0.7,
    )]
    kwargs = p._client.chat.completions.last_kwargs  # type: ignore[attr-defined]
    assert kwargs["temperature"] == 0.7


# ---------------------------------------------------------------------------
# reasoning_* fields stripped from outgoing request
# ---------------------------------------------------------------------------

def test_strip_reasoning_removes_both_fields() -> None:
    messages = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "hello",
            "reasoning_text": "was thinking",
            "reasoning_details": [{"type": "thinking", "thinking": "x", "signature": "s"}],
        },
    ]
    out = _strip_reasoning_for_request(messages)  # type: ignore[arg-type]
    assert out[0] == {"role": "user", "content": "hi"}
    assert out[1] == {"role": "assistant", "content": "hello"}


def test_strip_reasoning_leaves_clean_messages_untouched() -> None:
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    out = _strip_reasoning_for_request(messages)  # type: ignore[arg-type]
    assert out == messages


@pytest.mark.asyncio
async def test_stream_does_not_send_reasoning_fields_to_provider() -> None:
    chunks = [_chunk(_delta(), finish_reason="stop")]
    p = _make_provider(chunks)
    messages = [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": "a",
            "reasoning_text": "secret thought",
            "reasoning_details": [{"type": "thinking", "thinking": "x"}],
        },
        {"role": "user", "content": "next"},
    ]
    _ = [e async for e in p.stream_message(messages)]  # type: ignore[arg-type]
    sent = p._client.chat.completions.last_kwargs["messages"]  # type: ignore[attr-defined]
    assistant = [m for m in sent if m["role"] == "assistant"][0]
    assert "reasoning_text" not in assistant
    assert "reasoning_details" not in assistant
