"""Tests for ContextAssembler — 5-zone composition + compression triggering."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.context.assembler import ContextAssembler
from src.context.engine import CompressionEngine, estimate_tokens
from src.core.env_injector import EnvironmentInjector
from src.core.plan_mode import PlanMode
from src.core.types import PermissionLevel, ToolDefinition, ToolSchema
from src.tools.registry import ToolRegistry


def _make_registry_with_deferred() -> ToolRegistry:
    reg = ToolRegistry()

    async def noop(**kwargs) -> str:
        return "ok"

    reg._tools.clear()
    reg._activated.clear()
    reg.register(ToolDefinition(
        name="active_tool",
        description="active",
        handler=noop,
        schema=ToolSchema(name="active_tool", description="active",
                          parameters={"type": "object"},
                          permission_level=PermissionLevel.READ_ONLY,
                          deferred=False),
    ))
    reg.register(ToolDefinition(
        name="hidden_tool",
        description="hidden",
        handler=noop,
        schema=ToolSchema(name="hidden_tool", description="hidden",
                          parameters={"type": "object"},
                          permission_level=PermissionLevel.READ_ONLY,
                          deferred=True),
    ))
    return reg


@pytest.fixture
def plan_mode(tmp_path: Path) -> PlanMode:
    return PlanMode(save_dir=str(tmp_path / "plans"))


@pytest.fixture
def env_injector() -> EnvironmentInjector:
    inj = MagicMock(spec=EnvironmentInjector)

    async def get_context(task_type: str) -> str:
        return ""

    inj.get_context = get_context
    return inj


# ---------------------------------------------------------------------------
# Zone composition
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_zone1_static_system_prompt_first(plan_mode, env_injector) -> None:
    asm = ContextAssembler(system_prompt="SYS", plan_mode=plan_mode)
    msgs = await asm.assemble(
        tool_registry=ToolRegistry(),
        env_injector=env_injector,
        conversation=[{"role": "user", "content": "hi"}],
        task_type="coding",
    )
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == "SYS"


@pytest.mark.asyncio
async def test_deferred_tools_hint_added(plan_mode, env_injector) -> None:
    reg = _make_registry_with_deferred()
    asm = ContextAssembler(system_prompt="SYS", plan_mode=plan_mode)
    msgs = await asm.assemble(
        tool_registry=reg, env_injector=env_injector,
        conversation=[{"role": "user", "content": "hi"}],
        task_type="coding",
    )
    sys_msgs = [m for m in msgs if m["role"] == "system"]
    joined = "\n".join(m["content"] for m in sys_msgs)
    assert "hidden_tool" in joined
    assert "active_tool" not in joined  # not deferred → no hint


@pytest.mark.asyncio
async def test_no_deferred_hint_when_all_active(plan_mode, env_injector) -> None:
    asm = ContextAssembler(system_prompt="SYS", plan_mode=plan_mode)
    msgs = await asm.assemble(
        tool_registry=ToolRegistry(),
        env_injector=env_injector,
        conversation=[{"role": "user", "content": "hi"}],
        task_type="coding",
    )
    # Only the static system msg + the conversation user msg.
    assert sum(1 for m in msgs if m["role"] == "system") == 1


@pytest.mark.asyncio
async def test_plan_mode_notice_in_zone3(plan_mode, env_injector) -> None:
    plan_mode.enter("test-session")
    asm = ContextAssembler(system_prompt="SYS", plan_mode=plan_mode)
    msgs = await asm.assemble(
        tool_registry=ToolRegistry(),
        env_injector=env_injector,
        conversation=[{"role": "user", "content": "hi"}],
        task_type="coding",
    )
    sys_text = "\n".join(m["content"] for m in msgs if m["role"] == "system")
    assert "PLAN MODE IS ACTIVE" in sys_text


@pytest.mark.asyncio
async def test_env_text_injected_in_zone3(plan_mode) -> None:
    inj = MagicMock(spec=EnvironmentInjector)

    async def get_context(task_type: str) -> str:
        return "branch: master"

    inj.get_context = get_context

    asm = ContextAssembler(system_prompt="SYS", plan_mode=plan_mode)
    msgs = await asm.assemble(
        tool_registry=ToolRegistry(), env_injector=inj,
        conversation=[{"role": "user", "content": "hi"}],
        task_type="coding",
    )
    sys_text = "\n".join(m["content"] for m in msgs if m["role"] == "system")
    assert "branch: master" in sys_text


@pytest.mark.asyncio
async def test_memory_index_injected_into_zone3(plan_mode, env_injector, tmp_path) -> None:
    """When a memory store has a MEMORY.md index, it surfaces in Zone 3."""
    from src.memory.markdown_store import MarkdownMemoryStore

    store = MarkdownMemoryStore(tmp_path / "mem")
    store.write_topic("debugging", "how we traced the FTS bug")
    store.record_index_entry("debugging", "Debugging", "FTS trigger insight")

    asm = ContextAssembler(
        system_prompt="SYS", plan_mode=plan_mode, memory_store=store,
    )
    msgs = await asm.assemble(
        tool_registry=ToolRegistry(), env_injector=env_injector,
        conversation=[{"role": "user", "content": "hi"}],
        task_type="coding",
    )
    sys_text = "\n".join(m["content"] for m in msgs if m["role"] == "system")
    assert "[Memory index" in sys_text
    assert "debugging.md" in sys_text


@pytest.mark.asyncio
async def test_no_memory_section_when_store_empty(plan_mode, env_injector, tmp_path) -> None:
    from src.memory.markdown_store import MarkdownMemoryStore

    store = MarkdownMemoryStore(tmp_path / "mem")  # never written
    asm = ContextAssembler(
        system_prompt="SYS", plan_mode=plan_mode, memory_store=store,
    )
    msgs = await asm.assemble(
        tool_registry=ToolRegistry(), env_injector=env_injector,
        conversation=[{"role": "user", "content": "hi"}],
        task_type="coding",
    )
    sys_text = "\n".join(m["content"] for m in msgs if m["role"] == "system")
    assert "[Memory index" not in sys_text


@pytest.mark.asyncio
async def test_conversation_appended_after_system_zones(plan_mode, env_injector) -> None:
    asm = ContextAssembler(system_prompt="SYS", plan_mode=plan_mode)
    msgs = await asm.assemble(
        tool_registry=ToolRegistry(),
        env_injector=env_injector,
        conversation=[
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ],
        task_type="coding",
    )
    # Last 2 messages are the conversation.
    assert msgs[-2]["role"] == "user"
    assert msgs[-1]["role"] == "assistant"


# ---------------------------------------------------------------------------
# Cache stability
# ---------------------------------------------------------------------------

def test_static_zone_hash_stable_across_assemble_calls(plan_mode) -> None:
    asm = ContextAssembler(system_prompt="SYS", plan_mode=plan_mode)
    h1 = asm.cache.prefix_hash
    h2 = asm.cache.prefix_hash
    assert h1 == h2


# ---------------------------------------------------------------------------
# Compression triggered when over budget
# ---------------------------------------------------------------------------

class _ShrinkingEngine:
    """Test-only engine: returns a constant-size short list to verify wiring."""
    def __init__(self) -> None:
        self.compress_called_with_target: int | None = None

    async def compress(self, conversation, target_tokens, on_layer=None):
        self.compress_called_with_target = target_tokens
        return [{"role": "user", "content": "compressed"}]


@pytest.mark.asyncio
async def test_compression_triggered_when_over_budget(plan_mode, env_injector) -> None:
    fake = _ShrinkingEngine()
    asm = ContextAssembler(
        system_prompt="SYS", plan_mode=plan_mode,
        compression_engine=fake,
        model_context_window=1000,         # tiny window forces compression
        output_reserve=200, safety_buffer=100,
    )
    big = [{"role": "user", "content": "x" * 5000}]
    msgs = await asm.assemble(
        tool_registry=ToolRegistry(), env_injector=env_injector,
        conversation=big, task_type="coding",
    )
    assert fake.compress_called_with_target is not None
    # The conversation portion is the compressed output.
    user_msgs = [m for m in msgs if m["role"] == "user"]
    assert any(m["content"] == "compressed" for m in user_msgs)


@pytest.mark.asyncio
async def test_compression_skipped_when_under_budget(plan_mode, env_injector) -> None:
    fake = _ShrinkingEngine()
    asm = ContextAssembler(
        system_prompt="SYS", plan_mode=plan_mode,
        compression_engine=fake,
    )
    small = [{"role": "user", "content": "tiny"}]
    await asm.assemble(
        tool_registry=ToolRegistry(), env_injector=env_injector,
        conversation=small, task_type="coding",
    )
    assert fake.compress_called_with_target is None
