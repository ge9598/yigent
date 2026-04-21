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
async def test_env_text_prefixed_onto_latest_user_message(plan_mode) -> None:
    """Unit 9 — env context is prefixed onto the latest user message
    (not added as a separate Zone-3 system message that grows messages[]
    every turn). Per ARCHITECTURE.md §I-bis _inject_env spec."""
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
    user_msgs = [m for m in msgs if m.get("role") == "user"]
    assert len(user_msgs) == 1
    assert "branch: master" in user_msgs[0]["content"]
    assert "hi" in user_msgs[0]["content"]
    # System messages should NOT contain the env text.
    sys_text = "\n".join(m["content"] for m in msgs if m["role"] == "system")
    assert "branch: master" not in sys_text


@pytest.mark.asyncio
async def test_env_falls_back_to_system_when_no_user_msg(plan_mode) -> None:
    """If the conversation has no user message yet, env falls back to a
    standalone system message (back-compat)."""
    inj = MagicMock(spec=EnvironmentInjector)

    async def get_context(task_type: str) -> str:
        return "branch: master"

    inj.get_context = get_context

    asm = ContextAssembler(system_prompt="SYS", plan_mode=plan_mode)
    msgs = await asm.assemble(
        tool_registry=ToolRegistry(), env_injector=inj,
        conversation=[],  # no user msg yet
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


# ---------------------------------------------------------------------------
# Unit 9 — three-tier thresholds (warn / compress / hard) and budget_warning
# ---------------------------------------------------------------------------

def test_three_tier_threshold_offsets(plan_mode):
    """Numbers come from ARCHITECTURE.md §I — must match exactly."""
    asm = ContextAssembler(
        system_prompt="SYS", plan_mode=plan_mode,
        model_context_window=200_000,
    )
    assert asm.warn_threshold == 200_000 - 40_000
    assert asm.compress_threshold == 200_000 - 33_000
    assert asm.hard_cutoff == 200_000 - 23_000


def test_three_tier_thresholds_ordered(plan_mode):
    """warn < compress < hard must hold so the tiers fire in the right order."""
    asm = ContextAssembler(
        system_prompt="SYS", plan_mode=plan_mode,
        model_context_window=128_000,
    )
    assert asm.warn_threshold < asm.compress_threshold < asm.hard_cutoff


@pytest.mark.asyncio
async def test_budget_warning_hook_fires_when_warn_threshold_crossed(
    plan_mode, env_injector,
):
    """Crossing warn_threshold but not compress_threshold should fire the
    hook without invoking the compression engine."""
    from src.safety.hook_system import HookSystem
    hooks = HookSystem()
    fired: list[dict] = []

    async def _record(**data):
        fired.append(data)

    hooks.register("budget_warning", _record)

    fake = _ShrinkingEngine()
    # Pick a window so warn=200, compress=300, hard=500 — a payload of
    # ~210 tokens crosses warn but stays under compress.
    asm = ContextAssembler(
        system_prompt="SYS", plan_mode=plan_mode,
        compression_engine=fake,
        model_context_window=40_200,  # warn=200, compress=7200, hard=17200
        output_reserve=0, safety_buffer=0,
        hook_system=hooks,
    )
    payload = "word " * 200  # ~210 tokens
    conv = [{"role": "user", "content": payload}]
    await asm.assemble(
        tool_registry=ToolRegistry(), env_injector=env_injector,
        conversation=conv, task_type="coding",
    )
    assert len(fired) >= 1
    ev = fired[0]
    assert "used_tokens" in ev
    assert "warn_threshold" in ev
    assert "compress_threshold" in ev
    assert "hard_cutoff" in ev


@pytest.mark.asyncio
async def test_budget_warning_does_not_fire_repeatedly_in_same_turn(
    plan_mode, env_injector,
):
    """Once warn fires for a turn, it shouldn't fire again from the same
    assemble() call."""
    from src.safety.hook_system import HookSystem
    hooks = HookSystem()
    fired: list[dict] = []

    async def _record(**data):
        fired.append(data)

    hooks.register("budget_warning", _record)

    fake = _ShrinkingEngine()
    asm = ContextAssembler(
        system_prompt="SYS", plan_mode=plan_mode,
        compression_engine=fake,
        model_context_window=40_200,
        output_reserve=0, safety_buffer=0,
        hook_system=hooks,
    )
    payload = "word " * 200
    conv = [{"role": "user", "content": payload}]
    await asm.assemble(
        tool_registry=ToolRegistry(), env_injector=env_injector,
        conversation=conv, task_type="coding",
    )
    # Single assemble() must not fire the hook more than once.
    assert len(fired) == 1
