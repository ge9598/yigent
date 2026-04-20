"""Tests for PermissionGate — 5-layer chain + plan mode authoritative."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.core.config import AgentConfig
from src.core.plan_mode import PlanMode
from src.core.types import (
    PermissionDecision, PermissionLevel, ToolCall, ToolContext,
    ToolDefinition, ToolSchema, ValidateResult,
)
from src.safety.hook_system import HookSystem
from src.safety.permission_gate import PermissionGate
from src.tools.registry import ToolRegistry


def _make_registry(*defs: ToolDefinition) -> ToolRegistry:
    reg = ToolRegistry()
    reg._tools.clear()
    reg._activated.clear()
    for d in defs:
        reg.register(d)
    return reg


async def _noop_handler(**kwargs) -> str:
    return "ok"


def _def(name: str, level: PermissionLevel = PermissionLevel.READ_ONLY,
         validate=None) -> ToolDefinition:
    return ToolDefinition(
        name=name, description=name, handler=_noop_handler,
        schema=ToolSchema(name=name, description=name,
                          parameters={"type": "object"},
                          permission_level=level),
        validate=validate,
    )


def _ctx(tmp_path: Path) -> ToolContext:
    plan_mode = PlanMode(save_dir=str(tmp_path / "plans"))
    return ToolContext(
        plan_mode=plan_mode, registry=ToolRegistry(),
        config=AgentConfig(), working_dir=tmp_path,
    )


async def _always_allow(tc: ToolCall) -> PermissionDecision:
    return PermissionDecision.ALLOW


async def _always_deny(tc: ToolCall) -> PermissionDecision:
    return PermissionDecision.BLOCK


# ---------------------------------------------------------------------------
# Layer 1 — schema validation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_unknown_tool_blocked(tmp_path) -> None:
    reg = _make_registry(_def("known"))
    gate = PermissionGate(reg, _ctx(tmp_path))
    tc = ToolCall(id="1", name="ghost", arguments={})
    decision = await gate.check(tc, _always_allow)
    assert decision == PermissionDecision.BLOCK
    assert "unknown tool" in gate.last_block_reason


# ---------------------------------------------------------------------------
# Layer 2 — tool self-check
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_tool_validate_deny_blocks(tmp_path) -> None:
    async def validate(**kw) -> ValidateResult:
        return ValidateResult(decision="deny", reason="bad path")

    reg = _make_registry(_def("read", validate=validate))
    gate = PermissionGate(reg, _ctx(tmp_path))
    tc = ToolCall(id="1", name="read", arguments={"path": "/etc/shadow"})
    decision = await gate.check(tc, _always_allow)
    assert decision == PermissionDecision.BLOCK
    assert "validation failed" in gate.last_block_reason
    assert "bad path" in gate.last_block_reason


@pytest.mark.asyncio
async def test_tool_validate_allow_allows(tmp_path) -> None:
    async def validate(**kw) -> ValidateResult:
        return ValidateResult(decision="allow")

    reg = _make_registry(_def("read", validate=validate))
    gate = PermissionGate(reg, _ctx(tmp_path))
    tc = ToolCall(id="1", name="read", arguments={})
    decision = await gate.check(tc, _always_allow)
    assert decision == PermissionDecision.ALLOW


@pytest.mark.asyncio
async def test_tool_validate_receives_ctx(tmp_path) -> None:
    seen = {}

    async def validate(*, ctx, **kw) -> ValidateResult:
        seen["ctx"] = ctx
        return ValidateResult(decision="allow")

    reg = _make_registry(_def("read", validate=validate))
    ctx = _ctx(tmp_path)
    gate = PermissionGate(reg, ctx)
    await gate.check(ToolCall(id="1", name="read", arguments={}), _always_allow)
    assert seen["ctx"] is ctx


@pytest.mark.asyncio
async def test_tool_validate_updated_input_rewrites_args(tmp_path) -> None:
    async def validate(*, ctx, path=None, **kw) -> ValidateResult:
        # Normalise the path — e.g. strip trailing slash.
        return ValidateResult(
            decision="allow",
            updated_input={"path": (path or "").rstrip("/")},
        )

    reg = _make_registry(_def("read", validate=validate))
    gate = PermissionGate(reg, _ctx(tmp_path))
    tc = ToolCall(id="1", name="read", arguments={"path": "/tmp/foo/"})
    await gate.check(tc, _always_allow)
    assert tc.arguments == {"path": "/tmp/foo"}


@pytest.mark.asyncio
async def test_tool_validate_ask_forces_user_prompt_even_for_readonly(tmp_path) -> None:
    """Layer 2's ask decision overrides READ_ONLY auto-allow — layer 5 prompts."""
    async def validate(**kw) -> ValidateResult:
        return ValidateResult(decision="ask", reason="borderline")

    prompted = {}

    async def cb(tc):
        prompted["yes"] = True
        return PermissionDecision.BLOCK

    reg = _make_registry(_def("read", PermissionLevel.READ_ONLY, validate=validate))
    gate = PermissionGate(reg, _ctx(tmp_path))
    decision = await gate.check(ToolCall(id="1", name="read", arguments={}), cb)
    assert prompted == {"yes": True}
    assert decision == PermissionDecision.BLOCK  # cb said block


@pytest.mark.asyncio
async def test_tool_validate_ask_overrides_yolo_mode(tmp_path) -> None:
    """Even in YOLO, a validator that asks still asks."""
    async def validate(**kw) -> ValidateResult:
        return ValidateResult(decision="ask", reason="sensitive write")

    prompted = {}

    async def cb(tc):
        prompted["yes"] = True
        return PermissionDecision.ALLOW

    reg = _make_registry(_def("write", PermissionLevel.WRITE, validate=validate))
    gate = PermissionGate(reg, _ctx(tmp_path), yolo_mode=True)
    await gate.check(ToolCall(id="1", name="write", arguments={}), cb)
    assert prompted == {"yes": True}


@pytest.mark.asyncio
async def test_tool_validate_ask_still_blocked_by_plan_mode(tmp_path) -> None:
    """Plan mode is authoritative — validator's ask can't bypass it."""
    async def validate(**kw) -> ValidateResult:
        return ValidateResult(decision="ask")

    reg = _make_registry(_def("write", PermissionLevel.WRITE, validate=validate))
    ctx = _ctx(tmp_path)
    ctx.plan_mode.enter("test")
    gate = PermissionGate(reg, ctx)
    decision = await gate.check(
        ToolCall(id="1", name="write", arguments={}), _always_allow,
    )
    assert decision == PermissionDecision.BLOCK
    assert "plan mode" in gate.last_block_reason


@pytest.mark.asyncio
async def test_tool_validate_returning_wrong_type_blocks(tmp_path) -> None:
    """Defensive: legacy str-returning validators now fail closed."""
    async def validate(**kw):
        return "bad path"  # old-style return — no longer accepted

    reg = _make_registry(_def("read", validate=validate))
    gate = PermissionGate(reg, _ctx(tmp_path))
    decision = await gate.check(
        ToolCall(id="1", name="read", arguments={}), _always_allow,
    )
    assert decision == PermissionDecision.BLOCK
    assert "ValidateResult" in gate.last_block_reason


# ---------------------------------------------------------------------------
# Layer 3 — plan mode is authoritative (hooks can't override)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_plan_mode_blocks_writes(tmp_path) -> None:
    reg = _make_registry(_def("write_file", PermissionLevel.WRITE))
    ctx = _ctx(tmp_path)
    ctx.plan_mode.enter("test")
    gate = PermissionGate(reg, ctx)
    tc = ToolCall(id="1", name="write_file", arguments={})
    decision = await gate.check(tc, _always_allow)
    assert decision == PermissionDecision.BLOCK
    assert "plan mode" in gate.last_block_reason


@pytest.mark.asyncio
async def test_plan_mode_authoritative_over_hook_allow(tmp_path) -> None:
    """A hook cannot un-block a plan-mode-blocked tool."""
    reg = _make_registry(_def("write_file", PermissionLevel.WRITE))
    ctx = _ctx(tmp_path)
    ctx.plan_mode.enter("test")
    hooks = HookSystem()
    hooks.register("pre_tool_use", lambda **kw: "allow")  # explicit allow
    gate = PermissionGate(reg, ctx, hooks=hooks)
    tc = ToolCall(id="1", name="write_file", arguments={})
    decision = await gate.check(tc, _always_allow)
    assert decision == PermissionDecision.BLOCK


# ---------------------------------------------------------------------------
# Layer 4 — hook denial
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_hook_deny_blocks(tmp_path) -> None:
    reg = _make_registry(_def("read"))
    hooks = HookSystem()
    hooks.register("pre_tool_use", lambda **kw: "deny")
    gate = PermissionGate(reg, _ctx(tmp_path), hooks=hooks)
    tc = ToolCall(id="1", name="read", arguments={})
    decision = await gate.check(tc, _always_allow)
    assert decision == PermissionDecision.BLOCK
    assert "hook" in gate.last_block_reason


@pytest.mark.asyncio
async def test_hook_receives_tool_call(tmp_path) -> None:
    reg = _make_registry(_def("read"))
    seen: list = []

    def hook(**kw):
        seen.append(kw)

    hooks = HookSystem()
    hooks.register("pre_tool_use", hook)
    gate = PermissionGate(reg, _ctx(tmp_path), hooks=hooks)
    tc = ToolCall(id="abc", name="read", arguments={"path": "x"})
    await gate.check(tc, _always_allow)
    assert seen and seen[0]["tool_call"].id == "abc"


# ---------------------------------------------------------------------------
# Layer 5 — permission level
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_read_only_allowed(tmp_path) -> None:
    reg = _make_registry(_def("read", PermissionLevel.READ_ONLY))
    gate = PermissionGate(reg, _ctx(tmp_path))
    decision = await gate.check(ToolCall(id="1", name="read", arguments={}), _always_allow)
    assert decision == PermissionDecision.ALLOW


@pytest.mark.asyncio
async def test_destructive_always_blocked(tmp_path) -> None:
    reg = _make_registry(_def("rm", PermissionLevel.DESTRUCTIVE))
    gate = PermissionGate(reg, _ctx(tmp_path))
    decision = await gate.check(ToolCall(id="1", name="rm", arguments={}), _always_allow)
    assert decision == PermissionDecision.BLOCK
    assert "destructive" in gate.last_block_reason


@pytest.mark.asyncio
async def test_write_asks_user(tmp_path) -> None:
    reg = _make_registry(_def("write", PermissionLevel.WRITE))
    gate = PermissionGate(reg, _ctx(tmp_path))
    # Callback returns BLOCK to verify it was actually consulted.
    decision = await gate.check(
        ToolCall(id="1", name="write", arguments={}), _always_deny,
    )
    assert decision == PermissionDecision.BLOCK


@pytest.mark.asyncio
async def test_yolo_skips_user_prompt_for_write(tmp_path) -> None:
    reg = _make_registry(_def("write", PermissionLevel.WRITE))
    gate = PermissionGate(reg, _ctx(tmp_path), yolo_mode=True)
    decision = await gate.check(
        ToolCall(id="1", name="write", arguments={}), _always_deny,
    )
    # YOLO mode says ALLOW even though callback would have blocked.
    assert decision == PermissionDecision.ALLOW


@pytest.mark.asyncio
async def test_yolo_does_not_unblock_destructive(tmp_path) -> None:
    reg = _make_registry(_def("rm", PermissionLevel.DESTRUCTIVE))
    gate = PermissionGate(reg, _ctx(tmp_path), yolo_mode=True)
    decision = await gate.check(
        ToolCall(id="1", name="rm", arguments={}), _always_allow,
    )
    assert decision == PermissionDecision.BLOCK
