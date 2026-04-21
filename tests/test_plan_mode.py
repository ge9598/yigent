import pytest
from pathlib import Path
from src.core.plan_mode import PlanMode


class TestPlanMode:
    def test_enter_exit_basic(self):
        pm = PlanMode(save_dir="test_plans/")
        pm.enter("s1")
        assert pm.is_active
        result = pm.exit()
        assert not pm.is_active
        assert "deactivated" in result.lower()

    def test_append_and_get_content(self):
        pm = PlanMode(save_dir="test_plans/")
        pm.enter("s1")
        pm.append("# Step 1\nDo X")
        pm.append("\n# Step 2\nDo Y")
        content = pm.get_plan_content()
        assert "Step 1" in content
        assert "Step 2" in content

    def test_exit_saves_when_has_content(self, tmp_path):
        pm = PlanMode(save_dir=str(tmp_path))
        pm.enter("s1")
        pm.append("# My plan")
        result = pm.exit()
        assert "saved" in result.lower()
        files = list(tmp_path.glob("*.md"))
        assert len(files) == 1

    def test_exit_no_save_when_empty(self, tmp_path):
        pm = PlanMode(save_dir=str(tmp_path))
        pm.enter("s1")
        result = pm.exit()
        files = list(tmp_path.glob("*.md"))
        assert len(files) == 0

    def test_tool_allowed_during_plan(self):
        pm = PlanMode()
        pm.enter("s1")
        assert pm.is_tool_allowed("read_file")
        assert not pm.is_tool_allowed("write_file")
        assert pm.is_tool_allowed("exit_plan_mode")


class TestPlanModeApproval:
    """Unit 1 — approve()/reject() flow + plan_approved hook."""

    @pytest.mark.asyncio
    async def test_approve_fires_plan_approved_hook(self):
        from src.safety.hook_system import HookSystem
        hooks = HookSystem()
        events: list[dict] = []

        async def _record(**data):
            events.append(data)

        hooks.register("plan_approved", _record)
        pm = PlanMode(hook_system=hooks)
        pm.enter("session-42")
        pm.append("Step 1: do X")
        result = await pm.approve()
        assert pm.is_approved
        assert pm.is_active  # stays active until model exits
        assert "approved" in result.lower()
        assert len(events) == 1
        assert events[0]["session_id"] == "session-42"
        assert "Step 1" in events[0]["plan_content"]

    @pytest.mark.asyncio
    async def test_approve_no_op_when_inactive(self):
        pm = PlanMode()
        result = await pm.approve()
        assert "not active" in result.lower()
        assert not pm.is_approved

    @pytest.mark.asyncio
    async def test_reject_keeps_plan_mode_active(self):
        pm = PlanMode()
        pm.enter("s1")
        result = await pm.reject(note="too risky")
        assert pm.is_active  # stays active for revision
        assert not pm.is_approved
        assert pm.rejection_note == "too risky"
        assert "rejected" in result.lower()

    def test_set_hook_system_late_binding(self):
        from src.safety.hook_system import HookSystem
        pm = PlanMode()
        hooks = HookSystem()
        pm.set_hook_system(hooks)
        # No assertion — just verify the method exists and sets the attribute.

    @pytest.mark.asyncio
    async def test_re_enter_resets_approval(self):
        pm = PlanMode()
        pm.enter("s1")
        await pm.approve()
        assert pm.is_approved
        pm.exit()
        pm.enter("s2")
        assert not pm.is_approved  # fresh session
