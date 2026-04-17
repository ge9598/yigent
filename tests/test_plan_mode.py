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
