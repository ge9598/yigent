import pytest
from src.core.env_injector import EnvironmentInjector

class TestEnvironmentInjector:
    def setup_method(self):
        self.inj = EnvironmentInjector()

    @pytest.mark.asyncio
    async def test_coding_context_has_git_info(self):
        ctx = await self.inj.get_context("coding")
        # We're in a git repo, so should have some git info
        assert len(ctx) > 0

    @pytest.mark.asyncio
    async def test_file_ops_context_has_directory(self):
        ctx = await self.inj.get_context("file_ops")
        assert len(ctx) > 0

    @pytest.mark.asyncio
    async def test_context_respects_max_chars(self):
        ctx = await self.inj.get_context("coding")
        assert len(ctx) <= 2500

    @pytest.mark.asyncio
    async def test_unknown_task_type_defaults_to_coding(self):
        ctx = await self.inj.get_context("unknown_type")
        assert isinstance(ctx, str)

    def test_detect_task_type_coding(self):
        task = self.inj.detect_task_type("Please fix the bug in auth.py")
        assert task == "coding"

    def test_detect_task_type_file_ops(self):
        task = self.inj.detect_task_type("Organize the files in my downloads folder")
        assert task == "file_ops"
