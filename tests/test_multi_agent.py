"""Tests for multi-agent coordinator — TaskBoard + spawn modes."""

from __future__ import annotations

import asyncio

import pytest

from src.core.multi_agent import MultiAgentCoordinator, SpawnedAgent, TaskBoard


@pytest.mark.asyncio
async def test_task_board_create_claim_complete():
    board = TaskBoard()
    await board.create("t1", description="do thing")
    status = await board.get_status()
    assert status["t1"]["state"] == "pending"

    claimed = await board.claim("t1", agent_id="a1")
    assert claimed is True

    claimed_again = await board.claim("t1", agent_id="a2")
    assert claimed_again is False

    await board.complete("t1", result="done")
    status = await board.get_status()
    assert status["t1"]["state"] == "completed"
    assert status["t1"]["result"] == "done"


@pytest.mark.asyncio
async def test_task_board_rejects_duplicate_create():
    board = TaskBoard()
    await board.create("t", description="x")
    with pytest.raises(ValueError, match="already exists"):
        await board.create("t", description="y")


@pytest.mark.asyncio
async def test_task_board_respects_dependencies():
    board = TaskBoard()
    await board.create("parent", description="parent")
    await board.create("child", description="child", depends_on=["parent"])

    claimed = await board.claim("child", agent_id="a1")
    assert claimed is False  # parent not complete

    await board.claim("parent", agent_id="a1")
    await board.complete("parent", result="p-done")

    claimed = await board.claim("child", agent_id="a2")
    assert claimed is True


@pytest.mark.asyncio
async def test_task_board_claim_is_atomic():
    board = TaskBoard()
    await board.create("t", description="racey")

    results = await asyncio.gather(
        *[board.claim("t", agent_id=f"a{i}") for i in range(10)]
    )
    assert results.count(True) == 1
    assert results.count(False) == 9


@pytest.mark.asyncio
async def test_task_board_fail():
    board = TaskBoard()
    await board.create("t", description="x")
    await board.claim("t", agent_id="a")
    await board.fail("t", reason="broke")
    status = await board.get_status()
    assert status["t"]["state"] == "failed"
    assert status["t"]["result"]["error"] == "broke"


@pytest.mark.asyncio
async def test_task_board_claim_unknown_returns_false():
    board = TaskBoard()
    assert await board.claim("nope", agent_id="a") is False


@pytest.mark.asyncio
async def test_task_board_complete_unknown_raises():
    board = TaskBoard()
    with pytest.raises(KeyError):
        await board.complete("nope", result="x")


@pytest.mark.asyncio
async def test_spawn_fork_shares_conversation_and_budget():
    from src.context.prompt_cache import PromptCache
    from src.core.iteration_budget import IterationBudget

    parent_cache = PromptCache(frozen_system=[{"role": "system", "content": "p"}])
    parent_budget = IterationBudget(total=50, enable_warning=False)
    parent_conv = [{"role": "user", "content": "hi"}]

    coord = MultiAgentCoordinator(
        parent_budget=parent_budget,
        parent_conversation=parent_conv,
        parent_cache=parent_cache,
    )

    child = await coord.spawn_fork(budget_share=10)
    assert isinstance(child, SpawnedAgent)
    assert child.mode == "fork"
    # Unit 8 — Fork has an ISOLATED conversation (deep copy), NOT a reference.
    # Mutating the child's conversation must not affect the parent. Cache is
    # still shared (same prefix_hash) so warm cache is reused.
    assert child.conversation == parent_conv
    assert child.conversation is not parent_conv
    child.conversation.append({"role": "user", "content": "child-only"})
    assert len(parent_conv) == 1, "Fork mutation leaked into parent"
    # Fork's cache has same prefix_hash as parent
    assert child.cache.prefix_hash == parent_cache.prefix_hash
    # Unit 10 — budget is now truly shared. allocate() returns a child handle
    # with a local cap (10) but does NOT pre-deduct from the parent.
    assert parent_budget.remaining == 50  # unchanged
    assert child.budget.total == 10  # local cap
    # output_file defaults to trajectories/fork_*.md
    assert child.output_file is not None
    assert child.output_file.parent.name == "trajectories"
    assert child.output_file.name.startswith("fork_")


@pytest.mark.asyncio
async def test_spawn_subagent_has_fresh_context():
    from src.context.prompt_cache import PromptCache
    from src.core.iteration_budget import IterationBudget

    parent_cache = PromptCache(frozen_system=[{"role": "system", "content": "p"}])
    parent_budget = IterationBudget(total=50, enable_warning=False)
    parent_conv = [{"role": "user", "content": "hi"}]

    coord = MultiAgentCoordinator(
        parent_budget=parent_budget,
        parent_conversation=parent_conv,
        parent_cache=parent_cache,
    )

    sub_system = [{"role": "system", "content": "subagent system"}]
    child = await coord.spawn_subagent(
        budget_alloc=15,
        system_prompt=sub_system,
    )
    assert child.mode == "subagent"
    # Subagent has its own (fresh) conversation
    assert child.conversation is not parent_conv
    assert child.conversation == []
    # Subagent's cache has a different prefix_hash
    assert child.cache.prefix_hash != parent_cache.prefix_hash
    # Unit 10 — shared budget; allocate() doesn't pre-deduct.
    assert parent_budget.remaining == 50  # unchanged until either side spends


@pytest.mark.asyncio
async def test_coordinator_shares_taskboard_across_spawns():
    from src.context.prompt_cache import PromptCache
    from src.core.iteration_budget import IterationBudget

    parent_cache = PromptCache(frozen_system=[{"role": "system", "content": "p"}])
    parent_budget = IterationBudget(total=100, enable_warning=False)

    coord = MultiAgentCoordinator(
        parent_budget=parent_budget,
        parent_conversation=[],
        parent_cache=parent_cache,
    )

    fork = await coord.spawn_fork(budget_share=10)
    sub = await coord.spawn_subagent(
        budget_alloc=10,
        system_prompt=[{"role": "system", "content": "sub"}],
    )

    # All three share the same task board
    assert fork.task_board is coord.task_board
    assert sub.task_board is coord.task_board


@pytest.mark.asyncio
async def test_task_tools_roundtrip():
    from src.core.multi_agent import TaskBoard
    from src.tools.task_tools import make_task_tools

    board = TaskBoard()
    tools = {t.name: t for t in make_task_tools(board)}
    assert set(tools) == {"create_task", "claim_task", "complete_task", "task_status"}

    result = await tools["create_task"].handler(task_id="x", description="do x")
    assert "created" in result.lower()

    result = await tools["claim_task"].handler(task_id="x", agent_id="a1")
    assert "claimed" == result

    result = await tools["complete_task"].handler(task_id="x", result="ok")
    assert "completed" in result.lower()

    status = await tools["task_status"].handler()
    import json as _json
    parsed = _json.loads(status)
    assert parsed["x"]["state"] == "completed"


@pytest.mark.asyncio
async def test_task_tools_permission_level():
    """create/claim/complete must be WRITE; status must be READ_ONLY."""
    from src.core.multi_agent import TaskBoard
    from src.core.types import PermissionLevel
    from src.tools.task_tools import make_task_tools

    board = TaskBoard()
    tools = {t.name: t for t in make_task_tools(board)}
    assert tools["create_task"].schema.permission_level == PermissionLevel.WRITE
    assert tools["claim_task"].schema.permission_level == PermissionLevel.WRITE
    assert tools["complete_task"].schema.permission_level == PermissionLevel.WRITE
    assert tools["task_status"].schema.permission_level == PermissionLevel.READ_ONLY


# ---------------------------------------------------------------------------
# Unit 8 — Fork output_file + main_handle + write_result
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fork_output_file_explicit(tmp_path):
    from src.context.prompt_cache import PromptCache
    from src.core.iteration_budget import IterationBudget
    parent_cache = PromptCache(frozen_system=[{"role": "system", "content": "p"}])
    parent_budget = IterationBudget(total=50, enable_warning=False)
    coord = MultiAgentCoordinator(
        parent_budget=parent_budget,
        parent_conversation=[],
        parent_cache=parent_cache,
        trajectories_dir=tmp_path,
    )
    out = tmp_path / "my_fork.md"
    child = await coord.spawn_fork(budget_share=5, output_file=out)
    assert child.output_file == out

    child.write_result("# Final answer\n\nDid the thing.")
    assert out.exists()
    assert "Did the thing" in out.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_fork_output_file_default_in_trajectories_dir(tmp_path):
    from src.context.prompt_cache import PromptCache
    from src.core.iteration_budget import IterationBudget
    parent_cache = PromptCache(frozen_system=[{"role": "system", "content": "p"}])
    parent_budget = IterationBudget(total=50, enable_warning=False)
    coord = MultiAgentCoordinator(
        parent_budget=parent_budget,
        parent_conversation=[],
        parent_cache=parent_cache,
        trajectories_dir=tmp_path,
    )
    child = await coord.spawn_fork(budget_share=5)
    assert child.output_file is not None
    assert child.output_file.parent == tmp_path
    assert child.output_file.name.startswith("fork_")
    assert child.output_file.suffix == ".md"

    child.write_result("hello")
    assert child.output_file.exists()


def test_main_handle_returns_main_mode():
    from src.context.prompt_cache import PromptCache
    from src.core.iteration_budget import IterationBudget
    parent_cache = PromptCache(frozen_system=[{"role": "system", "content": "p"}])
    parent_budget = IterationBudget(total=50, enable_warning=False)
    parent_conv = [{"role": "user", "content": "hi"}]
    coord = MultiAgentCoordinator(
        parent_budget=parent_budget,
        parent_conversation=parent_conv,
        parent_cache=parent_cache,
    )
    handle = coord.main_handle()
    assert handle.mode == "main"
    # main wraps the LIVE conversation, not a copy.
    assert handle.conversation is parent_conv
    assert handle.cache is parent_cache
    assert handle.output_file is None


def test_write_result_noop_when_no_output_file():
    """Subagent / main have no output_file — write_result is silent no-op."""
    from src.context.prompt_cache import PromptCache
    from src.core.iteration_budget import IterationBudget
    parent_cache = PromptCache(frozen_system=[{"role": "system", "content": "p"}])
    parent_budget = IterationBudget(total=50, enable_warning=False)
    coord = MultiAgentCoordinator(
        parent_budget=parent_budget,
        parent_conversation=[],
        parent_cache=parent_cache,
    )
    handle = coord.main_handle()
    handle.write_result("anything")  # must not raise
