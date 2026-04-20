"""Multi-agent coordinator — spawn modes + shared task board.

Three spawn modes inspired by Claude Code's architecture:

  - Main: the default agent. Owns its context, budget, cache.
  - Fork: shares parent's conversation (by reference) and cache hash.
    Gets a slice of the parent's iteration budget. Useful for "go do
    this side task but keep all the context we already have."
  - Subagent: fresh conversation + fresh cache. Allocated budget is
    deducted from parent's pool. Useful for delegated subtasks that
    don't need the parent's history.

``TaskBoard`` is an in-memory DAG of work items, atomically claimable.
All spawned agents share the same board so they can coordinate.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.context.prompt_cache import PromptCache
    from src.core.iteration_budget import IterationBudget


@dataclass
class _TaskEntry:
    task_id: str
    description: str
    depends_on: list[str] = field(default_factory=list)
    state: str = "pending"  # pending | claimed | completed | failed
    assignee: str | None = None
    result: Any = None


class TaskBoard:
    """Thread-safe (asyncio.Lock) in-memory task registry with dependencies."""

    def __init__(self) -> None:
        self._tasks: dict[str, _TaskEntry] = {}
        self._lock = asyncio.Lock()

    async def create(
        self,
        task_id: str,
        description: str,
        depends_on: list[str] | None = None,
    ) -> None:
        async with self._lock:
            if task_id in self._tasks:
                raise ValueError(f"Task {task_id!r} already exists")
            self._tasks[task_id] = _TaskEntry(
                task_id=task_id,
                description=description,
                depends_on=list(depends_on or []),
            )

    async def claim(self, task_id: str, agent_id: str) -> bool:
        """Atomically claim a task. Returns True on success, False otherwise."""
        async with self._lock:
            entry = self._tasks.get(task_id)
            if entry is None:
                return False
            if entry.state != "pending":
                return False
            for dep in entry.depends_on:
                dep_entry = self._tasks.get(dep)
                if dep_entry is None or dep_entry.state != "completed":
                    return False
            entry.state = "claimed"
            entry.assignee = agent_id
            return True

    async def complete(self, task_id: str, result: Any = None) -> None:
        async with self._lock:
            entry = self._tasks.get(task_id)
            if entry is None:
                raise KeyError(f"Unknown task {task_id!r}")
            entry.state = "completed"
            entry.result = result

    async def fail(self, task_id: str, reason: str) -> None:
        async with self._lock:
            entry = self._tasks.get(task_id)
            if entry is None:
                raise KeyError(f"Unknown task {task_id!r}")
            entry.state = "failed"
            entry.result = {"error": reason}

    async def get_status(self) -> dict[str, dict[str, Any]]:
        async with self._lock:
            return {
                t.task_id: {
                    "description": t.description,
                    "depends_on": list(t.depends_on),
                    "state": t.state,
                    "assignee": t.assignee,
                    "result": t.result,
                }
                for t in self._tasks.values()
            }


@dataclass
class SpawnedAgent:
    """Handle returned from spawn_fork / spawn_subagent."""
    mode: str  # "fork" | "subagent"
    budget: "IterationBudget"
    conversation: list
    cache: "PromptCache"
    task_board: TaskBoard


class MultiAgentCoordinator:
    """Factory for spawning Fork and Subagent children.

    One coordinator per "main" agent. Children share this coordinator's
    task board so they can pick up work from each other.
    """

    def __init__(
        self,
        parent_budget: "IterationBudget",
        parent_conversation: list,
        parent_cache: "PromptCache",
        task_board: TaskBoard | None = None,
    ) -> None:
        self._parent_budget = parent_budget
        self._parent_conversation = parent_conversation
        self._parent_cache = parent_cache
        self._task_board = task_board or TaskBoard()

    async def spawn_fork(self, budget_share: int) -> SpawnedAgent:
        """Fork: inherits parent conversation+cache, allocated budget slice."""
        sub_budget = await self._parent_budget.allocate(budget_share)
        return SpawnedAgent(
            mode="fork",
            budget=sub_budget,
            conversation=self._parent_conversation,
            cache=self._parent_cache.on_fork(),
            task_board=self._task_board,
        )

    async def spawn_subagent(
        self,
        budget_alloc: int,
        system_prompt: list,
    ) -> SpawnedAgent:
        """Subagent: fresh conversation + fresh cache, allocated budget slice."""
        sub_budget = await self._parent_budget.allocate(budget_alloc)
        return SpawnedAgent(
            mode="subagent",
            budget=sub_budget,
            conversation=[],
            cache=self._parent_cache.on_subagent(system_prompt),
            task_board=self._task_board,
        )

    @property
    def task_board(self) -> TaskBoard:
        return self._task_board
