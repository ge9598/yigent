"""Multi-agent coordinator — spawn modes + shared task board.

Three spawn modes inspired by Claude Code's architecture:

  - Main: the default agent. Owns its context, budget, cache.
  - Fork: shares parent's prompt cache (so warm cache is reused) but runs
    in an ISOLATED conversation. The child's final answer is written to an
    ``output_file`` — intermediate tool output never leaks back into the
    parent's context. Gets a slice of the parent's iteration budget.
    Useful for "go do this side task but don't clutter my context."
  - Subagent: fresh conversation + fresh cache. Allocated budget is
    deducted from parent's pool. Useful for delegated subtasks that
    don't need the parent's history.

``TaskBoard`` is an in-memory DAG of work items, atomically claimable.
All spawned agents share the same board so they can coordinate.
"""

from __future__ import annotations

import asyncio
import copy
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

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


SpawnMode = Literal["main", "fork", "subagent"]


@dataclass
class SpawnedAgent:
    """Handle returned from spawn_fork / spawn_subagent.

    ``mode`` is one of ``main`` / ``fork`` / ``subagent``. ``main`` is the root
    handle (returned by ``MultiAgentCoordinator.main_handle()``); ``fork`` and
    ``subagent`` come from spawn methods.

    ``output_file`` is set on Fork: the child writes its final answer here so
    the parent can read it without inheriting the child's intermediate tool
    output. Subagents return their answer through the parent's normal
    conversation flow, so output_file stays None.
    """
    mode: SpawnMode
    budget: "IterationBudget"
    conversation: list
    cache: "PromptCache"
    task_board: TaskBoard
    output_file: Path | None = None

    def write_result(self, content: str) -> None:
        """Persist the child's final answer to ``output_file``.

        No-op if ``output_file`` is None (e.g. main / subagent modes). For
        forks, ensures parent dirs exist and writes UTF-8 markdown.
        """
        if self.output_file is None:
            return
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self.output_file.write_text(content, encoding="utf-8")


class MultiAgentCoordinator:
    """Factory for spawning Fork and Subagent children.

    One coordinator per "main" agent. Children share this coordinator's
    task board so they can pick up work from each other.

    ``trajectories_dir`` is the default directory under which Fork output
    files are created when the caller doesn't provide one explicitly.
    Defaults to ``trajectories/`` (per CLAUDE.md project map).
    """

    def __init__(
        self,
        parent_budget: "IterationBudget",
        parent_conversation: list,
        parent_cache: "PromptCache",
        task_board: TaskBoard | None = None,
        trajectories_dir: Path | str = "trajectories",
    ) -> None:
        self._parent_budget = parent_budget
        self._parent_conversation = parent_conversation
        self._parent_cache = parent_cache
        self._task_board = task_board or TaskBoard()
        self._trajectories_dir = Path(trajectories_dir)

    def main_handle(self) -> SpawnedAgent:
        """Return a SpawnedAgent handle for the main agent itself.

        Useful for surfacing the root agent in introspection / logging
        alongside its forks and subagents — same dataclass shape, mode
        flagged as ``main``. The conversation reference IS shared (the
        main handle wraps the live parent conversation).
        """
        return SpawnedAgent(
            mode="main",
            budget=self._parent_budget,
            conversation=self._parent_conversation,
            cache=self._parent_cache,
            task_board=self._task_board,
            output_file=None,
        )

    async def spawn_fork(
        self,
        budget_share: int,
        output_file: Path | str | None = None,
    ) -> SpawnedAgent:
        """Fork: shared prompt cache, ISOLATED conversation, output to file.

        The child gets a deep copy of the parent's conversation (so its
        tool output and intermediate reasoning never leak back). When the
        child finishes it calls ``write_result(final_answer)`` to persist
        its answer; the parent reads this back to inject just the result
        — not the work — into its own context.

        If ``output_file`` is None, defaults to
        ``{trajectories_dir}/fork_{uuid}.md``.
        """
        sub_budget = await self._parent_budget.allocate(budget_share)
        if output_file is None:
            output_file = self._trajectories_dir / f"fork_{uuid.uuid4().hex[:12]}.md"
        else:
            output_file = Path(output_file)
        return SpawnedAgent(
            mode="fork",
            budget=sub_budget,
            # Deep copy: child's mutations don't affect parent's history.
            # Cheap because conversations are dicts of strings/lists.
            conversation=copy.deepcopy(self._parent_conversation),
            cache=self._parent_cache.on_fork(),
            task_board=self._task_board,
            output_file=output_file,
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
            output_file=None,
        )

    @property
    def task_board(self) -> TaskBoard:
        return self._task_board
