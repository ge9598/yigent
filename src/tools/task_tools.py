"""Task board tools — create/claim/complete via a shared TaskBoard.

Bound to a TaskBoard instance at construction time. Typically exposed to
the main agent; forks/subagents can also use them to coordinate.
"""

from __future__ import annotations

import json

from src.core.multi_agent import TaskBoard
from src.core.types import (
    PermissionLevel,
    ToolDefinition,
    ToolSchema,
)


def make_task_tools(board: TaskBoard) -> list[ToolDefinition]:
    """Return four ToolDefinitions (create_task / claim_task / complete_task / task_status)."""

    async def _create_task(
        task_id: str,
        description: str,
        depends_on: list[str] | None = None,
    ) -> str:
        try:
            await board.create(
                task_id=task_id,
                description=description,
                depends_on=depends_on or [],
            )
            return f"Task {task_id!r} created."
        except ValueError as exc:
            return f"Error: {exc}"

    async def _claim_task(task_id: str, agent_id: str) -> str:
        ok = await board.claim(task_id=task_id, agent_id=agent_id)
        return "claimed" if ok else "could not claim (already claimed or deps unmet)"

    async def _complete_task(task_id: str, result: str | None = None) -> str:
        try:
            await board.complete(task_id=task_id, result=result)
            return f"Task {task_id!r} completed."
        except KeyError:
            return f"Error: unknown task {task_id!r}"

    async def _task_status() -> str:
        status = await board.get_status()
        return json.dumps(status, indent=2, default=str)

    create_tool = ToolDefinition(
        name="create_task",
        description="Create a new task on the shared task board. Supports dependencies.",
        handler=_create_task,
        schema=ToolSchema(
            name="create_task",
            description="Create a new task on the shared task board.",
            parameters={
                "type": "object",
                "properties": {
                    "task_id": {"type": "string", "description": "Unique task identifier"},
                    "description": {"type": "string", "description": "What the task is about"},
                    "depends_on": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Task IDs that must complete first",
                    },
                },
                "required": ["task_id", "description"],
            },
            permission_level=PermissionLevel.WRITE,
            timeout=5,
            deferred=False,
        ),
    )

    claim_tool = ToolDefinition(
        name="claim_task",
        description="Claim a pending task whose dependencies are satisfied.",
        handler=_claim_task,
        schema=ToolSchema(
            name="claim_task",
            description="Claim a pending task on the shared task board.",
            parameters={
                "type": "object",
                "properties": {
                    "task_id": {"type": "string"},
                    "agent_id": {"type": "string"},
                },
                "required": ["task_id", "agent_id"],
            },
            permission_level=PermissionLevel.WRITE,
            timeout=5,
            deferred=False,
        ),
    )

    complete_tool = ToolDefinition(
        name="complete_task",
        description="Mark a claimed task as completed, with an optional result string.",
        handler=_complete_task,
        schema=ToolSchema(
            name="complete_task",
            description="Mark a task as completed.",
            parameters={
                "type": "object",
                "properties": {
                    "task_id": {"type": "string"},
                    "result": {"type": "string"},
                },
                "required": ["task_id"],
            },
            permission_level=PermissionLevel.WRITE,
            timeout=5,
            deferred=False,
        ),
    )

    status_tool = ToolDefinition(
        name="task_status",
        description="Return a JSON snapshot of all tasks on the shared board.",
        handler=_task_status,
        schema=ToolSchema(
            name="task_status",
            description="Return a JSON snapshot of all tasks on the shared board.",
            parameters={"type": "object", "properties": {}},
            permission_level=PermissionLevel.READ_ONLY,
            timeout=5,
            deferred=False,
        ),
    )

    return [create_tool, claim_tool, complete_tool, status_tool]
