"""Null-object stubs for Phase 2 modules.

These let agent_loop.py accept the same parameters it will use in Phase 2,
but do nothing. Swap for real implementations with zero code change in the loop.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .types import Message


class NullContextAssembler:
    """Phase 2: replaced by src/context/assembler.py."""

    async def assemble(
        self,
        tool_registry: Any,
        env_injector: Any,
        conversation: list[Message],
        task_type: str,
    ) -> list[Message]:
        return list(conversation)


class NullHookSystem:
    """Phase 2: replaced by src/safety/hook_system.py."""

    async def fire(self, event_name: str, **data: Any) -> None:
        pass


class NullLearningLoop:
    """Phase 2: replaced by src/learning/nudge.py."""

    async def nudge(self, conversation: list[Message], turn_count: int) -> None:
        pass

    async def maybe_create_skill(self, conversation: list[Message]) -> None:
        pass


class NullTrajectoryRecorder:
    """Phase 2: replaced by src/learning/trajectory.py."""

    def record_turn(
        self,
        user_msg: Message | None,
        tool_calls: list[Any] | None,
        assistant_response: str | None,
    ) -> None:
        pass
