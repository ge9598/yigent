"""Shared types for the Yigent agent harness."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Coroutine, Literal, TypedDict

from pydantic import BaseModel

if TYPE_CHECKING:
    from src.core.config import AgentConfig
    from src.core.plan_mode import PlanMode
    from src.memory.markdown_store import MarkdownMemoryStore
    from src.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class PermissionLevel(str, Enum):
    READ_ONLY = "read_only"
    WRITE = "write"
    EXECUTE = "execute"
    DESTRUCTIVE = "destructive"


class PermissionDecision(str, Enum):
    ALLOW = "allow"
    ASK_USER = "ask_user"
    BLOCK = "block"


# ---------------------------------------------------------------------------
# Messages — TypedDict to stay dict-compatible with OpenAI chat format
# ---------------------------------------------------------------------------

class FunctionCall(TypedDict, total=False):
    name: str
    arguments: str  # JSON string


class ToolCallDict(TypedDict, total=False):
    id: str
    type: str  # "function"
    function: FunctionCall


class Message(TypedDict, total=False):
    role: str  # "system" | "user" | "assistant" | "tool"
    content: str | None
    tool_calls: list[ToolCallDict]
    tool_call_id: str
    name: str


# ---------------------------------------------------------------------------
# Tool call / result — runtime dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    status: Literal["pending", "running", "completed", "failed"] = "pending"


@dataclass
class ToolResult:
    tool_call_id: str
    name: str
    content: str
    is_error: bool = False

    def to_message(self) -> Message:
        return Message(
            role="tool",
            tool_call_id=self.tool_call_id,
            content=self.content,
        )


# ---------------------------------------------------------------------------
# Stream chunks — yielded by provider during streaming
# ---------------------------------------------------------------------------

@dataclass
class StreamChunk:
    type: Literal[
        "token",
        "tool_call_start",
        "tool_call_delta",
        "tool_call_complete",
        "done",
    ]
    data: Any = None
    model: str | None = None


# ---------------------------------------------------------------------------
# Events — yielded by agent_loop to the UI layer
# ---------------------------------------------------------------------------

@dataclass
class TokenEvent:
    token: str


@dataclass
class ToolCallStartEvent:
    tool_call: ToolCall


@dataclass
class ToolResultEvent:
    result: ToolResult


@dataclass
class PermissionRequestEvent:
    """Yielded when a tool call needs user approval.

    The consumer (CLI) sets `decision` before calling __anext__()
    on the agent loop generator. The loop reads `decision` after resume.
    """
    tool_call: ToolCall
    decision: PermissionDecision | None = None


@dataclass
class FinalAnswerEvent:
    content: str


@dataclass
class BudgetExhaustedEvent:
    remaining: int
    total: int


@dataclass
class ErrorEvent:
    error: str
    recoverable: bool = False


Event = (
    TokenEvent
    | ToolCallStartEvent
    | ToolResultEvent
    | PermissionRequestEvent
    | FinalAnswerEvent
    | BudgetExhaustedEvent
    | ErrorEvent
)


# ---------------------------------------------------------------------------
# Tool schema — Pydantic for validation
# ---------------------------------------------------------------------------

class ToolSchema(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema
    permission_level: PermissionLevel = PermissionLevel.READ_ONLY
    timeout: int = 30
    deferred: bool = False

    def to_openai_tool(self) -> dict[str, Any]:
        """Convert to the OpenAI tools parameter format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


# ---------------------------------------------------------------------------
# Tool definition — used by registry for registration
# ---------------------------------------------------------------------------

ToolHandler = Callable[..., Coroutine[Any, Any, str]]
UserCallback = Callable[[str], Coroutine[Any, Any, str]]


# ---------------------------------------------------------------------------
# ValidateResult — structured decision from a tool's self-check
# ---------------------------------------------------------------------------
#
# Modelled after Claude Code's ``checkPermissions`` return shape. A validator
# can:
#   - allow unconditionally
#   - deny with a reason (shown to the user + model)
#   - ask the user even for otherwise-auto-allowed tools (e.g. READ_ONLY)
#   - rewrite the tool's input (normalise paths, strip trailing whitespace)
#
# The ``ask`` decision does NOT bypass plan-mode / hook layers — those still
# run first. It only forces layer 5 to prompt the user regardless of the
# tool's permission_level.

@dataclass
class ValidateResult:
    """Result of ``ToolDefinition.validate()``.

    Attributes:
        decision: "allow" / "ask" / "deny"
        reason: human-readable explanation (required when deny or ask)
        updated_input: optional rewritten args; if set, the handler and the
                       persisted tool_call see these instead of the originals
    """
    decision: Literal["allow", "ask", "deny"]
    reason: str = ""
    updated_input: dict[str, Any] | None = None


ToolValidator = Callable[..., Coroutine[Any, Any, "ValidateResult"]]


@dataclass
class ToolContext:
    """Runtime dependencies injected into tool handlers that need them.

    Handlers declare need via ``ToolDefinition.needs_context=True``;
    simple handlers ignore this. Created once per session by the CLI/API layer.
    """
    plan_mode: PlanMode
    registry: ToolRegistry
    config: AgentConfig
    working_dir: Path
    user_callback: UserCallback | None = None
    session_id: str | None = None
    memory_store: "MarkdownMemoryStore | None" = None


@dataclass
class ToolDefinition:
    name: str
    description: str
    handler: ToolHandler
    schema: ToolSchema
    validate: ToolValidator | None = None
    needs_context: bool = False
