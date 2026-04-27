"""Shared types for the Yigent agent harness."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import (
    TYPE_CHECKING, Any, Callable, Coroutine, Literal, Protocol,
    TypedDict, runtime_checkable,
)

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


class FatalToolError(Exception):
    """Raised by a tool handler or the permission subsystem when an error is
    so severe that pending sibling tool calls in the same turn must be
    cancelled (not allowed to finish). Examples:

    - permission gate's own validation chain panics (not a normal deny)
    - tool handler hits OOM / catastrophic state corruption

    Normal tool failures (timeout, business-logic exception, permission
    BLOCK) do NOT raise this — they return error ToolResult and let
    siblings finish. Only true "the world is broken" errors should propagate.
    """


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
    # Reasoning / chain-of-thought content. Two fields with distinct roles:
    #   reasoning_details — native Anthropic thinking blocks preserved verbatim
    #     (with signature). Only populated when the provider is Anthropic
    #     official; re-sent on the next turn to satisfy extended-thinking
    #     multi-turn requirements. Stripped on third-party /anthropic endpoints
    #     (MiniMax, Bedrock) because the signature is Anthropic-proprietary.
    #   reasoning_text — plain-text summary extracted from any reasoning style
    #     (Anthropic thinking, OpenAI reasoning_content, <think> tags). Used
    #     for display, persistence, and aux-LLM consumption. Never round-tripped.
    reasoning_details: list[dict[str, Any]]
    reasoning_text: str


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
        "reasoning_delta",  # live thinking fragment — data is str
        "reasoning",  # final chain-of-thought — {"text": str, "details": list[dict] | None}
        "done",
    ]
    data: Any = None
    model: str | None = None


# ---------------------------------------------------------------------------
# Events — yielded by agent_loop to the UI layer
# ---------------------------------------------------------------------------

@dataclass
class TurnStartedEvent:
    """First event of every turn. The UI can show an immediate spinner.

    Distinct from :class:`ReasoningDeltaEvent` because the turn's earliest
    work (capability routing, context assembly, provider handshake) happens
    before any model output — yet still blocks for seconds on reasoning
    models. A "Preparing..." indicator fills that gap.
    """


@dataclass
class TokenEvent:
    token: str


@dataclass
class ReasoningDeltaEvent:
    """Live signal that the model is producing chain-of-thought.

    Emitted once per thinking fragment. UIs typically show a spinner and
    discard ``fragment`` (collapsed view), but a verbose mode can stream
    the text in a dim style. The final assembled reasoning is persisted
    separately on the assistant message via ``reasoning_text``.
    """
    fragment: str


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
class PlanModeTriggeredEvent:
    """Yielded when the capability router classifies a user turn as complex.

    The CLI reacts by entering plan mode (blocking writes and executes)
    before the agent loop continues.
    """
    reason: str = ""


@dataclass
class ErrorEvent:
    error: str
    recoverable: bool = False


@dataclass
class TruncatedEvent:
    """Yielded when the model stopped because it hit max_tokens.

    Distinct from a normal final answer: the assistant text is incomplete and
    the harness must NOT silently commit it as the final response. Today the
    UI surfaces this to the user; a future iteration could automatically issue
    a continuation prompt. ``content`` carries whatever partial text we got.
    """
    content: str
    finish_reason: str = "length"


@dataclass
class ProviderFallbackEvent:
    """Yielded when the agent loop falls back from primary to fallback provider.

    Surfaced so the UI can warn the user that the active model changed mid-task,
    and so logs/telemetry can attribute subsequent tokens to the fallback.
    """
    primary: str
    fallback: str
    reason: str


Event = (
    TurnStartedEvent
    | TokenEvent
    | ReasoningDeltaEvent
    | ToolCallStartEvent
    | ToolResultEvent
    | PermissionRequestEvent
    | FinalAnswerEvent
    | BudgetExhaustedEvent
    | PlanModeTriggeredEvent
    | ErrorEvent
    | TruncatedEvent
    | ProviderFallbackEvent
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
    # Mark tools that touch shared serial resources (stdin, terminal, UI prompt)
    # so the streaming executor doesn't run them concurrently with siblings.
    # Examples: ask_user (blocks for user input), exit_plan_mode (UI confirm).
    exclusive: bool = False

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


# ---------------------------------------------------------------------------
# Protocols — structural types for the agent_loop "optional" dependencies
# ---------------------------------------------------------------------------
#
# agent_loop accepts hooks / trajectory / learning as Protocol-typed args so
# the contract is explicit while still allowing simple test doubles. These
# are all runtime_checkable: ``isinstance(obj, HookSystemLike)`` works.
# They're deliberately loose (only the methods agent_loop actually calls).
# See audit Top-10 #4: before these, the loop used ``object | None`` with
# ``hasattr(...)`` probes — a typo in the learning module would silently turn
# nudges off with no error. Protocols let type-checkers + runtime checks
# catch that.


@runtime_checkable
class HookSystemLike(Protocol):
    """Fires lifecycle events. agent_loop only needs ``fire()``."""

    async def fire(self, event_name: str, **data: Any) -> Any: ...


@runtime_checkable
class TrajectoryRecorderLike(Protocol):
    """Always-on trajectory recorder. agent_loop calls record_turn + finalize."""

    def record_turn(
        self,
        user_message: str | None,
        assistant_text: str,
        tool_calls: list[Any],
        tool_results: list[Any],
        reasoning_text: str = "",
    ) -> None: ...

    def finalize(self, outcome: str = "completed") -> None: ...


@runtime_checkable
class LearningContainerLike(Protocol):
    """Container with optional nudge / skill_creator / skill_improver attrs.

    agent_loop attribute-probes each slot (they may be None individually)
    rather than requiring all three. The Protocol documents the surface
    without forcing every slot to exist.
    """

    nudge: Any  # NudgeEngineLike | None — checked via hasattr at use site
    skill_creator: Any
    skill_improver: Any
