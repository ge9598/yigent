"""Always-on trajectory recorder.

Records every turn of an agent session: the user input (if any), the
assistant's response (text + reasoning), the tool calls issued, and the tool
results. Recording is cheap — a dataclass append per turn, no I/O. Export is
explicit via ``save()`` or ``export_sharegpt()`` / ``export_rl()``.

Two export formats:

- **ShareGPT JSON** — the de-facto SFT format for tool-use fine-tuning.
  Each turn becomes 2-3 messages (human / gpt / tool) in a single
  conversation entry.
- **RL transitions** — (state, action, reward, next_state) tuples suitable
  for downstream GRPO / DPO / reward modeling. Reward is left to the caller
  (benchmark runner fills it based on rule-check + judge score); trajectory
  emits ``reward=None`` so it's obvious when reward hasn't been annotated.

The recorder is session-scoped. Spawn a new ``TrajectoryRecorder`` per
session; the ``session_id`` is embedded in every export so multi-session
dumps stay separable.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from src.core.types import Message, ToolCall, ToolResult


@dataclass
class TurnRecord:
    """One turn's worth of data.

    A turn is one iteration of the agent loop: user message (if this is the
    first turn of a multi-turn conversation), LLM response, any tool calls,
    any tool results. For agentic turns where the model immediately calls
    tools without user input, ``user_msg`` is ``None``.
    """

    turn_index: int
    timestamp: float
    assistant_msg: Message
    user_msg: Message | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    reasoning_text: str | None = None


class TrajectoryRecorder:
    """Records agent turns; exports as ShareGPT or RL transitions.

    Thread-safety note: the recorder assumes single-agent-loop ownership.
    Multi-agent (Fork / Subagent) should each get their own recorder — the
    parent coordinator can merge exports post-hoc if needed.
    """

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self._turns: list[TurnRecord] = []
        self._next_index = 0

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_turn(
        self,
        *,
        assistant_msg: Message,
        user_msg: Message | None = None,
        tool_calls: list[ToolCall] | None = None,
        tool_results: list[ToolResult] | None = None,
        reasoning_text: str | None = None,
    ) -> TurnRecord:
        """Append a turn. Returns the record so the caller can attach tool
        results later (see ``attach_tool_results``)."""
        record = TurnRecord(
            turn_index=self._next_index,
            timestamp=time.time(),
            assistant_msg=assistant_msg,
            user_msg=user_msg,
            tool_calls=list(tool_calls or []),
            tool_results=list(tool_results or []),
            reasoning_text=reasoning_text,
        )
        self._turns.append(record)
        self._next_index += 1
        return record

    def attach_tool_results(self, results: list[ToolResult]) -> None:
        """Attach tool results to the most recent turn.

        Agent loop records the assistant message when it's finalized (before
        tool execution), then tool results arrive asynchronously. Calling
        this after ``executor.execute_tool_calls`` ensures the turn is
        complete.
        """
        if not self._turns:
            return  # no turn to attach to — silently drop
        self._turns[-1].tool_results = list(results)

    @property
    def turns(self) -> list[TurnRecord]:
        return list(self._turns)

    def __len__(self) -> int:
        return len(self._turns)

    # ------------------------------------------------------------------
    # Exports
    # ------------------------------------------------------------------

    def export_sharegpt(self) -> dict[str, Any]:
        """Export as a ShareGPT conversation entry.

        Schema:
        ```
        {
            "id": session_id,
            "conversations": [
                {"from": "human", "value": "..."},
                {"from": "gpt", "value": "...", "tool_calls": [...]},
                {"from": "tool", "value": "..."},
                {"from": "gpt", "value": "..."},
                ...
            ]
        }
        ```
        """
        conversations: list[dict[str, Any]] = []
        for turn in self._turns:
            if turn.user_msg is not None:
                conversations.append({
                    "from": "human",
                    "value": turn.user_msg.get("content", "") or "",
                })
            gpt_msg: dict[str, Any] = {
                "from": "gpt",
                "value": turn.assistant_msg.get("content", "") or "",
            }
            if turn.tool_calls:
                gpt_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "name": tc.name,
                        "arguments": tc.arguments,
                    }
                    for tc in turn.tool_calls
                ]
            if turn.reasoning_text:
                gpt_msg["reasoning"] = turn.reasoning_text
            conversations.append(gpt_msg)
            for result in turn.tool_results:
                conversations.append({
                    "from": "tool",
                    "name": result.name,
                    "value": result.content,
                    "is_error": result.is_error,
                })
        return {"id": self.session_id, "conversations": conversations}

    def export_rl(self) -> list[dict[str, Any]]:
        """Export as RL transition tuples.

        Each turn with tool calls becomes one transition:
        ``{state, action, reward, next_state}``. ``state`` is the
        conversation up to that turn (as a list of messages). ``action`` is
        the tool calls + assistant text. ``reward`` is ``None`` until
        annotated by the caller (e.g. benchmark runner). ``next_state``
        appends the tool results to state.

        Final-answer turns (no tool calls) emit a terminal transition with
        ``action={"final_answer": text}`` and ``next_state=None``.
        """
        transitions: list[dict[str, Any]] = []
        state: list[Message] = []
        for turn in self._turns:
            if turn.user_msg is not None:
                state = state + [turn.user_msg]
            if turn.tool_calls:
                action: dict[str, Any] = {
                    "text": turn.assistant_msg.get("content", "") or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "name": tc.name,
                            "arguments": tc.arguments,
                        }
                        for tc in turn.tool_calls
                    ],
                }
                next_state = state + [
                    turn.assistant_msg,
                    *[r.to_message() for r in turn.tool_results],
                ]
                transitions.append({
                    "state": list(state),
                    "action": action,
                    "reward": None,
                    "next_state": list(next_state),
                    "turn_index": turn.turn_index,
                })
                state = next_state
            else:
                # Final-answer turn — terminal
                transitions.append({
                    "state": list(state),
                    "action": {
                        "final_answer": turn.assistant_msg.get("content", "") or "",
                    },
                    "reward": None,
                    "next_state": None,
                    "turn_index": turn.turn_index,
                    "terminal": True,
                })
                state = state + [turn.assistant_msg]
        return transitions

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(
        self,
        path: Path | str,
        fmt: Literal["sharegpt", "rl"] = "sharegpt",
    ) -> Path:
        """Write the trajectory to ``path`` as JSON. Returns the final path.

        The parent directory is created if missing. Existing files are
        overwritten — caller is responsible for picking a unique filename
        (the benchmark runner uses ``{session_id}_{timestamp}.json``).
        """
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if fmt == "sharegpt":
            payload: Any = self.export_sharegpt()
        elif fmt == "rl":
            payload = self.export_rl()
        else:
            raise ValueError(f"Unknown format {fmt!r}; use 'sharegpt' or 'rl'")
        out_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )
        return out_path


def _json_default(obj: Any) -> Any:
    """Fallback serializer for dataclasses / ToolCall / ToolResult."""
    try:
        return asdict(obj)
    except TypeError:
        return str(obj)
