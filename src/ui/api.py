"""FastAPI + SSE server — HTTP facade over agent_loop.

Exposes three endpoints:

- ``POST /chat`` — accepts ``{messages, task_type?, session_id?}`` and
  streams Server-Sent Events, one per Event yielded by agent_loop. Each
  frame's ``event:`` field is the event class name; ``data:`` is the
  JSON-serialized dataclass.
- ``GET /status`` — returns a snapshot of active sessions + budget.
- ``GET /trajectory/{session_id}`` — dumps the recorded trajectory as
  ShareGPT JSON.

v1 limitations (documented):
- Permission gate runs in auto-allow (``yolo_mode=True``). Interactive
  approval over SSE needs a bidirectional channel — deferred.
- In-memory session registry (not persisted across server restarts).
- One agent loop per request; no multi-agent / Fork / Subagent wiring.

The server shares the benchmark runner's pattern of building a fresh
minimal agent stack per request, so multiple parallel /chat requests
don't race on shared state.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import AsyncGenerator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.context.assembler import ContextAssembler
from src.context.engine import CompressionEngine
from src.core.agent_loop import agent_loop
from src.core.config import load_config
from src.core.env_injector import EnvironmentInjector
from src.core.iteration_budget import IterationBudget
from src.core.plan_mode import PlanMode
from src.core.streaming_executor import StreamingExecutor
from src.core.types import Message, ToolContext
from src.learning.trajectory import TrajectoryRecorder
from src.providers.resolver import resolve_auxiliary, resolve_provider
from src.safety.hook_system import HookSystem
from src.safety.permission_gate import PermissionGate
from src.tools.registry import get_registry

logger = logging.getLogger(__name__)

_API_SYSTEM_PROMPT = [
    Message(
        role="system",
        content="You are Yigent, a capable AI agent serving an HTTP API.",
    ),
]


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    task_type: str | None = None
    session_id: str | None = None


class StatusResponse(BaseModel):
    active_sessions: int
    total_sessions: int
    budget_default: int


# ---------------------------------------------------------------------------
# Session registry
# ---------------------------------------------------------------------------


@dataclass
class Session:
    id: str
    recorder: TrajectoryRecorder
    created_at: float = field(default_factory=lambda: 0.0)
    active: bool = True


class SessionRegistry:
    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}

    def create(self, session_id: str | None = None) -> Session:
        sid = session_id or uuid.uuid4().hex[:16]
        import time
        s = Session(
            id=sid,
            recorder=TrajectoryRecorder(session_id=sid),
            created_at=time.time(),
        )
        self._sessions[sid] = s
        return s

    def get(self, sid: str) -> Session | None:
        return self._sessions.get(sid)

    def mark_complete(self, sid: str) -> None:
        s = self._sessions.get(sid)
        if s is not None:
            s.active = False

    def active_count(self) -> int:
        return sum(1 for s in self._sessions.values() if s.active)

    def total_count(self) -> int:
        return len(self._sessions)


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(config=None) -> FastAPI:
    """Build a FastAPI app. ``config`` defaults to ``load_config()``.

    Exposed as a factory so tests can inject a fake config with a mock
    provider.
    """
    app = FastAPI(
        title="Yigent API",
        version="0.1.0",
        description=(
            "HTTP facade for the Yigent agent harness. "
            "v1 — auto-allow permissions, SSE streaming."
        ),
    )
    app.state.config = config or load_config()
    app.state.sessions = SessionRegistry()

    @app.get("/status", response_model=StatusResponse)
    async def status() -> StatusResponse:
        return StatusResponse(
            active_sessions=app.state.sessions.active_count(),
            total_sessions=app.state.sessions.total_count(),
            budget_default=app.state.config.agent.max_iterations,
        )

    @app.get("/trajectory/{session_id}")
    async def trajectory(session_id: str) -> dict[str, Any]:
        s = app.state.sessions.get(session_id)
        if s is None:
            raise HTTPException(status_code=404, detail="session not found")
        return s.recorder.export_sharegpt()

    @app.post("/chat")
    async def chat(req: ChatRequest) -> StreamingResponse:
        session = app.state.sessions.create(req.session_id)

        async def event_stream() -> AsyncGenerator[bytes, None]:
            try:
                async for frame in _run_chat(
                    app.state.config, req, session,
                ):
                    yield frame.encode("utf-8")
            finally:
                app.state.sessions.mark_complete(session.id)

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Session-Id": session.id,
            },
        )

    return app


# ---------------------------------------------------------------------------
# Chat streaming
# ---------------------------------------------------------------------------


async def _run_chat(
    config, req: ChatRequest, session: Session,
) -> AsyncGenerator[str, None]:
    """Build a one-shot agent stack, run agent_loop, and emit SSE frames."""
    provider = resolve_provider(config)
    aux = resolve_auxiliary(config)
    registry = get_registry()

    plan_mode = PlanMode()
    ctx = ToolContext(
        plan_mode=plan_mode, registry=registry,
        config=config, working_dir=Path.cwd(),
    )
    hooks = HookSystem()
    permission_gate = PermissionGate(
        registry=registry, ctx=ctx, hooks=hooks,
        yolo_mode=True,       # v1 — documented auto-allow
        aux_provider=aux,
    )
    executor = StreamingExecutor(registry, ctx, permission_gate=permission_gate)
    compression = CompressionEngine(auxiliary_provider=aux, hook_system=hooks)
    assembler = ContextAssembler(
        system_prompt=_API_SYSTEM_PROMPT,
        plan_mode=plan_mode,
        compression_engine=compression,
        output_reserve=config.context.output_reserve,
        safety_buffer=config.context.buffer,
        hook_system=hooks,
    )
    env_injector = EnvironmentInjector()
    budget = IterationBudget(config.agent.max_iterations)

    conversation: list[Message] = [
        Message(role=m.role, content=m.content) for m in req.messages
    ]

    try:
        async for event in agent_loop(
            conversation=conversation,
            tools=registry,
            budget=budget,
            provider=provider,
            executor=executor,
            env_injector=env_injector,
            plan_mode=plan_mode,
            config=config,
            assembler=assembler,
            hooks=hooks,
            trajectory=session.recorder,
        ):
            frame = _event_to_sse(event)
            if frame:
                yield frame
    except Exception as exc:  # noqa: BLE001
        logger.error("Chat stream failed: %s", exc)
        yield _sse_frame(
            "ErrorEvent",
            {"error": f"{type(exc).__name__}: {exc}", "recoverable": False},
        )


def _event_to_sse(event: Any) -> str:
    """Convert an agent_loop Event dataclass to an SSE frame."""
    name = type(event).__name__
    try:
        payload = asdict(event)
    except TypeError:
        payload = {"repr": repr(event)}
    return _sse_frame(name, payload)


def _sse_frame(name: str, payload: dict[str, Any]) -> str:
    data = json.dumps(payload, ensure_ascii=False, default=str)
    return f"event: {name}\ndata: {data}\n\n"


# ---------------------------------------------------------------------------
# Module-level app + CLI entry
# ---------------------------------------------------------------------------


app = create_app()


def main() -> None:
    import uvicorn
    uvicorn.run("src.ui.api:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
