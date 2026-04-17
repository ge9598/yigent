"""Plan mode tools: enter_plan_mode (deferred), exit_plan_mode, ask_user."""

from __future__ import annotations

from src.core.types import (
    PermissionLevel,
    ToolContext,
    ToolDefinition,
    ToolSchema,
)

from .registry import register


# ---------------------------------------------------------------------------
# enter_plan_mode — DEFERRED (must be discovered via tool_search)
# ---------------------------------------------------------------------------

async def _enter_plan_mode_handler(ctx: ToolContext) -> str:
    if ctx.plan_mode.is_active:
        return "Plan mode is already active."
    session_id = ctx.session_id or "session"
    ctx.plan_mode.enter(session_id=session_id)
    return (
        "Plan mode activated. Write and execute tools are blocked.\n"
        "Allowed tools during planning: read_file, list_dir, search_files, "
        "web_search, tool_search, ask_user, exit_plan_mode.\n"
        "When your plan is ready, call exit_plan_mode with the plan content."
    )


register(ToolDefinition(
    name="enter_plan_mode",
    description=(
        "Enter plan mode to draft an implementation plan before making changes. "
        "While active, write/execute tools are blocked at the permission layer."
    ),
    handler=_enter_plan_mode_handler,
    schema=ToolSchema(
        name="enter_plan_mode",
        description=(
            "Enter plan mode. Write and execute operations will be blocked "
            "until exit_plan_mode is called. Use this for multi-step tasks "
            "that need a plan before execution."
        ),
        parameters={"type": "object", "properties": {}, "required": []},
        permission_level=PermissionLevel.READ_ONLY,
        timeout=5,
        deferred=True,  # <-- not in initial tool list; discovered via tool_search
    ),
    needs_context=True,
))


# ---------------------------------------------------------------------------
# exit_plan_mode
# ---------------------------------------------------------------------------

async def _exit_plan_mode_handler(ctx: ToolContext) -> str:
    if not ctx.plan_mode.is_active:
        return "Error: plan mode is not active."
    return ctx.plan_mode.exit()


register(ToolDefinition(
    name="exit_plan_mode",
    description=(
        "Exit plan mode. Any plan content buffered via plan_mode.append() is "
        "automatically saved to plans/. Takes no parameters."
    ),
    handler=_exit_plan_mode_handler,
    schema=ToolSchema(
        name="exit_plan_mode",
        description=(
            "Exit plan mode and resume normal execution. If the agent buffered "
            "plan content during planning, it is auto-saved to plans/."
        ),
        parameters={"type": "object", "properties": {}, "required": []},
        # WRITE permission but the plan_mode whitelist exempts it from blocking.
        permission_level=PermissionLevel.WRITE,
        timeout=10,
    ),
    needs_context=True,
))


# ---------------------------------------------------------------------------
# ask_user — prompt the human for input
# ---------------------------------------------------------------------------

async def _ask_user_handler(ctx: ToolContext, question: str) -> str:
    if ctx.user_callback is None:
        return (
            "Error: user interaction is not available in this context "
            "(running headless or in a test)."
        )
    return await ctx.user_callback(question)


register(ToolDefinition(
    name="ask_user",
    description="Ask the user a clarifying question and wait for their reply.",
    handler=_ask_user_handler,
    schema=ToolSchema(
        name="ask_user",
        description=(
            "Pose a question to the user. Use sparingly — only for ambiguities "
            "that genuinely need human input. Returns the user's reply as a string."
        ),
        parameters={
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "Clear, specific question."},
            },
            "required": ["question"],
        },
        permission_level=PermissionLevel.READ_ONLY,
        timeout=300,  # wait up to 5 min for a human
    ),
    needs_context=True,
))
