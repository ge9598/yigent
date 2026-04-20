"""Five-zone context assembler.

Builds the full message list for one LLM call by composing:

    Zone 1 — Static system prompt (frozen at session init → cache-friendly)
    Zone 2 — Tool schemas (names always; full schemas for activated tools)
    Zone 3 — Environment context (git/schema/dir, refreshed each turn)
    Zone 4 — Conversation (compressed by CompressionEngine if over budget)
    Zone 5 — Reserved for model output (no message — just budget reservation)

The static zone is set once in ``__init__`` and never modified — this is
what makes prompt caching work. Zones 3+4 get re-built every turn.

This replaces the Phase 1 ``_assemble_messages()`` helper in agent_loop.py.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.context.engine import CompressionEngine, estimate_tokens
from src.context.prompt_cache import PromptCache

if TYPE_CHECKING:
    from src.core.env_injector import EnvironmentInjector
    from src.core.plan_mode import PlanMode
    from src.core.types import Message
    from src.memory.markdown_store import MarkdownMemoryStore
    from src.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class ContextAssembler:
    """Composes the 5-zone message list. Owns compression decisions."""

    def __init__(
        self,
        system_prompt: str,
        plan_mode: PlanMode,
        compression_engine: CompressionEngine | None = None,
        memory_store: "MarkdownMemoryStore | None" = None,
        model_context_window: int = 128_000,
        output_reserve: int = 20_000,
        safety_buffer: int = 13_000,
    ) -> None:
        self._base_system_prompt = system_prompt
        self._plan_mode = plan_mode
        self._compression = compression_engine
        self._memory_store = memory_store
        self._model_context_window = model_context_window
        self._output_reserve = output_reserve
        self._safety_buffer = safety_buffer

        # Frozen at init for prompt-cache stability. Plan-mode toggles produce
        # a *different* system message at assemble-time (Zone 3), not a mutation
        # of the base prompt — so this hash stays stable across the session.
        frozen = [self._base_system_prompt_message()]
        self._cache = PromptCache(frozen)

    # -- public --------------------------------------------------------------

    @property
    def cache(self) -> PromptCache:
        return self._cache

    @property
    def usable_budget(self) -> int:
        """Tokens left for Zone 4 (conversation) after reserves."""
        # Subtract a static estimate of zones 1+2 (~1500 tokens for system +
        # tool descriptions). The real number varies as tools activate; this
        # is a conservative floor.
        zones_1_2_estimate = 1500
        return (
            self._model_context_window
            - self._output_reserve
            - self._safety_buffer
            - zones_1_2_estimate
        )

    async def assemble(
        self,
        tool_registry: ToolRegistry,
        env_injector: EnvironmentInjector,
        conversation: list[Message],
        task_type: str,
    ) -> list[Message]:
        """Build the message list for one LLM call. Compresses if needed."""
        messages: list[Message] = []

        # --- Zone 1: static system prompt ---------------------------------
        messages.append(self._base_system_prompt_message())

        # --- Zone 2: tool hint (deferred-tool list) -----------------------
        # The actual tool *schemas* are passed via the provider's `tools=`
        # parameter, not as system text. We just hint about deferred tools
        # the model can discover via tool_search.
        deferred_hint = self._deferred_tool_hint(tool_registry)
        if deferred_hint:
            messages.append({"role": "system", "content": deferred_hint})

        # --- Zone 3: env + plan-mode notice + memory index ---------------
        env_text = await env_injector.get_context(task_type)
        zone3_parts: list[str] = []
        if self._plan_mode.is_active:
            zone3_parts.append(
                "PLAN MODE IS ACTIVE. You MUST NOT use write or execute tools. "
                "Only read-only tools, tool_search, ask_user, and exit_plan_mode "
                "are available."
            )
        if env_text:
            zone3_parts.append(f"[Environment]\n{env_text}")
        memory_index = self._read_memory_index()
        if memory_index:
            zone3_parts.append(
                "[Memory index — use read_memory(topic) to load any entry]\n"
                + memory_index
            )
        if zone3_parts:
            messages.append({"role": "system", "content": "\n\n".join(zone3_parts)})

        # --- Zone 4: conversation, compressed if needed -------------------
        zone_overhead = estimate_tokens(messages)
        target = self.usable_budget - zone_overhead
        if self._compression is not None and estimate_tokens(conversation) > target:
            logger.info(
                "Compressing conversation: %d > target %d",
                estimate_tokens(conversation), target,
            )
            conversation = await self._compression.compress(conversation, target)
        messages.extend(conversation)

        return messages

    # -- internals -----------------------------------------------------------

    def _base_system_prompt_message(self) -> Message:
        return {"role": "system", "content": self._base_system_prompt}

    def _read_memory_index(self) -> str:
        if self._memory_store is None:
            return ""
        try:
            return self._memory_store.read_index()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Memory index read failed: %s", exc)
            return ""

    @staticmethod
    def _deferred_tool_hint(registry: ToolRegistry) -> str:
        deferred_names = [
            name for name, t in registry._tools.items()
            if t.schema.deferred and not registry.is_activated(name)
        ]
        if not deferred_names:
            return ""
        return (
            "Additional tools are available via tool_search: "
            + ", ".join(sorted(deferred_names))
        )
