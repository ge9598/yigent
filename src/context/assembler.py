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
    from src.safety.hook_system import HookSystem
    from src.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


# Three-tier compression thresholds (Unit 9 — ARCHITECTURE.md §I formula).
# Subtract from model_context_window to get the actual budget.
#
#   warn_offset      → fire budget_warning hook, suggest the user wrap up
#   compress_offset  → run compression engine
#   hard_offset      → emergency hard truncate (layer 5)
#
# Numbers from CC source — picked so a 200K-window model still has 23K of
# breathing room after compression and reserves.
_WARN_OFFSET = 40_000
_COMPRESS_OFFSET = 33_000
_HARD_OFFSET = 23_000

# Ratio caps so small-context models (8K, 16K, 32K) don't end up with negative
# thresholds. For a 16K model, the absolute 33K compress_offset would push the
# threshold to -17K — every assemble() would compress unconditionally, never
# leaving room to write. We cap each offset at a fraction of the window.
# Audit Top10 #8 / B3.
_WARN_RATIO = 0.6      # warn at 40% capacity left
_COMPRESS_RATIO = 0.5  # compress at 50% capacity left
_HARD_RATIO = 0.35     # hard truncate at 65% capacity left


def _scaled_threshold(window: int, absolute_offset: int, ratio: float) -> int:
    """Return ``window - effective_offset``.

    ``effective_offset`` is the smaller of the absolute offset and ``ratio*window``.
    For large windows (e.g. 128K, 200K) the absolute number wins. For small
    windows (8K, 16K) the ratio prevents the threshold from going negative.
    """
    effective = min(absolute_offset, int(window * ratio))
    return window - effective


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
        hook_system: "HookSystem | None" = None,
    ) -> None:
        self._base_system_prompt = system_prompt
        self._plan_mode = plan_mode
        self._compression = compression_engine
        self._memory_store = memory_store
        self._model_context_window = model_context_window
        self._output_reserve = output_reserve
        self._safety_buffer = safety_buffer
        self._hook_system = hook_system
        # One-shot guard so we don't fire budget_warning every turn after
        # crossing the threshold once.
        self._warning_fired_at_turn: int = -1
        self._turn_counter = 0

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
    def warn_threshold(self) -> int:
        """Tokens-used level at which budget_warning fires.

        For ctx_win >= 100K this is ``ctx - 40K``. For smaller windows the
        offset shrinks proportionally so the threshold never goes negative.
        """
        return _scaled_threshold(self._model_context_window, _WARN_OFFSET, _WARN_RATIO)

    @property
    def compress_threshold(self) -> int:
        """Tokens-used level at which compression engine runs.

        For ctx_win >= 66K this is ``ctx - 33K``. For smaller windows the
        offset is capped at ``50% * ctx_win`` (compression triggers when
        half the window is consumed).
        """
        return _scaled_threshold(self._model_context_window, _COMPRESS_OFFSET, _COMPRESS_RATIO)

    @property
    def hard_cutoff(self) -> int:
        """Tokens-used level at which emergency hard truncation kicks in.

        For ctx_win >= 66K this is ``ctx - 23K``. For smaller windows the
        offset is capped at ``35% * ctx_win`` (hard cutoff at 65% capacity).
        """
        return _scaled_threshold(self._model_context_window, _HARD_OFFSET, _HARD_RATIO)

    @property
    def usable_budget(self) -> int:
        """DEPRECATED — use compress_threshold instead. Kept for back-compat
        with callers that still depend on a single-tier number."""
        return self.compress_threshold

    async def assemble(
        self,
        tool_registry: ToolRegistry,
        env_injector: EnvironmentInjector,
        conversation: list[Message],
        task_type: str,
    ) -> list[Message]:
        """Build the message list for one LLM call. Compresses if needed."""
        self._turn_counter += 1
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

        # --- Zone 3: plan-mode notice + memory index (system) -------------
        # Env context now goes into Zone 4 as a prefix on the latest user
        # message (per ARCHITECTURE.md §I-bis _inject_env spec) — putting it
        # in Zone 3 inflated messages[] every turn.
        env_text = await env_injector.get_context(task_type)
        zone3_parts: list[str] = []
        if self._plan_mode.is_active:
            zone3_parts.append(
                "PLAN MODE IS ACTIVE. You MUST NOT use write or execute tools. "
                "Only read-only tools, tool_search, ask_user, and exit_plan_mode "
                "are available."
            )
        memory_index = await self._read_memory_index()
        if memory_index:
            zone3_parts.append(
                "[Memory index — use read_memory(topic) to load any entry]\n"
                + memory_index
            )
        if zone3_parts:
            messages.append({"role": "system", "content": "\n\n".join(zone3_parts)})

        # --- Zone 4: conversation (env-prefixed + compressed) -------------
        # Inject env as prefix to the latest user message (NOT a separate
        # system message — that would inflate messages[] every turn).
        # Falls back to a system message only if no user message exists.
        if env_text:
            conversation = self._inject_env(conversation, env_text)

        # Live token measurement — no more hardcoded zone_1_2 guess.
        zones_pre = estimate_tokens(messages)
        conv_tokens = estimate_tokens(conversation)
        total_tokens = zones_pre + conv_tokens

        # Tier 1 (warn): just notify the user, don't compress yet. Fire once
        # per session per turn-count to avoid spamming.
        if total_tokens >= self.warn_threshold and self._warning_fired_at_turn != self._turn_counter:
            self._warning_fired_at_turn = self._turn_counter
            if self._hook_system is not None:
                await self._hook_system.fire(
                    "budget_warning",
                    used_tokens=total_tokens,
                    warn_threshold=self.warn_threshold,
                    compress_threshold=self.compress_threshold,
                    hard_cutoff=self.hard_cutoff,
                )

        # Tier 2 (compress): run the engine to fit conversation under
        # compress_threshold (minus what zones 1-3 already consumed).
        if total_tokens >= self.compress_threshold and self._compression is not None:
            target = self.compress_threshold - zones_pre
            logger.info(
                "Compressing conversation: total %d ≥ compress %d, target %d",
                total_tokens, self.compress_threshold, target,
            )
            conversation = await self._compression.compress(conversation, target)
            conv_tokens = estimate_tokens(conversation)
            total_tokens = zones_pre + conv_tokens

        # Tier 3 (hard cutoff): emergency truncation. Even if compression ran,
        # if we're STILL over the hard cutoff we keep only the last few turns.
        if total_tokens >= self.hard_cutoff:
            target = self.hard_cutoff - zones_pre
            logger.warning(
                "Hard cutoff: total %d ≥ hard %d, emergency truncation to %d",
                total_tokens, self.hard_cutoff, target,
            )
            if self._compression is not None:
                # Engine's layer-5 (hard truncate) is the right tool — call it
                # by giving an impossibly-tight target so other layers no-op
                # and layer 5 fires.
                conversation = await self._compression.compress(conversation, target)

        messages.extend(conversation)

        return messages

    def _inject_env(
        self, conversation: list["Message"], env_text: str,
    ) -> list["Message"]:
        """Prefix env context onto the most recent user message.

        Mutates a copy, not the original. If there is no user message yet
        (very first turn before user input is appended), append env as a
        standalone system message — back-compat fallback.
        """
        # Find the index of the last user message.
        last_user_idx = -1
        for i in range(len(conversation) - 1, -1, -1):
            if conversation[i].get("role") == "user":
                last_user_idx = i
                break
        if last_user_idx < 0:
            # No user message — fall back to system message.
            return list(conversation) + [
                {"role": "system", "content": f"[Environment]\n{env_text}"}
            ]
        # Build a copy of the user message with env prefix injected.
        out = list(conversation)
        existing = out[last_user_idx]
        existing_content = existing.get("content") or ""
        new_msg: "Message" = {
            **existing,  # preserve role + any other fields
            "content": f"[Environment]\n{env_text}\n\n{existing_content}",
        }
        out[last_user_idx] = new_msg
        return out

    # -- internals -----------------------------------------------------------

    def _base_system_prompt_message(self) -> Message:
        return {"role": "system", "content": self._base_system_prompt}

    async def _read_memory_index(self) -> str:
        if self._memory_store is None:
            return ""
        try:
            # Async wrapper if available (B5/Top10 #10), else fall back to sync
            # for stores that don't implement aread_index.
            if hasattr(self._memory_store, "aread_index"):
                return await self._memory_store.aread_index()
            return self._memory_store.read_index()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Memory index read failed: %s", type(exc).__name__)
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
