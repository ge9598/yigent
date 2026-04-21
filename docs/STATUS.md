# Implementation status

Last updated: 2026-04-19. 150 tests passing.

## Phase 1: Core loop + basic tools (Week 1) — ✅ DONE

- [x] `src/core/agent_loop.py` — async generator ReAct loop
- [x] `src/core/streaming_executor.py` — parallel tool execution during model output
- [x] `src/tools/registry.py` — self-registration + ToolSearch + deferred loading
- [x] `src/tools/file_ops.py` — read_file, write_file, list_dir, search_files
- [x] `src/tools/coding.py` — bash execution with timeout and sandboxing
- [x] `src/tools/interpreter.py` — Python REPL in isolated subprocess
- [x] `src/tools/search.py` — web search via API
- [x] `src/core/plan_mode.py` — enter/exit plan mode + permission-level write blocking
- [x] `src/tools/plan_tools.py` — EnterPlanMode (deferred), ExitPlanMode, AskUser
- [x] `src/providers/base.py` — Provider ABC with streaming interface
- [x] `src/providers/deepseek.py` — DeepSeek V3 via OpenAI-compat
- [x] `src/providers/openai_compat.py` — generic OpenAI-compatible provider
- [x] `src/providers/resolver.py` — runtime provider selection from config
- [x] `src/core/iteration_budget.py` — shared budget with allocate()
- [x] `src/core/env_injector.py` — git/schema/dir tree injection per task type
- [x] `src/ui/cli.py` — Rich TUI with streaming output + permission prompts
- [x] `configs/default.yaml` — default config with provider settings

### Post-Phase-1 additions

- [x] `src/providers/anthropic_compat.py` — Anthropic-protocol provider (native Claude + MiniMax `/anthropic`)
- [x] `docs/PROVIDER_COMPARISON.md` — reference comparison vs CCR / LiteLLM / Hermes / OpenClaude / OpenRouter
- [x] `src/ui/slash_commands.py` — SlashDispatcher (Aider-style `cmd_*` introspection; unknown commands intercepted, never leaked to LLM)

## Phase 2a: MVP — context, safety, memory (Week 2) — ✅ DONE

Focus: assembler + 5-layer compression + permission gate + hook system + L0/L1
memory. Multi-agent / router / MCP deferred to Phase 2b.

### Unit 1 — Context assembly + compression

- [x] `src/context/assembler.py` — 5-zone assembly (static / tools hint / env+plan / conversation / reserve), frozen system prompt for prompt cache
- [x] `src/context/engine.py` — 5-layer compression (truncate / dedup / summarize 1/3 / full rewrite / hard truncate) with per-layer breakers
- [x] `src/context/circuit_breaker.py` — per-layer failure counter (default threshold 3)
- [x] `src/context/prompt_cache.py` — frozen-prefix hash for cache identity, `on_fork()` for shared cache (Phase 2b use)
- [x] `src/core/agent_loop.py` — accept optional `assembler=...` (backward-compatible with Phase 1 inline assembly)

### Unit 2 — Safety: permission gate + hook system

- [x] `src/safety/permission_gate.py` — 5-layer chain (schema → tool self-check → plan-mode authoritative → hook → permission level), YOLO mode honored, destructive always blocked
- [x] `src/safety/hook_system.py` — 8 lifecycle events (session_start / pre_tool_use / post_tool_use / pre_compression / post_compression / plan_approved / budget_warning / session_end), sync+async, broken-hook isolation, dotted-path loader
- [x] `configs/hooks.yaml` — empty file with commented examples
- [x] `src/core/streaming_executor.py` — accept optional `permission_gate=...`; falls back to inline Phase 1 logic when absent
- [x] `src/core/agent_loop.py` — fire `session_start`, `session_end`, `budget_warning` lifecycle events via injected hook system

### Unit 3 — Memory: L0 + L1

- [x] `src/memory/working.py` — L0 in-memory dataclass wrapping conversation + todo
- [x] `src/memory/markdown_store.py` — L1 markdown MEMORY.md index + per-topic `.md` files at `~/.yigent/memory/<sha256(cwd)[:8]>/`; Claude-Code-style layout (200-line / 25 KB index cap, LLM-legible, git-friendly); **replaces the earlier SQLite+FTS5 implementation per 2026-04-20 decision**
- [x] `src/tools/memory_tools.py` — list_memory / read_memory / write_memory / delete_memory tools available to the model; `write_memory` is WRITE-level so permission gate asks by default
- [x] `src/ui/cli.py` — wire MarkdownMemoryStore into ToolContext and ContextAssembler; Zone 3 auto-loads MEMORY.md index; `/remember TOPIC: CONTENT` and `/memory [TOPIC]` slash commands for manual access
- [x] `src/core/agent_loop.py` — system prompt instructs model when to call `write_memory` (user preferences, conventions, gotchas) vs. what not to save (ephemeral state, code-derivable facts)

### Unit 4 — Provider hardening

- [x] `src/providers/anthropic_compat.py` — accumulator re-keyed by `tool_use.id` instead of SDK `event.index`; emit `tool_call_complete` on `content_block_stop` (better UX with parallel tools); orphan deltas defensively dropped (MiniMax `/anthropic` resilience)

## Phase 2b: deferred (post-MVP, pre-Phase-3)

Tracked but not in the current sprint. None blocking learning loop / eval.

### Provider-layer hardening

- [x] Credential pool with rotation strategy (Hermes pattern) — `keys: [...]` + `strategy: round_robin/fill_first/least_used/random` + 429/cooldown auto-rotate
- [x] Scenario routing (CCR pattern) — `routes: {default, background, long_context, thinking}` + per-task-type provider/model selection. agent_loop translates env_injector task types (`coding`/`data_analysis`/`file_ops`/`research`) into CCR route keys via `_TASK_TYPE_TO_ROUTE`.

### Multi-agent + router + MCP

- [x] `src/core/capability_router.py` — intent classification + strategy selection
- [x] `src/core/multi_agent.py` — Fork (shared cache), Subagent (independent), TaskBoard (in-memory dict + asyncio.Lock)
- [x] `src/tools/mcp_adapter.py` — MCP stdio + SSE transport + dynamic ToolSchema conversion
  - **Safety:** `MCPServerConfig.default_permission` (read_only | write | execute | destructive) — every tool from a server inherits this level, forcing explicit opt-in for dangerous servers. Default is read_only.
- [ ] `src/memory/skill_index.py` — skill registry + matching (waits for skill_creator in Phase 3)

## Phase 3: Learning + eval + polish (Week 3)

- [ ] `src/learning/nudge.py` — periodic self-evaluation via aux LLM
- [ ] `src/learning/skill_creator.py` — extract successful workflows as SKILL.md
- [ ] `src/learning/skill_improver.py` — iterative skill refinement + rollback
- [ ] `src/learning/trajectory.py` — always-on recording + ShareGPT/RL export
- [ ] `src/eval/benchmark.py` — 4-domain benchmark framework
- [ ] `src/eval/judges/` — LLM-as-Judge + rule-based checks
- [ ] `src/eval/reporter.py` — generate eval report markdown
- [ ] `configs/eval_tasks.yaml` — benchmark task definitions
- [ ] `src/ui/api.py` — FastAPI + SSE server
- [ ] Tests for all core modules
- [ ] README.md for GitHub
- [ ] `docs/EVAL_REPORT.md` with real benchmark numbers
- [ ] Demo video (2-3 minutes)

## Removed from scope (won't do)

- ~~L2 semantic memory (Qdrant)~~ — text-only via L1 FTS5, matching Claude Code and Hermes (which both ship without vector memory). Decision recorded 2026-04-19. Procedural memory is handled by markdown skills (Phase 3 `skill_creator`), not embeddings.
- ~~`docker-compose.yml` Qdrant service~~ — no longer needed.

## Test count timeline

| Milestone | Tests |
|-----------|-------|
| Phase 1 done | 34 |
| + AnthropicCompatProvider | 50 |
| + SlashDispatcher | 72 |
| + Phase 2a Unit 1 (context) | 102 |
| + Phase 2a Unit 2 (safety) | 129 |
| + Phase 2a Unit 3 (memory) | 147 |
| + Phase 2a Unit 4 (provider hardening) | 150 |
| + validate→ValidateResult + ctx | 156 |
| + markdown memory + memory tools + slash | 169 |
| + Phase 2b Unit 1 (credential pool) | 197 |
| + Phase 2b Unit 2 (scenario routing) | 208 |
| + Phase 2b Unit 3 (MCP adapter) | 219 |
| + Phase 2b Unit 4 (capability router) | 230 |
| + Phase 2b Unit 5 (multi-agent) | 250 |
| + auxiliary provider defaults to clone of primary | 257 |
| + reasoning unification (endpoint_quirks + thinking blocks + `<think>` FSM) | 292 |
| + parity remediation Units 0-4 (hooks 8/8 + MCP SSE + 401/402/429 + cross-provider routing) | 322 |
| + parity remediation Unit 5 (stop_reason + provider fallback) | 325 |
| + parity remediation Unit 6 (interruption tombstone + sibling abort) | 329 |
| + parity remediation Unit 7 (real streaming tool dispatch + exclusive serialization) | 333 |
| + parity remediation Unit 8 (Fork output_file + plan-mode CLI approval) | 338 |
| + parity remediation Unit 9 (three-tier thresholds + Hermes summary template + cursor + L2 widen) | 351 |
| + parity remediation Unit 10 (YOLO aux-LLM + plan dynamic allowlist + destructive confirm + shared budget + capability classifier) | 372 |

## Current focus

> Parity remediation complete — all 11 units delivered (plan:
> ~/.claude/plans/idempotent-swinging-flask.md). 372 tests passing.
> Phase 3 (learning loop + eval) starts next: periodic nudge,
> skill auto-creation, trajectory recording, 4-domain benchmark.
