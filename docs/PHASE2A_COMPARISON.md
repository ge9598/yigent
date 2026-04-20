# Phase 2a — comparison against reference projects

> Scope: `src/context/` + `src/safety/` + `src/memory/` (1185 LOC across 9
> files). Compares Yigent's Phase 2a implementation against five reference
> projects across four domains. Written 2026-04-19, after 150 tests passing.

**Reference projects:**

| Project | Language | Why included |
|---------|----------|--------------|
| **Claude Code (CC)** | TypeScript | Official Anthropic CLI; the gold standard we model |
| **Hermes Agent** | Python | Sister-philosophy harness; we lifted L1 schema from it |
| **OpenClaude** (Gitlawb fork) | TypeScript | Fork of CC for non-Anthropic providers; useful as **negative** reference |
| **Aider** | Python | Mature Python CLI agent; pragmatic, low-ceremony reference |
| **learn-claude-code** | Python | Pedagogical reimplementation of CC; useful for "minimum viable" comparisons |

---

## Domain 1 — Context assembly + Prompt cache

### Where assembly happens

| Project | File | Pattern |
|---------|------|---------|
| Yigent | `src/context/assembler.py:assemble()` | 5 zones, frozen system prompt at init, SHA-256 prefix hash via `PromptCache` |
| CC | `src/utils/queryContext.ts:fetchSystemPromptParts()` + `src/QueryEngine.ts:submitMessage()` | Multi-part system message; lodash `memoize` snapshots git/CLAUDE.md once per conversation |
| Hermes | `agent/prompt_builder.py` | 10 ordered layers (SOUL.md → tool guidance → Honcho → system msg → MEMORY snapshot → USER profile → skills index → context files → timestamp → platform) |
| OpenClaude | inherits CC unmodified | `src/utils/queryContext.ts`, `src/utils/systemPrompt.ts`, `src/context.ts` byte-identical fork |
| Aider | `aider/coders/chat_chunks.py:ChatChunks` | 8 ordinally-stable chunks: system → examples → readonly_files → repo → done → chat_files → cur → reminder |
| learn-claude-code | `agents/s10_team_protocols.py` | Single f-string at module import; one-line system prompt |

### Prompt cache management

**Real `cache_control` breakpoints (Anthropic SDK):**

| Project | Strategy | Breakpoint count |
|---------|----------|------------------|
| CC | Hierarchy: tools → system → messages | 4 (Anthropic max) |
| Hermes | "system_and_3" rolling window: system + last 3 non-system messages | 4 |
| Aider | Cascading priority: examples\|system, repo\|readonly, chat_files | up to 4 |
| **OpenClaude** | **Inherits CC's `cache_control` markers but emits them to OpenAI/DeepSeek/Gemini, which silently ignore them** — dead weight | 0 effective |
| **Yigent** | **None — only SHA-256 hash check (`PromptCache.is_cache_compatible`) for self-diagnostic** | 0 |
| learn-claude-code | None — relies on aggressive forgetting instead | 0 |

### What Yigent does well in this domain

- **Frozen-prefix-at-init** matches CC and Hermes exactly. Plan-mode toggles produce a *separate* Zone 3 system message rather than mutating Zone 1, which preserves the cache prefix byte-for-byte across mode changes.
- **Tool schemas via SDK `tools=` param**, not in system text — matches CC, Hermes, OpenClaude. Aider is the only outlier (no native tool calling).
- **`PromptCache.on_fork()` returns same prefix hash** for forked agents — same intent as Hermes's subagent inheritance pattern (matches `system_and_3` philosophy).

### What Yigent is missing

- **No real `cache_control` emission to provider.** SHA-256 self-check is useful for catching accidental prefix mutation during development, but does **not** produce cache hits on Anthropic/MiniMax. This is the single biggest gap in Phase 2a — porting Hermes's `system_and_3` strategy as a `AnthropicCompatProvider` concern (assembler stays pure) is the obvious fix.
- **`env_injector.get_context(task_type)` is called every turn.** CC and Hermes both freeze env (git status, CLAUDE.md, MEMORY.md) at session start. Per-turn refresh is counterintuitive but it's load-bearing for cache hits. Worth reviewing whether the per-turn re-pull actually changes content; if not, snapshot at session start.

### OpenClaude as cautionary tale

OpenClaude proves the cost of inheriting CC without adapting: the `cache_control` layer is dead weight against OpenAI/DeepSeek/Gemini backends. Yigent's hash-based `PromptCache` is genuinely provider-agnostic precisely because it's self-contained diagnostics, not a wire-format contract. **Lesson:** Don't fork an upstream project's caching contract without owning the wire-format mismatch.

---

## Domain 2 — Compression engine

### Layer count and shape

| Project | Layers | LLM-based? | Cost ordering |
|---------|--------|------------|---------------|
| **Yigent** | **5** (truncate / dedup / summarize 1/3 / full rewrite / hard truncate) | layers 3-4 | cheapest first |
| CC | 5 (`compact.ts` tool-result budget / `snipCompact.ts` / `microCompact.ts` / context collapse / `autoCompact.ts`) | layer 5 | cheapest first |
| Hermes | 4 phases (prune tool results / determine boundaries / structured summary / assemble) | phase 3 | sequential |
| OpenClaude | identical CC fork (verified — same constants `13_000`, `20_000`, `MAX_CONSECUTIVE_AUTOCOMPACT_FAILURES = 3`) | layer 5 | cheapest first |
| Aider | 1 (recursive binary split + summarize) | always | n/a |
| learn-claude-code | 3 (micro-compact / auto-compact / manual `compact` tool) | layer 2 | cheapest first |

### CC's 5 layers — concrete reference

From `src/services/compact/` (~3960 LOC):

| Level | File | Trigger | Mechanism |
|-------|------|---------|-----------|
| 1 | `compact.ts` | tool result > 50K chars | persist to disk, keep 2KB preview in `<persisted-output>` |
| 2 | `snipCompact.ts` | repetitive scaffolding | feeds `snipTokensFreed` into autocompact accounting |
| 3 | `microCompact.ts` + `cachedMicrocompact.ts` | older tool results | **cache-cold**: replace with `'[Old tool result content cleared]'`. **cache-hot**: API `cache_edits` block w/ `cache_reference: tool_use_id` (server-side delete preserving 100K+ cached prefix) |
| 4 | `compact.ts` | ~90% utilization | DB-view pattern; `projectView()` overlays filtered view, **reversible** |
| 5 | `autoCompact.ts` | `tokenUsage >= getAutoCompactThreshold(model)` | child-agent summarization, irreversible |

**Threshold formula** (verbatim from `autoCompact.ts`, verified via OpenClaude mirror):
```ts
const AUTOCOMPACT_BUFFER_TOKENS = 13_000
const MAX_OUTPUT_TOKENS_FOR_SUMMARY = 20_000
autoCompactThreshold = context_window − min(max_output, 20_000) − 13_000
```

For 200K model → fires at ~187K (93.5%). **Yigent's `usable_budget = context_window - output_reserve - safety_buffer - 1500` matches this exactly** (output_reserve=20_000, safety_buffer=13_000).

### Circuit breaker

| Project | Granularity | Threshold | Source of threshold |
|---------|-------------|-----------|---------------------|
| **Yigent** | **per-layer** (independent for L3, L4) | 3 | matches CC by coincidence |
| CC | **single global** per session | 3 | from real incident: 1,279 sessions hit 50+ failures, one hit 3,272, wasting ~250K API calls/day |
| Hermes | none — fail silently to "drop middle turns" | n/a | aux-LLM context-length errors swallowed |
| Aider | multi-model fallback chain (try next model on exception) | n/a | implicit |
| learn-claude-code | none | n/a | n/a |
| OpenClaude | identical to CC | 3 | preserved from upstream |

**Honest call-out:** Yigent's per-layer breakers are **arguably over-engineered**. CC has shipped one global breaker through a real production incident; we have no evidence the per-layer split helps with failure modes we've actually seen. Worth reverting to a single global breaker in a future cleanup unless we see L3 failures behaving differently from L4 failures.

### Summary template

| Project | Sections | Style |
|---------|----------|-------|
| CC | 9 (Primary Request / Key Tech / Files / Errors / Problem Solving / All User Msgs / Pending / Current Work / Next Step) | two-phase: `<analysis>` block stripped, `<summary>` kept |
| Hermes | 7 (Goal / Constraints / Progress (Done/InProgress/Blocked) / Decisions / Files / Next Steps / Critical Context) | structured |
| **Yigent** | **4 (User goals / Decisions / Tools+results / Open questions)** | structured |
| Aider | flowing prose | "User-first-person retelling: 'I asked you…'" |
| learn-claude-code | none | single-prompt "Summarize for continuity" |

### What Yigent is missing

1. **No `_sanitize_tool_pairs()` equivalent.** Hermes injects stub tool_results for removed tool_calls so the conversation protocol stays valid. Dropping an assistant message with a tool_call while keeping later messages that reference it will break strict-mode tool-use APIs (Anthropic, OpenAI native). **This is a real bug waiting to happen** in our L4/L5.

2. **No post-compression recovery.** CC's `runPostCompactCleanup()` re-hydrates: last 5 recently-read files (≤5K each), all activated skills (≤25K total), deferred tools, MCP directives, Plan mode reset. Yigent has nothing — after L4 fires, the agent loses skill activation state.

3. **No iterative recompression.** Hermes stores `_previous_summary` on the compressor instance; next call **updates** rather than regenerates. Yigent always regenerates from scratch — wastes aux-LLM calls and degrades summary quality across multiple compressions.

4. **No cache-aware compression.** CC's `microCompact` `cache_edits` path preserves the 100K+ cached prefix when possible. We have no such heuristic; every compression invalidates whatever cache we'd have had.

### Honest disclosure

The `40K/33K/23K` warn/compress/hard triad in `docs/ARCHITECTURE.md` is **Yigent's own elaboration**, not verbatim from CC. CC has a single auto-compact threshold at `context_window − 33K` (matching our `compress_threshold`); the warn (−40K) and hard-cutoff (−23K) values are our progressive-fallback design.

---

## Domain 3 — Permission gate + Hook system

### Layer chain comparison

| Layer | Yigent (5) | CC (6+) | Hermes (6) | Aider | OpenClaude |
|-------|-----------|---------|------------|-------|------------|
| 1 | schema validation | managed-settings deny (enterprise) | container/sandbox bypass | — | (likely flattened) |
| 2 | tool self-check | rule engine deny→ask→allow (5 settings tiers) | YOLO check | scattered `confirm_ask` | tool self-check |
| 3 | **plan mode (authoritative)** | PreToolUse hook | Tirith Rust scanner | n/a | gRPC user prompt |
| 4 | hook check | mode gate (plan/auto/bypass) | regex pattern engine | n/a | — |
| 5 | permission level | tool self-check | smart approval (aux-LLM scorer) | n/a | — |
| 6 | — | interactive prompt | interactive approval (Once/Session/Permanent) | — | — |

### Hook event count

| Project | Events |
|---------|--------|
| **CC** | **28** (SessionStart/End, UserPromptSubmit, Stop/StopFailure, PreToolUse, PermissionRequest, PermissionDenied, PostToolUse, PostToolUseFailure, SubagentStart/Stop, TaskCreated/Completed, TeammateIdle, PreCompact/PostCompact, Elicitation/ElicitationResult, InstructionsLoaded, ConfigChange, CwdChanged, FileChanged, WorktreeCreate/Remove, Notification, …) |
| Hermes | ~13 (gateway: startup/session/agent/command; plugin: pre/post tool, pre/post LLM, session lifecycle) |
| **Yigent** | **8** (session_start/end, pre/post_tool_use, pre/post_compression, plan_approved, budget_warning) |
| learn-claude-code | small subset (s08 teaches subset of CC) |
| Aider | **0** (`self.event()` is analytics telemetry, not extension points) |
| OpenClaude | likely flattened during fork restructure (no public event registry) |

### Sync/async, multi-hook, isolation

| Property | CC | Hermes | Yigent |
|----------|-----|--------|--------|
| Sync + async | both | both | both |
| Multi-hook per event | yes | yes | yes |
| Execution model | **parallel**, deduped by command/URL | sequential | sequential |
| Conflict resolution | `deny > defer > ask > allow` precedence across parallel results | first non-None wins | any "deny" wins |
| Broken-hook isolation | yes (exit code 0/2/other) | yes ("never crash agent") | yes (exception logged, chain continues) |
| Hook timeout | 600s cmd / 30s prompt / 60s agent | not documented | none |

### YOLO mode

| Project | Mechanism | Guardrails |
|---------|-----------|------------|
| **CC** | **`bypassPermissions` mode + `auto` mode (two-stage classifier)** | bypass still protects `.git`, `.claude`, `.vscode`, `.idea`, `.husky`. Auto mode: stage 1 single-token yes/no (8.5% FP), stage 2 CoT on flagged (0.4% FP). Reasoning-blind by design (sees only user msgs + commands, not model CoT). Escalates to human after 3 consecutive or 20 total denials |
| Hermes | `--yolo` / `HERMES_YOLO_MODE=1` / `/yolo` | flat bypass, but Tirith scanner + regex patterns still active |
| **Yigent** | **`yolo_mode: true` config** | only DESTRUCTIVE level still blocked; **no shadow classifier**, no protected-paths list |
| Aider | `--yes-always` | **none** — fully unconstrained |
| OpenClaude | not explicitly documented | unknown |

### Where Yigent compares well

- **Plan mode authoritative.** Yigent puts plan-mode at layer 3 (above hooks) so user-defined hooks can't unblock plan-mode-forbidden tools. CC achieves the same via mode gate above PreToolUse hook in topology — different placement, same architectural property.
- **Broken-hook isolation matches CC and Hermes** — bad hook is logged, chain continues. This is universal where hooks exist; we're not behind here.
- **`load_hooks_from_config()` dotted-path import + graceful skip on bad refs** matches Hermes's plugin discovery philosophy.

### Where Yigent is behind

- **Hook event count: 8 vs CC's 28.** Missing categories: turn-boundary (UserPromptSubmit, Stop, StopFailure), MCP elicitation (Elicitation, ElicitationResult), filesystem watches (FileChanged, CwdChanged, WorktreeCreate/Remove), task lifecycle (TaskCreated, TaskCompleted), config (ConfigChange, InstructionsLoaded). Many of these are nice-to-haves; the **must-haves** for a Phase 3 learning loop are `UserPromptSubmit` (for nudge timing) and `TaskCompleted` (for skill creation triggers).
- **No shadow classifier for YOLO.** CC's two-stage classifier with reasoning-blind design is the most defended part of their permission system. Our YOLO is honest (we say "we just skip prompts") but for a real demo we want at minimum a regex blocklist matching Hermes's Tirith patterns.
- **No protected-paths list.** Even with YOLO off, we don't refuse writes to `.git/`, `~/.ssh/`, `~/.aws/`. CC and Hermes both ship these as hard blocks. Trivial fix.
- **Sequential hook execution.** CC runs hooks in parallel for latency; our sequential model is simpler but visibly slower if a session registers many hooks. Not urgent.

### Aider as counter-example

Aider has no central gate, no hooks, no plan mode — just scattered `confirm_ask` calls in `aider/io.py` (`check_added_files`, `check_for_urls`, `confirm_ask` with `ConfirmGroup(allow_never=True)` for persistent never-prompt-again). **It works for its scope** (a coding chat assistant where the universe of "dangerous" actions is tiny). Yigent targets general-purpose agents where the action universe is bigger; Aider's procedural approach would not scale.

---

## Domain 4 — Memory (L0 working + L1 episodic)

### L1 storage choice

| Project | L1 backend | Rationale |
|---------|-----------|-----------|
| **CC** | **flat markdown files** under `~/.claude/projects/<project>/memory/`, MEMORY.md as index | "Deliberately boring. No embeddings, no similarity search, no magic." LLM-legible, human-editable, git-friendly |
| **Hermes** | **SQLite + FTS5** at `~/.hermes/state.db`, schema v6 | Production-grade WAL with retry; the schema we copied |
| **Yigent** | **SQLite + FTS5** at `data/sessions.db` | Lifted Hermes schema almost verbatim |
| Aider | **markdown append log** `.aider.chat.history.md` + `.aider.input.history` | Human-readable, no search, no session boundaries |
| learn-claude-code | **JSONL transcript** at `.transcripts/transcript_{ts}.jsonl` | Recovery-only, dumped at compression checkpoints |
| OpenClaude | **JSONL prompt log** at `<claude-config-home>/history.jsonl`, mode 0o600 | Prompt-recall buffer for TUI arrow-up; **NOT** episodic search. Inherits CC's auto-memory dir untouched |

### Schema fidelity vs Hermes (the project we copied)

| Component | Hermes | Yigent | Diff |
|-----------|--------|--------|------|
| `sessions` columns | 25 | 12 | **missing 13**: model_config, system_prompt, end_reason, tool_call_count, cache_read_tokens, cache_write_tokens, reasoning_tokens, billing_provider, billing_base_url, billing_mode, actual_cost_usd, cost_status, cost_source, pricing_version, title (we collapsed title→summary+outcome+tags) |
| `messages` columns | 13 | 11 | **missing 4**: token_count, finish_reason, reasoning_details, codex_reasoning_items |
| `messages_fts` | yes | yes | match |
| `sessions_fts` | **no** | **yes** | we added a virtual table Hermes doesn't have |
| Triggers on `messages` | 3 (insert/update/delete) | 3 (insert/update/delete) | match |
| Triggers on `sessions_fts` | n/a | **0** (manual INSERT in `end_session`) | inconsistent — we have the table but no auto-sync |
| WAL + busy_timeout 1s | yes | yes | match |
| Jittered retry, 15 attempts, 20-150ms | yes | yes | match |
| Checkpoint cadence | every 50 writes (PASSIVE) | every 50 writes (PASSIVE) | match |

### What Yigent does well

- **Schema concurrency policy is faithful to Hermes.** WAL + 1s busy_timeout + jittered retry + 50-write PASSIVE checkpoint — all four match. This is the production-hardened part of Hermes's design and we got it right.
- **`parent_session_id` field reserved** for compression lineage — Phase 2b/3 multi-agent and split-session tracking will use it. Hermes has it; CC tracks lineage via in-band `compact_boundary` UUID messages instead.
- **Graceful degrade** — if SQLite init fails (disk full, permission), agent still runs (`episodic = None`). No reference project documents this explicitly; Hermes assumes the DB always opens.
- **L0 `WorkingMemory` dataclass** wraps `conversation: list[Message]` + `todo: list[str]` with typed methods (`turn_count`, `last_user_text`). Modest elaboration over the plain-list pattern every other project uses.

### What Yigent is missing or inconsistent

1. **`sessions_fts` has no INSERT/UPDATE/DELETE triggers.** We INSERT manually in `end_session` but never UPDATE on session edits or DELETE on session removal. Either add triggers (3 more) or **drop `sessions_fts` entirely** to match Hermes — given we don't currently have a session-tag-search use case, dropping is the cleaner call.

2. **Missing `tool_call_count` on sessions, `token_count` + `finish_reason` on messages.** All three are cheap to add and directly feed Phase 3's eval benchmark `avg_steps` metric. Should add before Phase 3 starts so historical session data is comparable.

3. **No `cache_*_tokens` fields.** When we add real `cache_control` emission (Domain 1 gap), we'll want `cache_read_tokens` and `cache_write_tokens` to validate the `PromptCache.on_fork()` cache-sharing claim. Add at the same time as the provider-side caching work.

4. **No `prune_sessions / clear_messages / delete_session` primitives.** Hermes ships these; we leave it to the user's `rm`. Not urgent — agent works fine — but a 30-line addition.

### CC's choice is also valid

The most-used agent in the world (CC) chose **flat markdown files + grep**, explicitly rejecting embeddings and FTS. The argument: LLM-driven file selection via the MEMORY.md index pointer table is more legible (model can read MEMORY.md and decide which topic file to open) and human-editable (`git diff` shows real changes). Our SQLite choice is more queryable but opaque — you can't `cat data/sessions.db`.

This isn't a "we chose wrong" — it's that **both choices are defensible** and CC has actually shipped the markdown route to scale. Worth knowing if we ever face the question "should we let users edit memory directly?" — markdown wins instantly.

### OpenClaude clarification

OpenClaude's `src/history.ts` writes `history.jsonl` mode 0o600 with `MAX_HISTORY_ITEMS=100`, `MAX_PASTED_CONTENT_LENGTH=1024`. **This is a prompt-input recall log (closer to Aider's `.aider.input.history`), NOT episodic memory.** Don't conflate. The `~/.claude/projects/*/memory/` auto-memory dir is presumably inherited from upstream CC unmodified — not something OpenClaude added.

---

## Cross-cutting patterns

Things that recur across all 5 reference projects:

1. **Freeze static, mutate tail.** CC, Hermes, Yigent, Aider all freeze the static prefix at session/init and push dynamic content to the end. Only learn-claude-code skips this (no dynamic content at all).
2. **L0 is universally `list[dict]` of role+content.** No one wraps it in elaborate state machines. Yigent's `WorkingMemory` (adding todo + helpers) is reasonable, not excessive.
3. **Cheapest compression layer first.** Truncate > dedup > LLM-summarize > hard cutoff is the universal ordering. Aider is the exception (single LLM-summarize layer, recursion as escape hatch).
4. **First-blocker-wins permission chain.** Every project except Aider evaluates checks in fixed precedence; first BLOCK halts.
5. **Broken-hook isolation everywhere hooks exist.** Universal: log + skip, don't fail the chain.
6. **Per-conversation env snapshot, not per-turn.** This is counterintuitive but load-bearing for cache hits. **Yigent currently violates this** (`env_injector.get_context()` per turn).

## The biggest cross-cutting gap in Yigent

**No real provider-side prompt cache emission.** Three of the five reference projects (CC, Hermes, Aider) emit Anthropic `cache_control: {type: "ephemeral"}` markers; we only do the SHA-256 self-check. For Anthropic and MiniMax `/anthropic` providers, we're paying full input cost on every request despite a frozen prefix. Fix: port Hermes's `system_and_3` strategy into `AnthropicCompatProvider` (assembler stays pure). ~50 lines.

## Top 3 things to steal from the field, ranked by ROI

1. **`_sanitize_tool_pairs()` for compression** (from Hermes) — fix a real bug waiting to happen. ~20 lines in `engine.py`. **Critical for Anthropic strict-mode tool use.**
2. **Provider-side `cache_control` emission** (from Hermes + CC) — concrete cost savings on Anthropic. ~50 lines in `AnthropicCompatProvider`.
3. **`runPostCompactCleanup()` equivalent** (from CC) — re-hydrate skills, recently-read files, deferred tools after L4 compression fires. Otherwise the agent loses state every time we compress. ~80 lines in `engine.py` + a callback hook.

## Top 3 things NOT to steal

1. **CC's 28 hook events.** Most are CC-specific (worktree, MCP elicitation, IDE-integration). Add specific events as concrete need arises (next likely additions: `UserPromptSubmit` and `TaskCompleted` for Phase 3 learning loop), not in bulk.
2. **CC's per-tool `checkPermissions()` callback inside each `Tool.ts`.** Distributed validation is harder to audit. Our `defn.validate()` callable approach is simpler and equivalent for our scale.
3. **mem0's vector-first memory.** We already decided to skip L2. Re-affirmed: CC ships text-only at scale, and our intended Phase 3 demo (skill creation + nudge) doesn't need vector recall. Markdown skills + L1 FTS5 cover the use cases.

## Sources

### Domain 1 — Context assembly + Prompt cache

1. https://hermes-agent.nousresearch.com/docs/developer-guide/prompt-assembly
2. https://hermes-agent.nousresearch.com/docs/developer-guide/context-compression-and-caching (Hermes `system_and_3`)
3. https://platform.claude.com/docs/en/docs/build-with-claude/prompt-caching (Anthropic cache_control spec)
4. https://github.com/Gitlawb/openclaude (queryContext, systemPrompt, context.ts)
5. https://github.com/Aider-AI/aider/blob/main/aider/coders/chat_chunks.py (`ChatChunks` 8 fields)
6. https://github.com/Aider-AI/aider/blob/main/aider/coders/base_coder.py (`add_cache_headers` gate)
7. https://github.com/shareAI-lab/learn-claude-code (s10, s06)

### Domain 2 — Compression

8. https://harrisonsec.com/blog/claude-code-context-engineering-compression-pipeline/ (CC 5-level pipeline + 2026-03-10 incident)
9. https://github.com/Gitlawb/openclaude/blob/main/src/services/compact/autoCompact.ts (verified constants)
10. https://github.com/Aider-AI/aider/blob/main/aider/history.py (`ChatSummary`, max_tokens=1024, recursive split)
11. https://github.com/Aider-AI/aider/blob/main/aider/prompts.py (summary prompt template)
12. https://github.com/shareAI-lab/learn-claude-code/blob/main/docs/zh/s06-context-compact.md (3-layer pedagogical model)

### Domain 3 — Permissions + Hooks

13. https://code.claude.com/docs/en/hooks (28 events, decision values, precedence)
14. https://code.claude.com/docs/en/permissions (modes, rule syntax, 5-tier precedence)
15. https://www.anthropic.com/engineering/claude-code-auto-mode (two-stage classifier)
16. https://github.com/ghuntley/claude-code-source-code-deobfuscation
17. https://github.com/sanbuphy/claude-code-source-code (useCanUseTool, StreamingToolExecutor, rule engine)
18. https://deepwiki.com/NousResearch/hermes-agent/5.4-security-and-command-approval (Tirith, regex engine)
19. https://hermes-agent.nousresearch.com/docs/user-guide/features/hooks (gateway + plugin events)
20. https://github.com/Aider-AI/aider/blob/main/aider/io.py (`confirm_ask`, `ConfirmGroup`)
21. https://github.com/shareAI-lab/learn-claude-code/blob/main/docs/zh/s07-task-system.md
22. https://github.com/shareAI-lab/learn-claude-code/blob/main/docs/zh/s08-background-tasks.md

### Domain 4 — Memory

23. https://code.claude.com/docs/en/memory (CC auto-memory v2.1.59+)
24. https://claudefa.st/blog/guide/mechanics/auto-memory (auto-memory write triggers)
25. https://hermes-agent.nousresearch.com/docs/developer-guide/session-storage (Hermes schema v6, WAL, retry)
26. https://github.com/Gitlawb/openclaude/blob/main/src/history.ts (history.jsonl prompt log)
27. https://github.com/Aider-AI/aider/blob/main/aider/io.py (append_chat_history)
28. https://github.com/Aider-AI/aider/blob/main/aider/utils.py (split_chat_history_markdown)
