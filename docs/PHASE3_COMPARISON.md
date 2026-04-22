# Phase 3 — comparison against reference codebases

> Scope: `src/learning/`, `src/eval/`, `src/memory/skill_index.py`, `src/ui/api.py`,
> + cross-cutting changes into `src/core/agent_loop.py`. **3024 LOC across
> 13 files**, 16 commits on `feature/phase-3-learning`, merged to `master`.
> **549 tests passing** (Phase 2 ended at 395, +154).
>
> Written 2026-04-22. This report reads the actual source of the
> reference projects — paths below are real and the `file:line` citations
> are grep-verified against clones taken 2026-04-22.

**Reference codebases read for this report:**

| Project | Language | LOC (rough) | Clone command |
|---------|----------|-------------|---------------|
| **Hermes Agent** | Python | 100K+ | `git clone https://github.com/NousResearch/hermes-agent` |
| **learn-claude-code** | Python | ~5K | `git clone https://github.com/shareAI-lab/learn-claude-code` |
| **claude-code-source-code** | Markdown only | n/a | `git clone https://github.com/sanbuphy/claude-code-source-code` — turned out to be only 5 docs about telemetry & remote-control, NOT a source reimplementation |

Claude Code itself is not open-source, so all CC references here are
inferred from (a) its user-facing SKILL.md format and behavior and (b)
the sanbuphy analysis notes, which cover auth/telemetry and don't
touch learning/eval. We do not make file-level claims about CC.

---

## Domain 1 — Trajectory recording

### Where the recording lives

| Project | File | LOC | Design |
|---------|------|-----|--------|
| **Yigent** | `src/learning/trajectory.py` | 258 | `TrajectoryRecorder` class + `TurnRecord` dataclass. Per-turn in-memory append. Two explicit export methods (`export_sharegpt()` → dict, `export_rl()` → list of transition tuples) + `.save(path, fmt)` for disk. |
| **Hermes** | `agent/trajectory.py` + inline in `run_agent.py` | 56 (helpers only) + ~150 inline | A 56-line helper file (`save_trajectory()` writes one JSONL line, `convert_scratchpad_to_think()` rewrites tags) plus the actual construction in `AIAgent._convert_to_trajectory_format()` which stays embedded in the 11K-line monolith because "batch_runner.py calls agent._convert_to_trajectory_format" (trajectory.py:3-5). |
| **learn-claude-code** | None | 0 | No trajectory concept. `s01_agent_loop.py` just appends to `messages[]`. |

### Trajectory format

Both Hermes and Yigent emit **ShareGPT JSON**. The schemas are
field-identical for the common fields:

```python
# Yigent src/learning/trajectory.py:105-140
{"id": session_id, "conversations": [
    {"from": "human", "value": "..."},
    {"from": "gpt", "value": "...", "tool_calls": [...], "reasoning": "..."},
    {"from": "tool", "name": "...", "value": "...", "is_error": false},
]}

# Hermes agent/trajectory.py:44-49
{"conversations": [...], "timestamp": "...", "model": "...", "completed": true}
```

Hermes adds `completed:bool` at the top level (routes success vs.
failure to `trajectory_samples.jsonl` vs `failed_trajectories.jsonl`);
Yigent doesn't — we let the benchmark runner or the user decide what
"completed" means per-session.

### RL export

- Hermes: **no RL format in the base repo**. The `hermes-agent-self-evolution` sibling repo wires ShareGPT into Atropos separately.
- Yigent: `src/learning/trajectory.py:143-201` emits `{state, action, reward, next_state, turn_index, terminal}` tuples inline. `reward=None` sentinel forces the caller to annotate before downstream consumption.

This is a modest **advantage** for Yigent — no glue layer needed for GRPO/DPO use.

### Recorder call-site pattern

Hermes inlines the recording inside `AIAgent.run_conversation` —
every turn mutates `self._trajectory`. The recorder is not swappable.

Yigent injects via an optional constructor parameter:

```python
# src/core/agent_loop.py:82-84
async def agent_loop(
    conversation: list[Message],
    ...
    trajectory: object | None = None,  # duck-typed
    ...
):
```

`trajectory is not None` check at three sites (lines 365, 404, 430)
gates the recording; any object with `record_turn` / `attach_tool_results`
methods works. Tests use a spy; benchmark uses `TrajectoryRecorder`;
`None` disables the feature. **Hermes cannot turn off trajectory
recording** — it's always on, always writing.

---

## Domain 2 — Periodic nudge

This is the most interesting architectural divergence. Both projects
end up writing to memory periodically, but via **completely different
mechanisms**.

### Yigent approach — aux-LLM one-shot

| Aspect | Location |
|--------|----------|
| Trigger | `src/core/agent_loop.py:441-445` bucketed `count // interval` crossing |
| Engine | `src/learning/nudge.py:60-129` NudgeEngine.maybe_nudge() |
| Prompt | `src/learning/nudge_prompt.py:22-49` (NUDGE_SYSTEM_PROMPT constant) |
| LLM call | Single aux stream in `nudge.py:131-148` — uses the session's cheap aux provider, non-streaming internally |
| Output | Structured JSON `{topic, hook, body}` or literal `null` — parsed by `_parse_response` (nudge.py:167-204) |
| Sink | Directly calls `MarkdownMemoryStore.write_topic` + `.record_index_entry` (nudge.py:116-118) |
| Safety | `CircuitBreaker(threshold=3)` from `src/context/circuit_breaker.py` — 3 consecutive aux failures disable nudge for session |

**Cost per nudge**: 1 aux LLM call (small model, few hundred tokens).

### Hermes approach — background fork with full tool access

| Aspect | Location |
|--------|----------|
| Trigger | `run_agent.py:8562-8567` — `_turns_since_memory >= _memory_nudge_interval` (default 10 USER turns, not tool calls) |
| Engine | `run_agent.py:2796-2884` `_spawn_background_review()` spawns a thread |
| Prompt | `run_agent.py:2761-2793` three hard-coded prompts (`_MEMORY_REVIEW_PROMPT`, `_SKILL_REVIEW_PROMPT`, `_COMBINED_REVIEW_PROMPT`) |
| LLM call | **A whole new `AIAgent` instance** with max_iterations=8, same model as main session, running its own tool-use loop |
| Output | The forked agent uses the normal `memory` and `skill_manage` tools — writes happen via normal tool execution |
| Sink | Shared `_memory_store` / `_skill_store` references handed to the fork |
| Safety | Thread-contained; stdout/stderr redirected to /dev/null (`run_agent.py:2823-2825`) |

**Cost per nudge**: up to 8 main-LLM iterations with tool calls — an order of magnitude more expensive than Yigent's one-shot aux call.

### Comparison

| Criterion | Yigent | Hermes |
|-----------|--------|--------|
| Cost per trigger | ~1 aux-LLM call | up to 8 primary-LLM iterations |
| Output quality | Constrained by aux model | Benefits from main-LLM reasoning and its own tool access |
| Failure isolation | Circuit breaker per session | Thread-per-nudge (leaks on aux crash) |
| Extensibility | JSON-schema coupled | Any tool the main agent has |
| Blast radius | Can only write one topic+hook+body | The forked agent can **call any write tool** — memory, skill, even bash if enabled |
| Observability | `NudgeResult` dataclass + single log line | Redirected to /dev/null; silent failure |

**Hermes's design is strictly more powerful** (forked agent has full tool
access). **Yigent's design is strictly cheaper and more auditable.** For a
single-user assistant Yigent's trade-off is right; for production where
the model can afford 10× cost on a rare background task, Hermes wins.

**Concrete risk difference**: Hermes's background fork, running with
full tool access including `memory` and `skill_manage`, **could in
principle decide to rewrite an existing memory or skill** if the prompt
nudge leads it there. Yigent's nudge literally cannot — the `NudgeEngine`
only has `MarkdownMemoryStore.write_topic` access, no delete or skill
path. This is a defense-in-depth property worth keeping.

---

## Domain 3 — Skill auto-creation and self-improvement

### What "skill" means in each project

| Project | Definition | Authored by |
|---------|-----------|-------------|
| **Yigent** | A reusable workflow distilled from a successful trajectory — 4+ tool calls, 2+ distinct tools | Auto (aux LLM extracts) |
| **Hermes** | A named invocable command (`/skill-name arg1 arg2`) that the user types or that an agent can call | User (manually authored as SKILL.md) OR auto (via background review agent) |
| **learn-claude-code** | A SKILL.md file in `skills/` loaded at session start | User only (no auto path in any of s01-s12) |
| **Claude Code** | Same as learn-claude-code — user-authored SKILL.md | User only |

### Format (all four use YAML frontmatter + Markdown body — agentskills.io-compatible)

Yigent: `src/learning/skill_format.py:40-95` (Skill dataclass + `parse`/`render`).
Hermes: `agent/skill_utils.py:52-86` (`parse_frontmatter` function) + `agent/skill_commands.py:152-194` (`_load_skill_payload`).
learn-claude-code: `agents/s05_skill_loading.py:74-83` (`SkillLoader._parse_frontmatter`).

All three parse the `---` delimited frontmatter with `yaml.safe_load`
and keep the body as markdown. **Yigent preserves unknown frontmatter
fields via `Skill.extra: dict`** (skill_format.py:54) so round-tripping
through us doesn't lose fields Hermes or CC added. Hermes and
learn-claude-code drop unknown fields.

### Creation gates (auto-creation paths only)

| Project | Gate |
|---------|------|
| **Yigent** | `src/learning/skill_creator.py:123-133` `_passes_gate`: outcome==success AND ≥4 tool calls AND ≥2 distinct tools. Plus `skill_index.find_similar(description, threshold=0.6)` dedup (skill_creator.py:99-106 pre-aux, and :149-155 post-aux using the LLM-produced description). |
| **Hermes** | None explicit — any `_iters_since_skill >= _skill_nudge_interval` check-in (run_agent.py:8843-8845) fires the review. The review agent itself decides whether to save, via natural-language prompt (`run_agent.py:2772-2780`: "Was a non-trivial approach used ..."). |
| **learn-claude-code** | n/a — no auto creation. |

**Yigent's gate is numeric and enforceable outside an LLM**. Hermes's is
an instruction to the forked agent. Both can fail; Yigent's failure
mode is "false negative" (no skill written when one should have been),
Hermes's is "false positive" (a trivial session generates a skill).

### Self-improvement

| Project | Mechanism | Location |
|---------|-----------|----------|
| **Yigent** | `improvement_ratio < 0.8` (new run uses <80% of old expected_tool_count) → aux LLM rewrites Steps → version bump → old version archived | `src/learning/skill_improver.py:66-115` |
| **Hermes** | Only if the review agent chooses to update an existing skill via `skill_manage` tool (same background-review mechanism as creation) | `run_agent.py:2788-2791` in the SKILL_REVIEW_PROMPT |
| **learn-claude-code** | n/a | — |
| **hermes-agent-self-evolution** | DSPy/GEPA optimizer runs on successful trajectories; MIPROv2 prompt optimization; performance_history tracked per version | not in this repo clone |

### Rollback

Only Yigent has an explicit rollback mechanism:

- `src/learning/skill_improver.py:157-195` `rollback_to_previous(slug)`:
  - Moves current live file into `.history/` (reversible)
  - Picks highest-versioned archive, restores with real slug (fixes
    the round-trip slug bug at `skill_improver.py:150-162`)
  - Deletes the restored archive so repeat-rollback doesn't loop
- Archive layout: `skills/.history/{slug}_v{n}.md` — git-friendly,
  diff-visible, grep-able.

Hermes has no rollback. If the review agent corrupts a skill, the old
version is gone.

---

## Domain 4 — Skill index / registry

| Project | File | Design |
|---------|------|--------|
| **Yigent** | `src/memory/skill_index.py` (184 LOC) | `SkillIndex` class with `rebuild()` / `search(query, k)` / `load(slug)` / `register(skill)` / `find_similar(description, threshold)`. Matching: token-set Jaccard on name+description+tags (`skill_index.py:95-108`). |
| **Hermes** | `agent/skill_commands.py` (508 LOC) | `scan_skill_commands()` / `get_skill_commands()` / `resolve_skill_command_key()` — index of **invokable commands by /prefix name**, not a semantic matcher. Matching is exact command lookup (`skill_commands.py:410-427`). |
| **learn-claude-code** | `agents/s05_skill_loading.py:59-104` SkillLoader | `get_descriptions()` dumps one line per skill into the system prompt. No search — all skills are in-context all the time. |
| **Claude Code** | Similar to learn-claude-code based on SKILL.md convention | — |

**Takeaway**: Yigent is the only project in this list that does
**similarity-based skill retrieval** vs. either exact-match (Hermes) or
load-everything-upfront (learn-claude-code, CC).

For a 10-skill registry, all three approaches work. For a 100-skill
registry, Hermes still fine (commands are keyed); learn-claude-code
wastes context; Yigent scales with k (default k=3, returns top matches).

**What Yigent doesn't have**: embeddings. Jaccard treats `"RAG"` and
`"retrieval augmented generation"` as unrelated. Hermes doesn't need
this (command names are canonical). For Yigent to hit real 100+ skill
scale, sentence-transformers + FAISS would be needed — noted as future
work in `src/memory/skill_index.py:7-10` module docstring.

---

## Domain 5 — Evaluation benchmark

### Built-in benchmarks

| Project | File(s) | Task count | Domains |
|---------|---------|-----------|---------|
| **Yigent** | `src/eval/benchmark.py` (520) + `src/eval/reporter.py` (78) + `src/eval/judges/rule_checks.py` (378) + `src/eval/judges/llm_judge.py` (189) + `configs/eval_tasks.yaml` (94) | 12 | coding / research / data_analysis / file_management |
| **Hermes** | `environments/benchmarks/{terminalbench_2, yc_bench, tblite}/` | Wraps external benchmarks, doesn't define tasks itself | terminal use, YC interview qs, specific model evals |
| **learn-claude-code** | None | — | — |

### Per-task self-contained fixtures

Yigent's `src/eval/benchmark.py:_prepare_workspace` (lines 157-224)
synthesizes all fixtures per task:
- CSV with injected anomalies / groupby columns (lines 165-177)
- Log files with mixed INFO/ERROR (lines 178-187)
- Mixed file extensions (lines 188-192)
- Duplicate content files (lines 193-199)
- `buggy.py` with IndexError on line 15 (lines 202-221) — the line-15
  alignment is literal, verified by `_prepare_workspace` returning a
  file where `return items[2]` is on line 15 exactly

Hermes's benchmark runners shell out to external test harnesses
(`environments/benchmarks/terminalbench_2/run_eval.sh`) — they depend
on the benchmark's own fixture setup. **Zero setup required on the
Yigent side.**

### Scoring

| Project | Rule layer | LLM judge | Weighting |
|---------|-----------|-----------|-----------|
| **Yigent** | 11 check types (code_executes / has_statistics / duplicates_removed / etc.), plus 3 keyword-coverage checks added post-merge. Each returns `RuleResult(passed, score, reason, check_name)` | `JudgeResult(correctness, efficiency, robustness, reasoning)` on 0–10, temperature=0, **one retry** on JSON parse failure. Aux-error returns zero score. | `0.4 × rule + 0.6 × judge` configurable via `scoring:` in yaml |
| **Hermes** | Delegates to external bench | Delegates to external bench | — |

Rule check files:
- `check_code_executes` — `src/eval/judges/rule_checks.py:88-98`
- `check_bug_fixed` — requires ≥2 executor calls, final clean (`rule_checks.py:115-131`)
- `check_files_organized` — ≥2 subdirs with files (`rule_checks.py:173-184`)
- `check_duplicates_removed` — SHA-1 scan workspace, no dup content (`rule_checks.py:205-224`)
- `check_content_http_libs` / `check_content_rag_architectures` / `check_comparison_vllm_sglang` — keyword-coverage variants added in the post-merge "weak rules" fix (`rule_checks.py:167-220`)

Judge retry logic:
- `src/eval/judges/llm_judge.py:73-99` — loop `for attempt in (1, 2)`, on
  parse failure increments attempt and retries. After 2 failures, returns
  `JudgeResult.zero("unparseable response after retry")` so the rule
  channel can still produce a final score.

### Metrics

| Metric | Yigent | SWE-bench | Aider bench | Hermes (external) |
|--------|--------|-----------|-------------|-------------------|
| Per-domain completion rate | ✓ (`benchmark.py:387-388`) | Overall only | Overall only | Bench-dependent |
| Avg steps per task | ✓ | — | Token counts only | Bench-dependent |
| Recovery rate (had_errors ∧ passed) | ✓ (`benchmark.py:398-402`) | — | — | — |
| **Cross-domain consistency** — `1 − variance/0.25` | ✓ (`benchmark.py:391-396`) | — | — | — |
| Skill creation count | ✓ | — | — | — |

The **consistency_score** formula (`benchmark.py:391`) is specific to
Yigent — `statistics.pvariance(per_domain_completion.values())` divided
by `0.25` (max binomial variance), subtracted from `1.0` and clamped to
`[0, 1]`. A monoculture specialist scoring 100% on coding and 0%
everywhere else gets `consistency=0`. A well-rounded agent scoring 70%
on all four domains gets `consistency>0.95`. This directly measures
"generalization to untrained scenarios" from the role JD.

### Post-merge eval fixes worth noting

Three issues surfaced on the first live benchmark run and were fixed:

1. **Judge template braces** (`src/eval/judges/llm_judge.py:86-90` and commit `e0727a7`) — switched from `str.format()` to literal `.replace()` so the default judge prompt's example JSON `{"correctness": N, ...}` doesn't trigger `KeyError`.
2. **bash/python_repl cwd drift** (`src/tools/coding.py:70-90`, `src/tools/interpreter.py:35-55`, commit `f85a21d`) — both tools now honor `ctx.working_dir` and fail fast on stale dirs. This was the root cause of a benchmark run that moved `CLAUDE.md` and `README.md` into a stray `md/` directory via `mv *.md md/` in the repo root (instead of in the per-task workspace).
3. **Fixture-prompt mismatch** (`src/eval/benchmark.py:201-245`) — splits `buggy.py` content per task.check: `refactor_quality` gets a process-only fixture, `bug_fixed` gets a pick_third fixture with `return items[2]` literally on line 15.

None of these are reference-project-specific — they're Windows platform
issues + eval harness polish. But they reveal something about scale:
**Hermes and learn-claude-code don't have a comparable benchmark so
they can't catch these bugs** until they ship to real users. Yigent
can.

---

## Domain 6 — HTTP API (FastAPI + SSE)

| Project | Endpoints | Streaming | Session model |
|---------|-----------|-----------|---------------|
| **Yigent** | `POST /chat`, `GET /status`, `GET /trajectory/{id}` | SSE one-frame-per-Event | Fresh agent stack per request (clean isolation, ~200ms overhead) |
| **Hermes** | `gateway/run.py` (FastAPI, OpenAI-compat `/v1/chat/completions`) | SSE + websocket | Persistent agent stack guarded by asyncio.Lock per session |
| **learn-claude-code** | None | — | — |
| **Claude Code** | None (CLI only) | — | — |

Yigent session model is in `src/ui/api.py:83-111` (`SessionRegistry`).
Lives in memory; reset on server restart. Documented limitation.

Hermes's gateway is in `gateway/run.py` (~1000 LOC) and wraps the agent
in an OpenAI-compat facade — if you point the OpenAI SDK at Hermes, it
"just works." We don't do this; our `/chat` uses our own message schema.
Adding OpenAI compat is a future choice.

---

## Cross-cutting observations

### 1. Monolith vs. module split

Hermes's `run_agent.py` is **11,700 lines** in a single file containing
`AIAgent` class with ~200 methods. Nudging, skill review, tool execution,
provider fallback, compression, trajectory — all inline. The
`agent/*.py` files are helpers called from this monolith.

Yigent has `src/core/agent_loop.py` at 498 lines and every feature is
its own module. Testability is a direct consequence: 549 tests, each
targeting a small surface. Hermes's `tests/run_agent/test_run_agent.py`
has to set up the full agent just to test one code path.

This isn't an abstract preference. When we hit the `__parse_error__`
sentinel bug, we changed `anthropic_compat.py`, `streaming_executor.py`,
and a test file — ~50 LOC diff, 7 new tests, commit landed in an hour.
A comparable fix in Hermes would touch `run_agent.py` and require a
full agent spin-up for the test.

### 2. Cost asymmetry in learning triggers

Hermes's "periodic review" is an **8-iteration agent fork per trigger**.
At a default of every 10 user turns + every 10 tool iterations, this
can easily double or triple a session's total LLM cost on a long task.
Their config exposes a way to disable it (`_memory_nudge_interval=0`,
`_skill_nudge_interval=0`) but the default-on is aggressive.

Yigent's periodic nudge is **one aux-LLM call**. Aux is explicitly the
cheap model (DeepSeek-Chat, Haiku, etc.). Cost is negligible and it
stays that way at 15-turn intervals.

Design tension: **cheaper** means **less capable** (our nudge can't
invoke arbitrary tools). For our use case — a single-user harness
targeting research-ops demonstrations — cheaper is right.

### 3. The eval harness is the most load-bearing single component

Yigent has 12 tasks / 4 domains / 3 difficulty, ~20 minutes on a
reasoning model. Hermes wraps 3 external benchmarks (TerminalBench,
YC-bench, tblite), each of which is its own several-hour run.
learn-claude-code has none.

**Scale difference that matters**: Yigent's 12-task bench catches a
Windows cwd regression (bash tool moved repo files to `md/`) within
one run. Hermes can't catch that kind of cross-platform bug in a
benchmark — none of their external benchmarks test cwd semantics.

We can't match Hermes on "can we run SWE-bench?" (they can, we can't).
We beat Hermes on "can we catch a harness bug before it ships?" — our
12 tasks cover 4 domains × multiple tool types × Windows specifics.

---

## What's missing vs. each reference

### vs. Hermes
- **Embedding-based skill similarity** (our Jaccard will saturate at ~100 skills)
- **Performance-tracked skill versioning** (we have versions + archive but no success-rate dashboard)
- **OpenAI-compatible `/chat/completions`** (our API uses our own Message schema)
- **SWE-bench / TerminalBench wrappers** (we have internal bench only)

### vs. hermes-agent-self-evolution (Hermes's sibling, not cloned)
- **DSPy/GEPA skill optimization** (we have simple `improvement_ratio < 0.8` heuristic)
- **MIPROv2 prompt optimization** for skill bodies

### vs. Claude Code
- **`.claude/agents/` sub-agent definitions** (we have multi-agent in Phase 2b, but no declarative YAML for sub-agents)
- **Officially marketed ecosystem** (CC has skills marketplace vibes; we have a format)

### vs. learn-claude-code (nothing significant missing — ours is strictly more capable)

---

## LOC distribution

```
src/learning/            1368 LOC  (45%)
  trajectory.py           258  [ShareGPT + RL export]
  nudge.py                206  [circuit-broken aux nudge]
  nudge_prompt.py          74  [prompt constant]
  skill_format.py         190  [agentskills.io parser]
  skill_creator.py        329  [2-stage dedup + gate]
  skill_improver.py       311  [version bump + archive + rollback]

src/eval/                1184 LOC  (39%)
  benchmark.py            520  [runner + fixtures + workspace prep]
  reporter.py              78  [markdown output]
  judges/rule_checks.py   378  [14 check functions]
  judges/llm_judge.py     189  [aux scoring with retry]

src/memory/skill_index.py  184 LOC  (6%)  [Jaccard search + dedup]
src/ui/api.py              287 LOC  (9%)  [FastAPI + SSE + sessions]

Cross-cutting:
  agent_loop.py diff       +93 LOC  [trajectory/nudge/skill_creator hooks]
  permission_gate.py diff  +33 LOC  [YOLO circuit breaker]
  tools/{coding,interp}.py +61 LOC  [ctx.working_dir honored]
  anthropic_compat.py diff +105 LOC [parse-error sentinel + forensic log]

Tests: +154 (395 → 549) across 11 new test files
  test_trajectory.py         268
  test_nudge.py              308
  test_skill_creator.py      229
  test_skill_improver.py     264
  test_skill_index.py        211
  test_rule_checks.py        269
  test_llm_judge.py          169
  test_benchmark.py          292
  test_api.py                178
  test_anthropic_tool_parse  235
  test_bash_cwd.py            79
```

---

## Conclusion

Phase 3 delivers the learning loop + eval harness + API layer. Every
code borrowing decision is documented by reference: trajectory format
from Hermes ShareGPT, frontmatter parser logic equivalent to
`agent/skill_utils.py:52-86`, SKILL.md layout compatible with
`learn-claude-code/agents/s05_skill_loading.py:74-83` and
agentskills.io.

Where we diverge from Hermes intentionally:
- Cheaper nudge trigger (one aux call vs. 8-iter fork) with
  **explicitly narrower tool access** as a defense-in-depth property
- Structured JSON schema on nudge output vs. free-form prose
- Version+archive+rollback for skills vs. no history at all
- Internal 4-domain bench with cross-domain consistency vs. external
  benchmark wrappers

Where we're strictly behind:
- No embedding similarity (simpler Jaccard substitute)
- No DSPy/GEPA skill optimization (simpler heuristic)
- No OpenAI-compat API facade
- Smaller test scale (12 tasks vs. Hermes's thousands via external benches)

The trade-off profile is consistent: **Yigent targets single-user
demonstrable harness quality; Hermes targets production throughput and
Nous's ecosystem of trained tool-use models**. Neither is strictly
better — they're different products. Knowing that is itself the
outcome of this comparison.

---

## References

- Hermes Agent: https://github.com/NousResearch/hermes-agent (main, commit 2026-04-22)
- learn-claude-code: https://github.com/shareAI-lab/learn-claude-code (main, commit 2026-04-22)
- claude-code-source-code: https://github.com/sanbuphy/claude-code-source-code (docs-only, does not contain source)
- agentskills.io: https://agentskills.io
- SWE-bench: https://www.swebench.com/
- Hermes self-evolution (not cloned here): https://github.com/NousResearch/hermes-agent-self-evolution
