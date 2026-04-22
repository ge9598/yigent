# Phase 3 — comparison against reference projects

> Scope: `src/learning/`, `src/eval/`, `src/memory/skill_index.py`, `src/ui/api.py`,
> and the cross-cutting changes into `src/core/agent_loop.py`. **3024 LOC
> across 13 files**, 16 commits on `feature/phase-3-learning`, **549 tests
> passing** (Phase 2 ended at 395).
> Written 2026-04-22, after merge to master.

**Reference projects:**

| Project | Language | Why included |
|---------|----------|--------------|
| **Claude Code (CC)** | TypeScript | Official Anthropic CLI; gold standard for harness design |
| **Hermes Agent** | Python | Sister-philosophy harness; original source of the nudge/skill/trajectory loop |
| **hermes-agent-self-evolution** | Python | Hermes's own DSPy/GEPA-based skill optimization — the "aspirational" ceiling for skill improvement |
| **learn-claude-code** | Python | Pedagogical CC reimplementation; useful as "minimum viable" reference |
| **SWE-bench / Aider benchmark** | Python | Public eval harnesses — useful for calibrating what a benchmark should measure |

---

## Domain 1 — Trajectory recording

### Recording granularity

| Project | Unit of record | Storage | Export formats |
|---------|---------------|---------|----------------|
| Yigent | `TurnRecord` per agent-loop iteration (user?, assistant, tool_calls, tool_results, reasoning_text) | In-memory list; explicit `.save()` | ShareGPT JSON + RL transitions (state/action/reward/next_state) |
| CC | None at API level; only the session transcript in `~/.claude/*.jsonl` | Flat message log | — (no export path) |
| Hermes | Per-turn `Trajectory` objects wrapping `messages[] + metrics{}` | SQLite `trajectories` table | ShareGPT JSON via `export_trajectories.py`; RL format via Atropos integration |
| hermes-agent-self-evolution | Same as Hermes but also attaches `reward: float` from skill optimizer | SQLite + DSPy cache | SFT (ShareGPT) + GRPO |

### Where it plugs into the loop

| Project | Hook point | Overhead per turn |
|---------|-----------|-------------------|
| Yigent | `src/core/agent_loop.py:365` (final answer), `:404` (tool-call turn), `:430` (tool results attach) — three injection sites, all conditional on `trajectory is not None` | 1 dataclass append, zero I/O |
| CC | Transcript auto-flushes every message; no in-memory queue | ~1 fs write per message |
| Hermes | `session_loop.py:run_turn` wraps each turn in a `Trajectory` context manager | 1 SQLite insert per turn |

### Takeaway

Yigent's trajectory recorder is **architecturally closer to Hermes** (in-memory accumulator + explicit export) than CC (which has no export
path at all — you'd have to re-parse the jsonl transcript). The
**RL transitions export** (`reward=None` sentinel for downstream
annotation) matches Hermes-Atropos's format exactly — we can point the
exported file at Atropos without a glue layer. CC's absence of a
training-data export is a real gap for a research-ops agent, which our
JD cares about.

**What we don't have**: reward auto-annotation (hermes-agent-self-evolution
has this via its `GEPA` scorer). Reward stays `None` until the benchmark
runner or an external tool fills it in.

---

## Domain 2 — Periodic nudge

### The core loop

| Project | Trigger | What it does | Sink |
|---------|---------|--------------|------|
| Yigent | Every `N` tool calls (default 15, bucketed by `floor(count/N)`) | Aux LLM sees last 8 turns, emits `{topic, hook, body}` JSON or `null` | L1 markdown memory (`write_topic` + `record_index_entry`) |
| CC | No periodic nudge — only pre-/post-compression hooks | n/a | n/a |
| Hermes | Every `nudge_interval_turns` (default 10); full trajectory, not a slice | Aux LLM emits "insights" as plain prose | SQLite `memory.insights` table with embedding |
| learn-claude-code | s18 periodic-nudge example: just a `print()` call | — | stdout |

### Robustness

| Project | Circuit breaker | Prompt injection resistance |
|---------|----------------|---------------------------|
| Yigent | Reuses `src/context/circuit_breaker.py` (threshold 3) — a single success resets. Per-session, not persistent. | Hard-coded system prompt; aux output MUST parse as JSON or is discarded silently. "Null means skip" is explicit. |
| Hermes | Implicit retry on aux failure; no hard circuit breaker | Prompt lives in a template file; malformed output is logged but still persisted |
| CC | n/a | n/a |

### Takeaway

Yigent's nudge is **more defensive than Hermes** on two axes:
1. **Circuit breaker** — three consecutive aux failures kill the nudge for
   the rest of the session. Hermes keeps retrying and paying the
   round-trip cost indefinitely.
2. **Structured output schema** — `{topic, hook, body}` with explicit
   `null` for "nothing worth saving" lets us reject noise without a
   second LLM call to re-check. Hermes's free-form prose output bloats
   the memory table with tautologies.

Our **antipattern-aware prompt** (explicit "do NOT save ephemeral state,
code-derivable facts, generic advice") mirrors the user's own CLAUDE.md
memory rules — it's a small touch but it's why Yigent's memory table
doesn't fill up with "the user seems to want me to be helpful"-style
garbage.

**What we don't have**: embedding-based dedup (Hermes embeds every
insight and dedups against cosine similarity). We rely on the L1 store's
slug-level dedup (`write_topic` is an upsert by slug), which is weaker
but adequate for a single-user project.

---

## Domain 3 — Skill auto-creation and self-improvement

### Format

| Project | Skill format | Schema |
|---------|-------------|--------|
| Yigent | SKILL.md with YAML frontmatter | `name`, `description`, `version`, `tags[]`, `expected_tool_count`, `body` |
| CC | `.claude/skills/*.md` with YAML frontmatter | `name`, `description`, loose body |
| Hermes | JSON file under `skills/{id}.json` | `name`, `description`, `triggers[]`, `steps[]`, `version`, `embedding[]` |
| hermes-agent-self-evolution | Same JSON + DSPy `Signature` stored alongside | + `signature`, `performance_history[]` |

**agentskills.io compatibility:** Yigent's format is a strict subset of
agentskills.io. Our skills can be dropped into a CC or Cursor project
without edits; CC's skills can be loaded into Yigent (we'll just ignore
any frontmatter fields we don't know, preserved via the `extra: dict`
escape hatch in `Skill.parse()`).

### Creation gates

| Project | Pre-creation gates |
|---------|-------------------|
| Yigent | `outcome == success` AND `≥4 tool calls` AND `≥2 distinct tools` AND `find_similar(description) < 0.6` Jaccard threshold |
| Hermes | `outcome == success` only; dedup via cosine similarity against skill embeddings |
| hermes-agent-self-evolution | Plus: skill candidate must beat existing baseline on at least 2 held-out trajectories |

### Self-improvement

| Project | Improvement trigger | Rollback |
|---------|---------------------|----------|
| Yigent | actual tool count < 0.8 × expected → aux LLM rewrites Steps → version bump. Old version archived to `skills/.history/{slug}_v{n}.md` | `rollback_to_previous(slug)` — picks highest archived version, moves current into history (so rollback is itself reversible) |
| CC | Manual only (user edits `.claude/skills/*.md`); no auto-improver | n/a |
| Hermes | Improvement when a skill's 7-day success rate drops below baseline | Soft rollback: keeps old version, marks new as `quarantined` |
| hermes-agent-self-evolution | DSPy `GEPA` optimizer runs on every successful trajectory; candidate beats baseline → promoted via MIPROv2 | Performance history tracked per version; rollback is automatic via scoring |

### Takeaway

Yigent sits **between CC (no automation) and Hermes-evolution (full DSPy
pipeline)**. The improvement rule is deliberately simple — "20% fewer
tool calls than baseline" is easy to reason about and doesn't require a
scoring harness. DSPy/GEPA is strictly better for production but needed
a whole eval infra Yigent doesn't have yet.

**What we do better than Hermes**: our `skills/.history/{slug}_v{n}.md`
layout is grepable and git-friendly. Hermes's performance_history inside
a JSON blob is diff-unfriendly. CC has no history at all.

**What we don't have**:
- Embedding-based skill matching. Jaccard on tokenized
  (name+description+tags) works for 10-100 skills; above that we'd need
  sentence-transformers + FAISS.
- Skill composition — Hermes has "skill chains" where one skill invokes
  another. Ours are flat.
- Performance-based rollback. Our rollback is manual (benchmark runner
  calls `rollback_to_previous` on detected regression, but there's no
  persistent scoreboard).

---

## Domain 4 — Evaluation benchmark

### Task set

| Project | Scope | Domains × difficulty | Synthetic fixtures |
|---------|-------|---------------------|---------------------|
| Yigent | 12 tasks, 4 domains × 3 difficulty | coding / research / data_analysis / file_management | Per-task workspace seeded by `_prepare_workspace` (CSV / logs / mixed files / duplicates / buggy.py) — zero external dependencies |
| SWE-bench | 2294 tasks, coding only | — | Real GitHub PRs; docker images per task |
| Aider benchmark | 133 tasks, coding only | — | Exercism problems; docker-per-task |
| Hermes eval | "Evals are a todo" (literally the README wording) | — | n/a |
| CC | No built-in benchmark | — | n/a |

### Scoring

| Project | Rule layer | LLM judge | Weighting |
|---------|-----------|-----------|-----------|
| Yigent | 11 check types (code_executes, has_statistics, duplicates_removed, keyword_coverage variants…) with structured `RuleResult(passed, score, reason)` | `JudgeResult(correctness, efficiency, robustness, reasoning)` on 0–10; one retry on JSON parse fail; aux_error returns zero | `0.4 × rule + 0.6 × judge` (configurable) |
| SWE-bench | Git patch apply + `pytest` exit code | — | Pass@1 only |
| Aider | Unit tests pass + lint clean | — | Fraction passing |

### Cross-domain metrics

| Metric | Yigent | SWE-bench | Aider |
|--------|--------|-----------|-------|
| Completion rate (per domain + overall) | ✓ | Overall only | Overall only |
| Avg steps / cost | ✓ | — | Token counts |
| Recovery rate | ✓ (`had_errors ∧ passed`) | — | — |
| **Consistency score** | ✓ `1 − variance(per-domain rates) / 0.25` | — | — |
| Skill creation count | ✓ | — | — |

### Takeaway

Yigent's benchmark is **the only one in this list that explicitly
measures cross-domain consistency** — the JD's "generalization to
untrained scenarios" requirement maps to our `consistency_score`
directly. SWE-bench and Aider are excellent for coding but tell you
nothing about whether the agent can also clean up a file dump or
summarize a research topic. Hermes and CC don't have benchmarks at all.

**What we do well**:
- **Dual-channel scoring.** Rule checks catch hard-failures (compile /
  exec / file state); LLM judge catches soft-failures (truncated
  answer, wrong tool used). Either channel alone misses cases.
- **Self-contained fixtures.** No docker, no external datasets. The
  12-task run takes 20 minutes on a reasoning model with 30 iterations
  budget.
- **Topic-specific rule checks** (new in this phase's post-merge
  fixes). `content_http_libs` / `content_rag_architectures` /
  `comparison_vllm_sglang` require all keywords present — truncated
  answers now fail the rule channel, matching what the LLM judge sees
  independently.

**What we don't have**:
- Scale. 12 tasks is a sanity check, not a research-grade eval. SWE-
  bench's 2294 tasks have statistical power ours can't match.
- Docker isolation. A rogue agent command can still affect the host
  filesystem (we rely on workspace cwd + permission gate; no container
  boundary).
- Comparison to a "no-agent" baseline. We don't know how much of the
  score comes from the agent vs. the underlying model — adding a
  single-shot baseline ("just ask the model once, no tools") would fix
  this.

---

## Domain 5 — HTTP API (FastAPI + SSE)

| Project | Endpoints | Streaming | Permission handling |
|---------|-----------|-----------|---------------------|
| Yigent | POST /chat, GET /status, GET /trajectory/{id} | SSE one-frame-per-Event | Auto-allow (v1 — documented limitation) |
| CC | None (CLI-only) | — | — |
| Hermes | FastAPI `/v1/chat/completions` (OpenAI-compatible) + `/trajectories/*` | SSE + websocket | Full permission gate preserved over websocket |
| Aider | None | — | — |

### Takeaway

The HTTP layer is a straightforward shell. The one design call worth
flagging: **we kept the agent stack rebuilt per-request** (fresh
registry, executor, assembler, ToolContext) rather than sharing. This
costs ~200ms per /chat but means parallel requests can't race on
registry state. Hermes does the opposite — persistent stack, protected
by an asyncio.Lock per session — and gains throughput at the cost of
complex isolation bugs.

For a "personal agent" use case Yigent's choice is right. For a
multi-tenant service Hermes's choice is right.

---

## What this phase reveals

Three observations that go beyond the feature list:

### 1. The learning loop works *at architecture level*, not yet *at data level*

The code is in place: trajectories record, skills auto-create, nudges
trigger. But **`skill_creation_count = 0`** in the latest benchmark
run. Why? Because no single task hit the complexity gate (≥4 tool calls
× ≥2 distinct tools). For the skill ecosystem to actually grow, the
benchmark tasks would need to be **harder** (more tool-use, multi-step
workflows) or the gate would need to be relaxed. This is a **harness
calibration** question, not an implementation question.

### 2. Defensive programming paid off disproportionately

Three bugs surfaced on the first live benchmark run:
- MiniMax's `/anthropic` endpoint drops whitespace in long `tool_use`
  input — we caught this in the `__parse_error__` sentinel and emit
  forensic logs (hex window + per-delta breakdown). **Without these,
  debugging would have taken days.**
- MiniMax 529 overload on YOLO shadow classifier — the **circuit
  breaker** we added turned a session-crippling bug into a single log
  line.
- Windows `bash` cwd drift corrupted the repo once — the tests we
  wrote (`test_bash_cwd.py`) lock in the fix permanently.

CC's ethos of "permission gate is architecture, not a suggestion" shows
up here: all three defenses live at the **boundary layer** (provider
parser, permission gate, tool subprocess) where errors are unavoidable.
Had we put them in prompts or conventions, the first live run would
have silently failed and we'd still be debugging.

### 3. The eval harness is the most *load-bearing* single component

The first benchmark showed 17% completion rate — which turned out to be
3-4 real issues (bash cwd, prompt specificity, fixture drift, weak
rules) compounded. Fixing them took ~200 LOC across 4 files. The
resulting benchmark is now a **reliable instrument** for measuring
harness changes, which is exactly what Phase 3 needed to ship.

CC and Hermes ship without a benchmark and you can see it in their
release notes — improvement claims are qualitative ("now smarter at
X") rather than quantitative. Our eval won't stop us from shipping
regressions, but it'll *tell us* when we do.

---

## Lines of code distribution

```
src/learning/            1368 LOC  (45%)
  trajectory.py           258
  nudge.py                206
  nudge_prompt.py          74
  skill_format.py         190
  skill_creator.py        329
  skill_improver.py       311

src/eval/                1184 LOC  (39%)
  benchmark.py            520
  reporter.py              78
  judges/rule_checks.py   378
  judges/llm_judge.py     189
  judges/__init__.py       19

src/memory/skill_index.py  184 LOC  (6%)
src/ui/api.py              287 LOC  (9%)

agent_loop.py diff         +93 LOC  (trajectory + nudge + skill_creator hooks)
permission_gate.py diff    +33 LOC  (YOLO breaker)
tools/{coding,interp}.py   +61 LOC  (ctx.working_dir honored)
providers/anthropic_compat  +105 LOC  (parse-error sentinel + forensic log)

Total new code:    3024 LOC
Total test code:    2325 LOC across 11 new test files
Test count:          395 → 549 (+154)
```

---

## References

- Claude Code source analysis: https://github.com/sanbuphy/claude-code-source-code
- Hermes Agent: https://github.com/NousResearch/hermes-agent
- Hermes self-evolution: https://github.com/NousResearch/hermes-agent-self-evolution
- Inside Hermes (learning loop): https://mranand.substack.com/p/inside-hermes-agent-how-a-self-improving
- agentskills.io format: https://agentskills.io
- SWE-bench: https://www.swebench.com/
- Aider benchmark: https://aider.chat/docs/benchmarks.html
