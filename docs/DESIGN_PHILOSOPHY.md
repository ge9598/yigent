# Yigent Design Philosophy

> **One-sentence version:** Yigent is a Claude-Code-shaped **harness** that hosts a Hermes-Agent-shaped **learning loop**, with self-evolution (Hermes-Self-Evolution-style L3) as its multi-phase destination.
>
> Written 2026-04-22 after Phase 3 shipped. Expected to evolve as Phase 4 lands.

---

## 1. The split

Yigent splits deliberately into two halves that borrow from different sources:

| Half | Reference | Why this reference |
|---|---|---|
| **Execution harness** | Claude Code (CC) | CC is the gold standard for reliable, predictable agent execution. Plan mode is enforced at the permission layer (not via prompt instructions a model might ignore); tools have deferred loading; 5-layer compression keeps long sessions coherent; hooks make the agent a platform. Every "how do I run the loop safely" question we inherit from CC. |
| **Learning loop** | Hermes Agent + Hermes-Self-Evolution | Hermes's distinguishing feature is **continuous improvement of procedural knowledge** — trajectories get saved, skills get auto-generated, skills get auto-improved, and (at the L3 level) training data feeds back into model fine-tunes. This is **the reason to build Yigent at all** — a harness without a learning loop is just a shell. |

The **direction of travel** is also important:

- Phase 1–2 (done): build the CC-shaped harness.
- Phase 3 (done): add the Hermes-shaped L1/L2 learning loop on top.
- **Phase 4+ (planned): close the L3 loop so trajectories actually train the model that runs the harness.** This is the payoff.

---

## 2. Blood lineage — which subsystem comes from where

Each row says: module → where we took the idea from → what's literally shared with them → what's new in Yigent.

| Yigent module | Reference | Shared with reference | Yigent-specific |
|---|---|---|---|
| `src/core/agent_loop.py` | CC's `query.ts` (785KB, main while-loop — sanbuphy analysis) + `QueryEngine.ts` (SDK lifecycle) / claw-code `rust/crates/runtime/src/conversation.rs` (`AssistantEvent` enum at lines 29-40) | Async generator shape, ReAct cycle, event types (TokenDelta, ToolUse, MessageStop), TurnStartedEvent → Reasoning → Token progression | Injection seams for `trajectory=None` / `learning=None` / `assembler=None` so every feature is swappable |
| `src/core/streaming_executor.py` | CC's `services/tools/StreamingToolExecutor.ts` (sanbuphy analysis + zhuanlan 2022442135182406883) | Real streaming dispatch (tool starts before model finishes emitting), sibling-abort on fatal error, tombstone repair for Ctrl+C | Parse-error sentinel `__parse_error__` (our `anthropic_compat.py` invention, not in CC) |
| `src/core/plan_mode.py` | CC's Plan mode | Plan → Approve → Execute triphasic cycle, permission-layer enforcement, deferred tools | — |
| `src/context/engine.py` + `src/context/assembler.py` | CC 5-layer compression, dynamic thresholds `ctx_win - 40K/-33K/-23K` (zhuanlan 2022443175361388953) | 5-layer shape, per-layer circuit breaker, dynamic thresholds, cache-friendly static zone | Explicit `compression_cursor` bookkeeping persisted across turns |
| `src/safety/permission_gate.py` | CC 5-layer chain + YOLO shadow classifier (bilibili 鱼皮 video) | Schema → self-check → plan-mode → hook → level order; YOLO + aux-LLM fallback | Circuit breaker around the shadow classifier (our addition after MiniMax 529 overloads) |
| `src/safety/hook_system.py` | CC's 8 lifecycle events (learn-claude-code s08) | All 8 events, Python-callable or shell-command hooks, broken-hook isolation | — |
| `src/tools/registry.py` + `src/tools/mcp_adapter.py` | CC deferred loading via ToolSearch + MCP stdio/SSE (learn-claude-code s19) | Names-first, schemas-on-demand; MCP dual transport; `default_permission` per server | Hermes-style self-registration at import time (from Hermes `tools/registry.py`) — this one's a hybrid |
| `src/providers/resolver.py` + `scenario_router.py` | Hermes provider runtime (4-tier precedence) + CCR scenario routing | credential_pool with 4 rotation strategies, per-scenario routes (default/background/long_context/thinking), 401/402/429 differentiation | — |
| `src/memory/markdown_store.py` | CC's `~/.claude/` markdown memory (learn-claude-code s09) + Hermes MemoryManager (`agent/memory_manager.py:83`) | MEMORY.md index + per-topic files layout | Project-hash scoping so same user's different projects stay isolated |
| **`src/learning/trajectory.py`** | Hermes `agent/trajectory.py` (56 lines of helpers — inline in `run_agent.py`) | ShareGPT JSON conversation format (`{from: human/gpt/tool, value}`) | Also emits RL transitions (state/action/reward/next_state) inline — Hermes does this only in its self-evolution sibling repo |
| **`src/learning/nudge.py`** | Hermes background-review prompts (`run_agent.py:2761-2793` `_MEMORY_REVIEW_PROMPT`, `_SKILL_REVIEW_PROMPT`) | Trigger concept: periodically review recent activity to decide what's worth persisting | **Implementation strategy is different** — see §3. Our nudge is one aux-LLM call with structured JSON; Hermes spawns an 8-iteration background AIAgent fork with full tool access |
| **`src/learning/skill_creator.py`** | Hermes's background review + skill_manage tool + agentskills.io format | YAML frontmatter + markdown body, tags, description | Explicit numeric gate (≥4 tool calls, ≥2 distinct tools) enforceable outside an LLM; Hermes gates only via the review-agent's judgment |
| **`src/learning/skill_improver.py`** | Hermes's review-agent + hermes-self-evolution's DSPy/GEPA optimizer (the L3 target) | "Update a skill when a shorter path exists" concept | Numeric improvement ratio (< 0.8 of expected_tool_count); `.history/` archive with rollback; no DSPy yet (that's Phase 4 L3) |
| **`src/memory/skill_index.py`** | Hermes `agent/skill_commands.py` (508 LOC, exact-name-match lookup) + agentskills.io | SKILL.md discovery pattern | Jaccard similarity search (Hermes does exact name match; learn-claude-code loads everything into system prompt) |
| **`src/eval/`** | Nobody | — | **Entirely novel** — Hermes wraps external benches (TerminalBench, YC-bench); CC has none; learn-claude-code has none. Our 4-domain × 3-difficulty + dual-channel (rule + LLM-judge) + cross-domain consistency score is a Yigent contribution |
| `src/ui/cli.py` | CC's Rich TUI | Streaming tokens + tool call panels + permission prompts | — |
| `src/ui/api.py` | Hermes gateway (`gateway/run.py`) | FastAPI + SSE shape | Per-request fresh agent stack (Hermes uses per-session persistent stacks with asyncio.Lock) |

**Reading the table**: the execution half (top rows) is ~90% CC-lineage. The learning half (middle rows, bolded) is ~70% Hermes-lineage but with local variations — we're cheaper and more auditable where Hermes is more capable and more expensive. The eval half (bottom row) is novel.

---

## 3. Two key divergences from the references

### 3.1 Our periodic nudge is one aux-LLM call; Hermes's is an 8-iteration agent fork

This is the most consequential architectural difference in Phase 3. It's worth explaining because it gets re-examined every time a new contributor reads the learning-loop code.

**Hermes (`run_agent.py:2796-2884`, `_spawn_background_review()`):**
- Every N user turns, spawn a full `AIAgent` instance in a background thread
- The fork uses the same main LLM as the session, budgeted to 8 iterations
- The fork has normal access to `memory` and `skill_manage` tools
- Shared memory/skill stores let the fork persist its decisions
- Prompt (`run_agent.py:2772-2780` `_SKILL_REVIEW_PROMPT`): *"Was a non-trivial approach used to complete a task that required trial and error... If a relevant skill already exists, update it with what you learned. Otherwise, create a new skill if the approach is reusable."*
- stdout/stderr redirected to /dev/null — fire-and-forget

**Yigent (`src/learning/nudge.py:60-148`):**
- Every N tool calls (bucketed by `count // interval`), aux LLM sees last 8 turns
- Single streaming call to the cheap auxiliary provider
- Required output shape: `{topic, hook, body}` JSON or literal `null`
- Malformed JSON → silently dropped, no re-try to avoid noise
- Writes through `MarkdownMemoryStore.write_topic()` — **cannot invoke any other tool**
- Circuit breaker: 3 consecutive aux failures disable nudge for rest of session

**Why we chose cheaper over more capable:**

1. **Blast radius**: Hermes's fork has full tool access — in principle it could rewrite arbitrary memory or skills based on a noisy review prompt. Our nudge is structurally incapable of this because it has no tools. Defense-in-depth.
2. **Cost profile**: Hermes's default (every 10 turns + every 10 skill iterations) on a long coding task can 2-3× the session's LLM cost. Our aux-LLM call is cents.
3. **Auditability**: Our nudge emits a single `NudgeResult` log line. Hermes's fork is redirected to /dev/null — you have to read the sqlite memory table to know what happened.

**Why Hermes's design is strictly better for a different use case:**

- A production multi-user agent can amortize the 8-iteration cost across users
- The background fork can catch nuances a JSON schema cannot express
- In high-value enterprise sessions, the extra intelligence pays for itself

For Yigent's positioning — a personal/single-user research harness — cheaper + narrower-tool-access is the right trade. Phase 4 may revisit this if we add a "deep review" mode that explicitly invokes the Hermes-style pattern.

### 3.2 Our benchmark exists; most references' benchmarks don't

Hermes wraps external benchmarks (`environments/benchmarks/{terminalbench_2,yc_bench,tblite}/`). They don't author tasks — they run SWE-bench and the like.

Claude Code has no public benchmark.

learn-claude-code has no benchmark at all.

Yigent's 12-task internal benchmark (`src/eval/benchmark.py` + `configs/eval_tasks.yaml`) is novel. The cross-domain **consistency_score** (`1 − variance(per_domain_rates) / 0.25`) directly measures what the target JD describes as "generalization to untrained scenarios."

This is Yigent's strongest differentiator. The cost of authoring 12 tasks + 11 rule checks + workspace synthesis is ~1000 LOC, and it's the thing that will prove us to someone reviewing the project — because it's measurable.

---

## 4. Self-evolution — the three layers of "learning"

The Hermes ecosystem has three distinct layers of self-improvement, and they're worth naming explicitly because Yigent is at L2 and aiming for L3.

### L1 — Skill sedimentation
**Definition**: when an agent completes a reusable workflow successfully, that workflow becomes a named, loadable artifact.

- Hermes: background review agent decides "this was reusable, save as a skill"
- Yigent Phase 3: `src/learning/skill_creator.py` + `src/memory/skill_index.py` — numeric gate (≥4 tool calls, ≥2 distinct tools) + aux-LLM extraction + Jaccard dedup. ✅ **Done**.

### L2 — Skill refinement
**Definition**: when an existing skill is used multiple times, the system detects when a newer run is better than the skill's current "steps" and updates the skill.

- Hermes: same background review agent picks up "update if better" in the prompt
- Yigent Phase 3: `src/learning/skill_improver.py` — improvement ratio < 0.8 threshold + aux-LLM steps rewrite + version bump + `.history/` archive + rollback. ✅ **Done**.

### L3 — Model self-evolution
**Definition**: the trajectories accumulated by L1/L2 get used to fine-tune (SFT) or RL-optimize the underlying LLM, and the improved LLM gets redeployed as the session model. The agent literally becomes stronger over time.

- hermes-agent-self-evolution (the sibling repo, not cloned here): DSPy optimizer + GEPA scorer + MIPROv2 prompt optimization + Atropos RL integration
- Yigent: ❌ **Phase 4 target**. Trajectory recorder already emits ShareGPT + RL transitions; the plumbing downstream (reward annotator, training runner, model version manager, regression detector) is not yet built.

---

## 5. The L3 plan (Phase 4 architecture)

We're not writing code for this yet — just the architecture so when Phase 4 starts the shape is clear.

### 5.1 What we already have (trajectory export → training-ready data)

`src/learning/trajectory.py` already emits two formats:

- **ShareGPT JSON** (`export_sharegpt()`): for supervised fine-tuning (SFT). Direct drop-in format for most SFT trainers (Axolotl, LLaMA-Factory, trl.SFTTrainer).
- **RL transitions** (`export_rl()`): `{state, action, reward, next_state, turn_index, terminal}` tuples with `reward=None` sentinels. Drop-in for Atropos, trl.GRPOTrainer, or hand-rolled DPO pipelines.

Dataset size at current scale (benchmark generates 12 trajectories per run) is tiny — but the schema is in place and stable.

### 5.2 What Phase 4 needs to build

#### 5.2.1 Reward annotator (`src/learning/reward_annotator.py`)

Takes a `TrajectoryRecorder.export_rl()` output and fills in the `reward=None` fields. Reward sources:

1. **Benchmark judge scores** — the easiest source. Every task from `configs/eval_tasks.yaml` already has a `final_score` in `[0, 10]` from `BenchmarkReport.per_task[i]`. Map to per-turn reward via credit-assignment heuristics (uniform per-turn, last-turn-only, or decay-from-end).
2. **Outcome labels** — `passed / failed / recovered` binary or ternary, broadcast to turns.
3. **Downstream usage signals** — if a skill gets invoked later successfully, retroactively reward the trajectory that created the skill. Requires cross-session reward propagation (complex).

Interface sketch:
```python
class RewardAnnotator:
    def __init__(self, benchmark_report: BenchmarkReport | None = None): ...
    def annotate(self, transitions: list[dict]) -> list[dict]:
        """Mutates `reward=None` → float."""
```

#### 5.2.2 Training data pipeline (`src/learning/training_pipeline.py`)

Aggregates multiple trajectories into a training-ready dataset:

- Deduplication (trajectories with identical user prompts collapsed, rewards averaged)
- Filtering (drop `reward < threshold` / truncated / error-terminal)
- Format routing: SFT (`jsonl`) vs RL (`parquet` or `webdataset` for scale)
- Versioned output: `trajectories/training/v{N}/{sft,rl}/...`

#### 5.2.3 Training runner (`src/learning/trainer.py`)

Thin wrapper around an external trainer (we don't reimplement training). Two supported paths:

- **SFT path**: Axolotl config + LoRA adapter output. Yigent invokes `axolotl train ...` as a subprocess.
- **RL path**: Atropos environment + trl.GRPOTrainer. Yigent configures the environment, launches trl.

Output is **always a LoRA adapter** (not a full model) — cheap to train, cheap to swap, easy to roll back.

#### 5.2.4 Model version manager (`src/providers/model_version_manager.py`)

Extends `src/providers/resolver.py` with a version registry:

- `models_dir/{base_model_id}/v1/adapter.safetensors`
- `models_dir/{base_model_id}/v2/adapter.safetensors`
- Each version has a manifest: `training_dataset_hash`, `trained_at`, `benchmark_score`

Provider resolves `config.provider.model_version = "latest"` or `"v3"` or `"baseline"`.

#### 5.2.5 Regression detector (`src/eval/regression_detector.py`)

Every new trained version must run the full benchmark and beat a fraction of the previous version's tasks. If not, auto-rollback. This is the safety valve that prevents L3 from degrading the agent.

- Decision rule: new version retained iff `new.avg_score >= old.avg_score − tolerance` AND `new.consistency_score >= old.consistency_score − tolerance`
- Fallback: keep the previous version as active, archive the rejected version for analysis

#### 5.2.6 Orchestration (`src/learning/evolution_cycle.py`)

A `python -m src.learning.evolve` entry point that runs the full cycle:

```
1. Collect trajectories from `trajectories/` since last evolution
2. RewardAnnotator fills in rewards using the last benchmark report
3. TrainingPipeline builds a dataset, writes to training/v{N}/
4. Trainer produces a LoRA adapter → models/v{N}/
5. ModelVersionManager registers v{N}
6. BenchmarkRunner runs the full benchmark with the new version
7. RegressionDetector: retain or rollback
8. If retained, update config.provider.model_version → v{N}
9. Session reset → next user interaction uses the evolved model
```

### 5.3 Scope boundaries for L3

**In scope for Phase 4:**
- The orchestration + plumbing above
- LoRA-only (no full-model fine-tunes)
- One base-model at a time (no multi-model ensemble)
- Reward from benchmark scores only (no human preference labels)

**Out of scope (deferred to L4+):**
- Full-parameter fine-tuning
- Multi-model MoE or ensemble
- RLHF / DPO from user preferences
- Distributed training across multiple GPUs (single-GPU LoRA only)

### 5.4 Minimum viable Phase 4 deliverable

A single demonstrable cycle:
1. Run benchmark → 40% completion (baseline)
2. Run `python -m src.learning.evolve` on the baseline's trajectories
3. Run benchmark again → > 40% completion (demonstrates improvement)

If that demo works even once on one domain (say, `data_analysis`), L3 is validated. If it works across multiple domains, the "evolving harness" narrative is complete.

---

## 6. Anti-patterns — things we explicitly don't do

Based on reading CC, Hermes, claw-code, and learn-claude-code:

1. **No LangGraph / LangChain / CrewAI.** The whole point is building the harness from scratch to understand each layer. See `docs/DECISIONS.md D2`.
2. **No monolith.** Hermes's `run_agent.py` is 11,700 lines; every feature inline. We keep features in separate modules with injection seams (`trajectory=None`, `learning=None`). Testability is the payoff.
3. **No prompt-based safety.** Plan mode blocks writes at the **permission layer**, not by asking the model nicely. Same for YOLO shadow classifier — the regex catches `rm -rf /` *before* the aux LLM has an opinion.
4. **No mandatory features.** Every Phase 2/3 addition is opt-in via constructor parameter. An agent loop with no learning, no trajectory, no nudge still works — that's the baseline test suite.
5. **No vendor lock-in.** 4+ providers (DeepSeek, OpenAI-compat, Anthropic-compat, MiniMax-/anthropic) are hot-swappable. `docs/PROVIDER_COMPARISON.md` (gitignored local doc) explains the runtime resolution.
6. **No silent failures.** `__parse_error__` sentinel, circuit breakers everywhere, forensic logging on provider anomalies. If something goes wrong, the logs tell you **what** and **why** and **what the agent observed**.

---

## 7. Where to look when

A lookup table for future-you:

| If you're working on... | Read first |
|---|---|
| **"What does CC do about X?"** | sanbuphy/claude-code-source-code README — full `src/` tree + 12 harness mechanisms + data flow diagrams mapped to real CC file names (`query.ts`, `QueryEngine.ts`, `StreamingToolExecutor.ts`, `services/compact/`, `AgentTool/`, etc.) |
| Agent loop changes | `src/core/agent_loop.py` + `docs/ARCHITECTURE.md §A` + sanbuphy README "Data Flow: A Single Query Lifecycle" + claw-code `rust/crates/runtime/src/conversation.rs` for shape validation |
| Permission / safety | `src/safety/permission_gate.py` + `docs/ARCHITECTURE.md §G` + sanbuphy README "Tool System Architecture" section + CC Plan-mode deep-dive zhuanlan articles |
| Context compression | `src/context/engine.py` + `docs/ARCHITECTURE.md §I` + sanbuphy README "Context Management" section + Hermes `context_compressor.py` |
| New provider | `src/providers/base.py` + `src/providers/resolver.py` + `docs/PROVIDER_COMPARISON.md` (gitignored, ask first) |
| New tool | `src/tools/` (look at `file_ops.py` for a simple example; `mcp_adapter.py` for external tool integration); sanbuphy README "Tool Interface" section shows CC's Tool shape |
| Skill lifecycle | `src/learning/skill_creator.py` / `skill_improver.py` + `src/memory/skill_index.py`. Hermes counterpart: `agent/skill_commands.py` + `run_agent.py:2772` prompts. CC counterpart: `src/tools/SkillTool/` per sanbuphy README |
| Sub-agents / multi-agent | `src/core/multi_agent.py`. CC counterpart per sanbuphy: `src/tools/AgentTool/`, `forkSubagent.ts`, `src/tasks/{LocalShellTask, LocalAgentTask, InProcessTeammateTask, DreamTask}/`, `utils/swarm/` |
| Trajectory / training | `src/learning/trajectory.py` + §5 above for L3 plan |
| Evaluation | `src/eval/benchmark.py` + `configs/eval_tasks.yaml` + `src/eval/judges/rule_checks.py` |
| Feature flags / growth / telemetry | sanbuphy `docs/en/01-telemetry-and-privacy.md` + `docs/en/04-remote-control-and-killswitches.md` (CC's hidden flag and kill-switch system — useful reference for what a production harness exposes) |
| Why a decision was made | `docs/DECISIONS.md` (12 numbered decisions) |

---

## 8. The elevator pitch

*"Yigent is a Claude-Code-shaped general-purpose agent harness built from scratch in Python, hosting a Hermes-Agent-shaped learning loop on top. Phase 1-2 gave it the CC reliability substrate: streaming execution, Plan mode, 5-layer compression, multi-agent, MCP, 8 hooks, 5-layer permission gate. Phase 3 added Hermes's evolutionary half: auto-created skills, skill self-improvement with rollback, always-on trajectory recording in SFT+RL formats. Phase 4 closes the loop — trajectories train LoRA adapters, the adapter replaces the session model, the agent becomes stronger with use. The destination is model self-evolution; the current milestone is procedural memory self-evolution; the foundation is production-grade execution safety."*
