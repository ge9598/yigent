# Yigent

> As free as [OpenClaw](https://github.com/Martian-Engineering/lossless-claw), as harness as [Claude Code](https://claude.com/code), and as evolving as [Hermes Agent](https://github.com/NousResearch/hermes-agent).

A general-purpose AI agent harness built **from scratch** in Python — no
LangGraph, no LangChain. Combines Claude Code's reliability engineering
(streaming tool execution, Plan mode, 5-layer context compression) with
Hermes Agent's learning loop (periodic nudge, auto-created skills,
trajectory export). The whole point is understanding what those
frameworks do under the hood by reimplementing them.

**Current status:** Phase 1 (core loop + tools), Phase 2 (context /
safety / memory / multi-agent / MCP / provider routing), and Phase 3
(learning loop + eval benchmark + API server) are complete. **526 tests
passing.**

## What this project studies

| Mechanism | Source | Our implementation |
|---|---|---|
| Async generator ReAct loop | Claude Code | `src/core/agent_loop.py` |
| Streaming tool execution + tombstone repair | Claude Code | `src/core/streaming_executor.py` |
| Deferred tool loading + ToolSearch | Claude Code | `src/tools/registry.py` |
| Plan mode (permission-layer enforcement) | Claude Code | `src/core/plan_mode.py` |
| 5-layer context compression + dynamic thresholds | Claude Code + Hermes | `src/context/` |
| Permission gate (5-layer chain + YOLO shadow) | Claude Code | `src/safety/permission_gate.py` |
| 8 lifecycle hooks | Claude Code | `src/safety/hook_system.py` |
| Multi-agent (Main / Fork / Subagent + TaskBoard) | Claude Code | `src/core/multi_agent.py` |
| MCP adapter (stdio + SSE) | Claude Code + Hermes | `src/tools/mcp_adapter.py` |
| Multi-provider hot-swap + credential pool | Hermes Agent | `src/providers/` |
| Scenario routing (CCR-compatible) | CCR | `src/providers/scenario_router.py` |
| Iteration budget with shared allocate | Hermes Agent | `src/core/iteration_budget.py` |
| Markdown memory (L1, Claude-Code-style) | Claude Code | `src/memory/markdown_store.py` |
| Periodic nudge (aux LLM + circuit breaker) | Hermes Agent | `src/learning/nudge.py` |
| Skill auto-creation (agentskills.io format) | Hermes Agent | `src/learning/skill_creator.py` |
| Skill self-improvement + rollback | Hermes Agent | `src/learning/skill_improver.py` |
| Trajectory export (ShareGPT + RL transitions) | Hermes Agent | `src/learning/trajectory.py` |
| 4-domain benchmark (LLM-as-Judge + rule checks) | own design | `src/eval/` |

## Architecture

```
User input
    → Plan mode? (complex tasks: plan → approve → execute)
    → Capability router (intent → tool pre-activation)
    → Context assembler (5 zones, cache-friendly system prompt)
    → Agent loop (async generator, ReAct cycle)
         ├─ Streaming tool executor (parallel dispatch during model output)
         ├─ Permission gate (schema → self-check → plan-mode → hook → level)
         └─ Hooks (8 lifecycle events)
    → Multi-agent coordinator (Main / Fork with shared cache / Subagent)
    → Learning loop (nudge every N tool calls → skill on success → always-on trajectory)
    → Eval benchmark (per-task workspace, dual-channel scoring)
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for module-by-module
design.

## Quick start

```bash
# 1. Install
git clone https://github.com/cgxy1995/yigent.git && cd yigent
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

# 2. Configure (copy default, add your API key)
cp configs/default.yaml configs/local.yaml
# Any OpenAI-compatible endpoint works (DeepSeek, MiniMax, GLM, Kimi, Qwen, local vLLM...).

# 3. Run
python -m src.ui.cli                            # interactive Rich CLI
python -m src.ui.cli --smoke-test               # verify connectivity
python -m src.ui.api                            # FastAPI + SSE server on :8000
python -m src.eval.benchmark --suite all -v     # full benchmark
pytest tests/ -q                                # 526 tests
```

Example `configs/local.yaml`:

```yaml
provider:
  name: openai_compat
  api_key: "your-key"
  base_url: "https://api.deepseek.com/v1"
  model: "deepseek-chat"
  # Optional: credential pool + scenario routing
  keys: ["key1", "key2"]
  strategy: "round_robin"
  auxiliary:
    model: "deepseek-chat"   # nudge, skill, judge
  routes:
    background: "auxiliary"
    long_context: "self"
    thinking: "self"
```

## Feature tour

### Reliability engineering (Phase 1–2)

- **Streaming tool execution.** Tools start preparing as soon as the
  model emits their call — no wait for end of stream. Parallel sibling
  tools overlap; one fatal failure cancels pending siblings via
  tombstone repair.
- **Plan mode.** Three-phase cycle (Plan → Approve → Execute) enforced
  at the permission layer. Writes are blocked at system level, not via
  a prompt instruction the model might ignore.
- **5-layer context compression.** Truncate → dedup → summarize 1/3 →
  full rewrite → hard truncate. Per-layer circuit breakers. Dynamic
  thresholds at `context_window − 40K / −33K / −23K`, so swapping in a
  longer-context model automatically delays compression.
- **Permission gate.** Five-layer chain — schema validation → tool
  self-check → plan-mode enforcement → hook chain → permission-level
  classification. YOLO mode adds an aux-LLM shadow classifier for
  dangerous operations.
- **Hook system.** 8 lifecycle events (session_start, pre/post_tool_use,
  pre/post_compression, plan_approved, budget_warning, session_end) —
  Python callables or shell commands.
- **MCP integration.** stdio + SSE transports, per-server
  `default_permission` level forcing explicit opt-in for dangerous
  servers.
- **Provider hot-swap.** DeepSeek, OpenAI-compatible, Anthropic/MiniMax
  /anthropic. Credential pool with round_robin/fill_first/least_used/
  random strategies, 429/402 cooldown, 401 permanent expiry.
  CCR-compatible scenario routing (`default`/`background`/`long_context`
  /`thinking`) keyed by env_injector task types.
- **Multi-agent.** Three spawn modes (Main / Fork with shared prompt
  cache / Subagent with independent stack), in-memory TaskBoard with
  asyncio.Lock.

### Self-improvement (Phase 3)

- **Trajectory recorder.** Always-on recording of every turn. Exports as
  ShareGPT JSON for SFT or RL transitions (state, action, reward,
  next_state) for GRPO/DPO. Agent-loop overhead is near-zero: one
  dataclass append per turn, no I/O until explicit save.
- **Periodic nudge.** Every N tool calls (default 15), aux LLM inspects
  recent activity and persists non-obvious patterns to L1 markdown
  memory. Circuit breaker after 3 consecutive failures disables nudge
  for the session. Resists sycophancy — the system prompt lists
  antipatterns (don't save: ephemeral state, code-derivable facts,
  generic advice).
- **Skill auto-creation.** On successful complex tasks (≥4 tool calls,
  ≥2 distinct tools), extract the workflow as SKILL.md in
  agentskills.io format. Two-stage Jaccard dedup against existing
  skills avoids index pollution.
- **Skill self-improvement.** When a skill's expected_tool_count is
  beaten by >20%, aux LLM rewrites the Steps section. Old versions
  archived under `skills/.history/`; rollback via
  `SkillImprover.rollback_to_previous(slug)` — wired to the benchmark
  runner's regression detector.
- **Eval benchmark.** 4 domains × 3 difficulty = 12 tasks (coding /
  data_analysis / research / file_management). Dual-channel scoring:
  deterministic rule checks (11 types) + LLM-as-Judge on correctness /
  efficiency / robustness at temperature 0. Per-task workspace
  synthesis (CSV / logs / mixed files / duplicates). Metrics include
  **cross-domain consistency score** = `1 − variance(per-domain rates)
  / 0.25` — a direct measure of generalization, the JD requirement.

## Project layout

```
src/
├── core/          # agent_loop, streaming_executor, plan_mode, multi_agent,
│                   capability_router, env_injector, iteration_budget, config
├── context/       # assembler (5-zone), engine (5-layer), prompt_cache,
│                   circuit_breaker
├── providers/     # base, deepseek, openai_compat, anthropic_compat,
│                   resolver, credential_pool, scenario_router,
│                   endpoint_quirks, reasoning_extractor
├── tools/         # registry + tool_search, plan_tools, task_tools,
│                   memory_tools, mcp_adapter, file_ops, coding, interpreter,
│                   search
├── memory/        # markdown_store (L1), skill_index
├── safety/        # permission_gate, hook_system
├── learning/      # trajectory, nudge, skill_creator, skill_improver,
│                   skill_format, nudge_prompt
├── eval/          # benchmark, reporter, judges/{rule_checks, llm_judge}
└── ui/            # cli (Rich TUI), api (FastAPI + SSE), slash_commands

configs/       # default.yaml, eval_tasks.yaml, hooks.yaml
skills/        # Auto-created skills (agentskills.io format)
trajectories/  # Exported trajectories
tests/         # 526 tests (pytest + pytest-asyncio)
docs/          # ARCHITECTURE.md, DECISIONS.md, STATUS.md, EVAL_REPORT.md,
               # PROVIDER_COMPARISON.md
```

## Benchmark

The built-in benchmark measures cross-domain consistency:

```bash
python -m src.eval.benchmark --suite all -v
# → docs/EVAL_REPORT.md with per-domain + overall metrics
# → benchmark_runs/report.json for programmatic consumption
# → benchmark_runs/{domain}_{difficulty}.json trajectories (ShareGPT)
```

Each task runs in an isolated workspace with a synthetic dataset seeded
from the task's `setup:` hint. The agent gets a fresh IterationBudget,
TrajectoryRecorder, and tool registry per task. Rule checks and the LLM
judge run at scoring time.

Metrics produced:

- **completion_rate** — passed both rule check and received > 0 judge score
- **avg_score** — `0.4 × rule_score + 0.6 × judge_score`, each ∈ [0, 10]
- **avg_steps** — mean tool-call count per task (efficiency proxy)
- **recovery_rate** — of tasks that hit errors, fraction that still passed
- **consistency_score** — `1 − variance(per-domain rates) / 0.25`
- **skill_creation_count** — skills auto-created during the run

Real numbers from a live run go in
[`docs/EVAL_REPORT.md`](docs/EVAL_REPORT.md).

## HTTP API

```bash
python -m src.ui.api   # :8000
```

```bash
# Stream a chat via SSE
curl -N -X POST http://localhost:8000/chat \
     -H 'Content-Type: application/json' \
     -d '{"messages":[{"role":"user","content":"list the tools you have"}]}'
# → event: TokenEvent ... event: ToolCallStartEvent ... event: FinalAnswerEvent

# Status
curl http://localhost:8000/status

# Download a session's trajectory
curl http://localhost:8000/trajectory/{session_id}
```

v1 limitations: permissions are auto-allow (single-channel SSE can't
confirm interactively), sessions are in-memory only.

## Comparison

| Feature | Yigent | Claude Code | Hermes Agent |
|---|---|---|---|
| Language | Python 3.11+ | TypeScript | Python |
| Framework dependency | none | proprietary | proprietary |
| Streaming tool execution | ✓ | ✓ | — |
| Plan mode | ✓ (permission-level enforcement) | ✓ | — |
| 5-layer compression | ✓ (dynamic thresholds) | ✓ | single-layer |
| MCP integration | ✓ stdio + SSE | ✓ | ✓ |
| Multi-agent | ✓ Main / Fork / Subagent | ✓ | ✓ |
| Credential pool rotation | ✓ 4 strategies | — | ✓ |
| Scenario routing | ✓ (CCR-compatible) | — | — |
| Periodic nudge | ✓ | — | ✓ |
| Skill auto-creation | ✓ | — | ✓ |
| Skill self-improvement + rollback | ✓ | — | ✓ (DSPy/GEPA) |
| Trajectory export (ShareGPT + RL) | ✓ | — | ✓ |
| Built-in eval benchmark | ✓ 4-domain | — | — |

## Design philosophy

1. **Model is the agent, code is the harness.** The LLM reasons and
   decides. Our code gives it a reliable execution environment.
2. **Permissions are architecture, not suggestions.** Safety constraints
   live in the permission gate, not in prompts.
3. **Learning is architecture, not an addon.** The nudge-skill-trajectory
   loop is part of the core cycle.
4. **Progressive disclosure.** Tool schemas load on demand via
   ToolSearch. Context compresses in layers. Skills load when relevant.

## Docs

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — module-by-module design
- [`docs/DECISIONS.md`](docs/DECISIONS.md) — 12 technical decisions with
  rationale
- [`docs/STATUS.md`](docs/STATUS.md) — implementation status and test
  count timeline (34 → 526)
- [`docs/EVAL_REPORT.md`](docs/EVAL_REPORT.md) — real benchmark numbers
  (generated by `python -m src.eval.benchmark`)
- [`docs/PROVIDER_COMPARISON.md`](docs/PROVIDER_COMPARISON.md) —
  provider system vs. CCR / LiteLLM / Hermes / OpenClaude / OpenRouter

## References

- [Claude Code source analysis](https://github.com/sanbuphy/claude-code-source-code)
- [Claude Code deep dive — 5-layer compression, Fork, Plan mode](https://zhuanlan.zhihu.com/p/2022442135182406883)
- [Hermes Agent architecture](https://hermes-agent.nousresearch.com/docs/developer-guide/architecture)
- [Inside Hermes Agent — learning loop, procedural memory](https://mranand.substack.com/p/inside-hermes-agent-how-a-self-improving)
- [learn-claude-code tutorials (19 sessions)](https://github.com/shareAI-lab/learn-claude-code)

## License

MIT
