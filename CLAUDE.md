# Yigent Harness

## Design intent (read this first)

Yigent's shape is **deliberately split**:

- **Harness = Claude Code–style.** Streaming tool execution, Plan mode, 5-layer compression, permission gate as architecture (not prompt), 8 lifecycle hooks, deferred tool loading via ToolSearch, multi-agent (Main/Fork/Subagent). Every "how a harness executes" decision defers to how CC does it. This is the reliable-execution substrate.
- **Learning loop = Hermes Agent–style, with explicit self-evolution as the goal.** Periodic nudge, skill auto-creation from successful trajectories, skill self-improvement with rollback, always-on trajectory recording (ShareGPT + RL transitions). Hermes's most valuable property — **continuous iterative evolution of the agent's procedural knowledge** — is what Yigent is ultimately built for.

Phase 3 shipped L1 (skill sedimentation) and L2 (skill improvement).
**Phase 4's target is L3 — model self-evolution via trajectory → training → redeployment, the Hermes-self-evolution direction.** See `docs/DESIGN_PHILOSOPHY.md` for the full rationale, the "blood-lineage" table of which subsystem borrows from which reference, and the L3 architecture-level plan.

Built from scratch in Python — no LangGraph, no framework dependency. Targets ByteDance Seed "通用Agent研究工程师" role.

Design doc with full rationale: @docs/ARCHITECTURE.md
Philosophy + L3 roadmap: @docs/DESIGN_PHILOSOPHY.md

## What (project map)

```
src/
├── core/          # Agent loop, streaming executor, multi-agent, plan mode, router
├── tools/         # Tool registry + ToolSearch + built-in tools (search, coding, interpreter, file_ops)
├── memory/        # L0 working (messages/todo), L1 episodic (SQLite+FTS5), L2 semantic (Qdrant)
├── context/       # Context assembler (5-zone layout), 5-layer compression, circuit breaker, prompt cache
├── learning/      # Periodic nudge, skill creator, skill improver, trajectory recorder
├── safety/        # Permission gate (4-layer chain), hook system, shadow classifier
├── eval/          # Benchmark framework (4 domains × 3 difficulty), LLM-as-Judge
├── providers/     # Multi-provider support (DeepSeek, OpenAI-compat, hot-swap)
└── ui/            # Rich CLI + FastAPI/SSE API server
```

configs/       — YAML configs (default, tools, hooks, eval tasks)
skills/        — Auto-created skills (agentskills.io format)
trajectories/  — Exported trajectories (ShareGPT/RL format)
docs/          — Detailed architecture, decisions, eval reports

## How

### Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp configs/default.yaml configs/local.yaml   # edit with your API keys
```

### Run
```bash
python -m src.ui.cli                         # interactive CLI
python -m src.ui.api                         # API server (localhost:8000)
```

### Test
```bash
pytest tests/ -q                             # unit tests
python -m src.eval.benchmark --suite all     # full eval benchmark
python -m src.eval.benchmark --suite coding  # single domain
```

### Verify changes
After any code change, run this sequence:
```bash
pytest tests/ -q && python -m src.ui.cli --smoke-test
```

## Working discipline

- **No shortcuts to explain away a problem.** If you catch yourself reaching for a cheap explanation — "the doc is aspirational", "the test is flaky", "close enough for MVP", rewriting the doc to match the code instead of making the code match the doc — stop immediately and solve the real problem. Docs are promises; code has to catch up, not the other way around.

## Constraints

- Python 3.11+. Use async/await throughout — the agent loop is an async generator.
- No LangGraph, no LangChain. The whole point is building the harness from scratch.
- Every tool call passes through the permission gate. No exceptions. No shortcuts.
- Plan mode blocks writes at the permission layer, not via prompt instructions.
- Context compression uses three dynamic thresholds: warn at `context_window - 40K`, compress at `-33K`, hard cutoff at `-23K`. See ARCHITECTURE.md §I.
- IterationBudget (default 90) is shared across parent and all child agents.
- Tools use deferred loading: names at startup, full schema on first use via ToolSearch.
- Hook system fires on every tool lifecycle event. Hooks can be Python callables or shell scripts.
- Skills follow agentskills.io format. Auto-created skills go to `skills/`.
- Trajectories export as ShareGPT JSON. Recording is always-on; export is explicit.

## Architecture quick reference

```
User Input
    → Plan Mode? (complex tasks get plan-then-execute)
    → Capability Router (classify intent → select tools)
    → Context Assembler (build 5-zone message list each turn):
        Zone 1: Static system prompt (frozen at init → prompt cache)
        Zone 2: Tool schemas (names always, full schema on demand via ToolSearch)
        Zone 3: Environment injection (git/schema/dir, refreshed every turn)
        Zone 4: Conversation (grows, compressed by 5-layer engine when over budget)
        Zone 5: (reserved for model output — 20K + 13K buffer)
    → Agent Loop (async generator ReAct cycle)
        → StreamingToolExecutor (parallel tool exec during model output)
        → Permission Gate (schema → self-check → hooks → rules → plan-mode)
    → Multi-Agent Coordinator (Main / Fork / Subagent + TaskBoard)
    → Learning Loop (periodic nudge → skill creation → trajectory recording)
    → Eval Benchmark (optional: LLM-as-Judge + rule checks)
```

**Critical distinction:** The agent loop owns `conversation[]` (user/assistant/tool messages only). The assembler prepends system prompt + tool schemas + env context to produce `assembled_messages[]` for each LLM call. The conversation[] persists across turns; assembled_messages[] is rebuilt every turn.

For full architecture details: @docs/ARCHITECTURE.md
For technical decisions and trade-offs: @docs/DECISIONS.md
For implementation status and next steps: @docs/STATUS.md

## References

When implementing a specific module, consult the relevant reference:

**Agent Loop / Streaming / Interruption / Permissions:**
- Claude Code source analysis (sanbuphy/claude-code-source-code) — a detailed public analysis of CC's actual source: full `src/` directory tree (~1884 .ts files, 512K lines), 12 progressive harness mechanisms mapped to real CC files (`query.ts` 785KB = main loop, `QueryEngine.ts`, `StreamingToolExecutor.ts`, `services/compact/`, `memdir/`, `SkillTool/`, `AgentTool/`, `forkSubagent.ts`, etc.). Plus 5 deep-analysis reports on telemetry / codenames / undercover mode / remote control / roadmap. **Read this when you want to know what a real CC module is called or what file it lives in:** https://github.com/sanbuphy/claude-code-source-code
- claw-code (third-party CC reimplementation in Python + Rust; `rust/crates/runtime/` has 43 files covering conversation/permissions/compact/hooks/mcp/sandbox — useful as a second implementation perspective on CC's shape): https://github.com/ge9598/claw-code
- Claude Code source deep dive (Fork, StreamingToolExecutor, 5-layer compression, Plan mode): https://zhuanlan.zhihu.com/p/2022442135182406883
- Claude Code 两万字核心机制详解 (Plan mode enforcement, ToolSearch, context compression details): https://zhuanlan.zhihu.com/p/2022443175361388953
- Claude Code 51万行源码解读 (design philosophy, file structure, async generator pattern): https://zhuanlan.zhihu.com/p/2022433246449780672
- B站视频 程序员鱼皮 (11个隐藏设计: YOLO shadow classifier, circuit breaker): https://bilibili.com/video/BV1ZB9EBmEAU

**Learning Loop / Memory / Skills / Trajectory / Provider — the self-evolution direction:**
- Hermes Agent (source): https://github.com/NousResearch/hermes-agent — key files for Phase 3: `agent/trajectory.py`, `agent/memory_manager.py`, `agent/skill_commands.py`, `agent/skill_utils.py`, and the background-review prompts in `run_agent.py:2761-2793`
- Hermes Agent architecture (official): https://hermes-agent.nousresearch.com/docs/developer-guide/architecture
- Hermes Agent loop internals (turn lifecycle, budget, fallback, tool execution): https://hermes-agent.nousresearch.com/docs/developer-guide/agent-loop
- Hermes context compression and caching (dual compression, 4-phase algorithm, prompt caching): https://hermes-agent.nousresearch.com/docs/developer-guide/context-compression-and-caching
- Hermes prompt assembly: https://hermes-agent.nousresearch.com/docs/developer-guide/prompt-assembly
- Inside Hermes Agent (learning loop, 4-layer memory, periodic nudge, skill creation): https://mranand.substack.com/p/inside-hermes-agent-how-a-self-improving
- **hermes-agent-self-evolution** (the L3 reference — DSPy/GEPA optimizer, MIPROv2, trajectory → training → redeployment loop): https://github.com/NousResearch/hermes-agent-self-evolution

**Progressive tutorial (build agent from scratch):**
- learn-claude-code (19 sessions, s01→s19): https://github.com/shareAI-lab/learn-claude-code
- learn-claude-code 中文文档: https://github.com/shareAI-lab/learn-claude-code/blob/main/README-zh.md

**Claude Code official docs:**
- How Claude Code works: https://code.claude.com/docs/en/how-claude-code-works
- Best practices: https://code.claude.com/docs/en/best-practices
