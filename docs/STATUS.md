# Implementation status

## Phase 1: Core loop + basic tools (Week 1)

- [ ] `src/core/agent_loop.py` — async generator ReAct loop
- [ ] `src/core/streaming_executor.py` — parallel tool execution during model output
- [ ] `src/tools/registry.py` — self-registration + ToolSearch + deferred loading
- [ ] `src/tools/file_ops.py` — read_file, write_file, list_dir, search_files
- [ ] `src/tools/coding.py` — bash execution with timeout and sandboxing
- [ ] `src/tools/interpreter.py` — Python REPL in isolated subprocess
- [ ] `src/tools/search.py` — web search via API
- [ ] `src/core/plan_mode.py` — enter/exit plan mode + permission-level write blocking
- [ ] `src/tools/plan_tools.py` — EnterPlanMode (deferred), ExitPlanMode, AskUser
- [ ] `src/providers/base.py` — Provider ABC with streaming interface
- [ ] `src/providers/deepseek.py` — DeepSeek V3 via OpenAI-compat
- [ ] `src/providers/openai_compat.py` — generic OpenAI-compatible provider
- [ ] `src/providers/resolver.py` — runtime provider selection from config
- [ ] `src/core/iteration_budget.py` — shared budget with allocate()
- [ ] `src/core/env_injector.py` — git/schema/dir tree injection per task type
- [ ] `src/ui/cli.py` — Rich TUI with streaming output + permission prompts
- [ ] `configs/default.yaml` — default config with provider settings

## Phase 2: Multi-agent + memory + context (Week 2)

- [ ] `src/core/capability_router.py` — intent classification + strategy selection
- [ ] `src/context/assembler.py` — 5-zone context assembly (static/tools/env/conv/reserve)
- [ ] `src/context/prompt_cache.py` — frozen prefix management, Fork cache sharing
- [ ] `src/core/multi_agent.py` — Fork, Subagent, TaskBoard
- [ ] `src/context/engine.py` — 5-layer progressive compression
- [ ] `src/context/circuit_breaker.py` — failure tracking with auto-disable
- [ ] `src/memory/working.py` — L0 in-memory messages + todo
- [ ] `src/memory/episodic.py` — L1 SQLite + FTS5 session store
- [ ] `src/memory/skill_index.py` — skill registry + matching
- [ ] `src/safety/permission_gate.py` — 4-layer check chain
- [ ] `src/safety/hook_system.py` — lifecycle events + hook registration
- [ ] `src/tools/mcp_adapter.py` — MCP stdio transport + dynamic tool conversion
- [ ] `configs/hooks.yaml` — default hook definitions

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
- [ ] `docker-compose.yml` — Qdrant + API server
- [ ] Tests for all core modules
- [ ] README.md for GitHub
- [ ] docs/EVAL_REPORT.md with real benchmark numbers
- [ ] Demo video (2-3 minutes)

## Current focus

> Not started yet. Begin with Phase 1, Day 1-2: agent_loop.py + streaming_executor.py.
