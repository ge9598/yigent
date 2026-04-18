# Yigent

> As free as [OpenClaw](https://github.com/Martian-Engineering/lossless-claw), as harness as [Claude Code](https://claude.com/code), and as evolving as [Hermes Agent](https://github.com/NousResearch/hermes-agent).

A learning project building a general-purpose agent harness **from scratch** in Python — no LangGraph, no LangChain. The goal is to deeply understand the internals of modern coding agents by reimplementing their core mechanisms.

## What this project studies

| Mechanism | Source | Our implementation |
|---|---|---|
| Async generator ReAct loop | Claude Code | `src/core/agent_loop.py` |
| Streaming tool execution | Claude Code | `src/core/streaming_executor.py` |
| Deferred tool loading + ToolSearch | Claude Code | `src/tools/registry.py` |
| Plan mode (permission-layer enforcement) | Claude Code | `src/core/plan_mode.py` |
| Multi-provider hot-swap | Hermes Agent | `src/providers/` |
| Iteration budget with allocate | Hermes Agent | `src/core/iteration_budget.py` |
| Self-registration tool system | Hermes Agent | `src/tools/` |
| 5-layer context compression | Claude Code + Hermes | `src/context/` (Phase 2) |
| Periodic nudge + skill auto-creation | Hermes Agent | `src/learning/` (Phase 3) |
| Trajectory export (ShareGPT/RL) | Hermes Agent | `src/learning/` (Phase 3) |

## Architecture

```
User Input
  → Agent Loop (async generator, yields events to UI)
    → Provider.stream_message() (OpenAI-compatible, multi-provider)
    → StreamingExecutor (parallel tool exec via TaskGroup)
      → Permission Gate (plan mode blocks writes at system level)
    → EnvironmentInjector (git/dir/data context per turn)
  → CLI (Rich TUI, streaming output, permission prompts)
```

## Quick start

```bash
# 1. Clone & install
git clone https://github.com/cgxy1995/yigent.git
cd yigent
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

# 2. Configure your LLM provider
cp configs/default.yaml configs/local.yaml
# Edit local.yaml — add your API key (DeepSeek, MiniMax, GLM, Kimi, etc.)

# 3. Run
python -m src.ui.cli           # interactive mode
python -m src.ui.cli --smoke-test  # verify connectivity
```

Any OpenAI-compatible provider works. Example `configs/local.yaml`:

```yaml
provider:
  name: openai_compat
  api_key: "your-key"
  base_url: "https://api.deepseek.com/v1"  # or any OpenAI-compat endpoint
  model: "deepseek-chat"
```

## Project structure

```
src/
├── core/          # Agent loop, streaming executor, plan mode, budget, env injector
├── tools/         # Tool registry + built-in tools (file ops, bash, python, search)
├── providers/     # Multi-provider support (DeepSeek, MiniMax, GLM, any OpenAI-compat)
├── context/       # Context assembler + 5-layer compression (Phase 2)
├── memory/        # Episodic (SQLite+FTS5) + semantic (Qdrant) memory (Phase 2)
├── learning/      # Periodic nudge, skill creator, trajectory export (Phase 3)
├── safety/        # Permission gate, hook system (Phase 2)
├── eval/          # 4-domain benchmark with LLM-as-Judge (Phase 3)
└── ui/            # Rich CLI + FastAPI/SSE API server
```

## Built-in tools (Phase 1)

| Tool | Permission | Description |
|---|---|---|
| `read_file` | read | Read text file with line numbers |
| `write_file` | write | Atomic write with parent dir creation |
| `list_dir` | read | Recursive directory tree |
| `search_files` | read | Regex search across files |
| `bash` | execute | Shell command (git-bash on Windows) |
| `python_repl` | execute | Stateless Python subprocess |
| `web_search` | read | Tavily (preferred) / DuckDuckGo |
| `tool_search` | read | Discover & activate deferred tools |
| `enter_plan_mode` | read (deferred) | Activate plan-then-execute mode |
| `exit_plan_mode` | write | Save plan & resume normal mode |
| `ask_user` | read | Prompt human for input |

## Design philosophy

1. **Model is the agent, code is the harness.** The LLM reasons and decides. Our code gives it a reliable execution environment.
2. **Permissions are architecture, not suggestions.** Plan mode blocks writes at the system level, not via prompts the model might ignore.
3. **Learning is architecture, not an addon.** (Phase 3) The nudge-skill-trajectory loop is part of the core cycle.
4. **Progressive disclosure.** Tool schemas load on demand via ToolSearch. Context compresses in layers.

## Tests

```bash
pytest tests/ -v        # 34 tests covering registry, budget, plan mode, executor, agent loop, env injector
```

## Roadmap

- [x] **Phase 1** — Core loop + tools + CLI (complete)
- [ ] **Phase 2** — Multi-agent + memory + context compression + permission gate
- [ ] **Phase 3** — Learning loop + eval benchmark + API server

## References

- [Claude Code source analysis](https://github.com/sanbuphy/claude-code-source-code)
- [Hermes Agent architecture](https://hermes-agent.nousresearch.com/docs/developer-guide/architecture)
- [learn-claude-code tutorials](https://github.com/shareAI-lab/learn-claude-code)

## License

MIT
