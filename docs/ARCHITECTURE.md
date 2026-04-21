# Architecture

> This document is the detailed design reference. CLAUDE.md links here via `@docs/ARCHITECTURE.md`.
> Read this when you need to understand how a specific module works or how modules interact.

## Design philosophy

1. **Model is the agent, code is the harness.** The LLM reasons and decides. Our code gives it a reliable execution environment: tools, memory, permissions, compression.
2. **Permissions are architecture, not suggestions.** Safety constraints live in the permission gate, not in prompts the model might ignore.
3. **Learning is architecture, not an addon.** The nudge-skill-trajectory loop is part of the core cycle, not a post-hoc feature.
4. **Progressive disclosure.** Tool schemas load on demand. Context compresses in layers. Skills load when relevant.

Sources: Claude Code source (streaming executor, Fork, Plan mode, 5-layer compression, Hook lifecycle), Hermes Agent (periodic nudge, skill auto-creation, trajectory export, multi-provider, IterationBudget).

---

## Module specifications

### A. Agent Loop Engine (`src/core/agent_loop.py`)

The central async generator that implements the ReAct cycle.

> Refs: [CC source deep dive — StreamingToolExecutor, tombstone, interruption](https://zhuanlan.zhihu.com/p/2022442135182406883) · [Hermes agent loop internals — turn lifecycle, budget, fallback](https://hermes-agent.nousresearch.com/docs/developer-guide/agent-loop) · [learn-claude-code s01](https://github.com/shareAI-lab/learn-claude-code/blob/main/docs/zh/s01-the-agent-loop.md)

**Interface:**
```python
async def agent_loop(
    conversation: list[Message],   # only the conversation, not system/tools
    tools: ToolRegistry,
    budget: IterationBudget,
    assembler: ContextAssembler,   # owns static zone + compression + assembly
    hooks: HookSystem,
    env_injector: EnvironmentInjector,
    learning: LearningLoop,
    trajectory: TrajectoryRecorder,
) -> AsyncGenerator[Event, None]:
```

**Per-turn sequence:**
1. `assembler.assemble(tools, env_injector, conversation, task_type)` — build full context (5 zones, compression if needed)
2. `llm.stream(assembled_messages, tools.get_active_schemas())` — stream LLM response
3. For each tool call in stream: `permission_gate.check()` → `hooks.fire("pre_tool_use")` → `executor.execute()` → `hooks.fire("post_tool_use")`
4. `trajectory.record_turn()` — log full turn for export
5. `budget.consume(1)` — decrement shared budget
6. If `stop_reason == "end_turn"`: check nudge interval, yield final answer, return
7. Else: append tool results to conversation[], goto 1

**Interruption handling:** On Ctrl+C, supplement each pending tool_use with an error-typed tool_result to maintain protocol consistency. Mark orphaned thinking blocks as tombstone.

**Events yielded:** `token`, `tool_call_start`, `tool_result`, `permission_request`, `final_answer`, `budget_exhausted`, `compression_triggered`.

---

### B. Streaming Tool Executor (`src/core/streaming_executor.py`)

Starts tool preparation while the model is still outputting tokens.

> Refs: [CC source — StreamingToolExecutor sibling-abort mechanism](https://zhuanlan.zhihu.com/p/2022442135182406883) · [learn-claude-code s02](https://github.com/shareAI-lab/learn-claude-code/blob/main/docs/zh/s02-tool-use.md)

**Behavior:**
- On `tool_call_start`: begin schema validation and permission pre-check
- On `tool_call_complete`: execute immediately, do not wait for other tool calls
- Sibling abort: if one tool fails fatally, cancel pending siblings
- Timeout: per-tool timeout (default 60s for bash, 30s for others)

---

### C. Plan Mode (`src/core/plan_mode.py`)

Three-phase cycle: Plan → Approve → Execute.

> Refs: [CC 两万字详解 — Plan mode permission-level enforcement, shouldDefer, ToolSearch](https://zhuanlan.zhihu.com/p/2022443175361388953)

**Key rules:**
- `EnterPlanMode` tool has `should_defer=True` — not in initial tool list, must be discovered via ToolSearch
- When active: `permission_gate` blocks all write/execute operations at system level
- Plan content saved to `plans/{session_id}_{timestamp}.md`
- User must explicitly approve before execution resumes
- Subagents cannot enter Plan mode (they have no UI for approval)

**Tools exposed during Plan mode:** read_file, list_dir, search, tool_search, ask_user_question. All write tools blocked.

---

### D. Multi-Agent Coordinator (`src/core/multi_agent.py`)

Three spawn modes:

> Refs: [CC source — Fork prompt cache sharing, 3 spawn modes, Swarm Mode](https://github.com/sanbuphy/claude-code-source-code) · [Hermes delegate_task + IterationBudget sharing](https://hermes-agent.nousresearch.com/docs/developer-guide/agent-loop) · [learn-claude-code s04 subagent, s09 agent teams](https://github.com/shareAI-lab/learn-claude-code/blob/main/docs/zh/s04-subagent.md)

| Mode | Context | Budget | Output | Cache | Use when |
|------|---------|--------|--------|-------|----------|
| Main | Own | Own | In-context | Own | Default |
| Fork | Inherited from parent | Shared | Isolated (output_file) | Shared | "I don't need intermediate output in my context" |
| Subagent | Fresh | Allocated subset | Returned to parent | Independent | Completely independent subtask |

**TaskBoard** (`src/core/multi_agent.py`):
- `create(task_id, description, depends_on)` — create task
- `claim(task_id, agent_id)` — atomic claim with dependency check
- `complete(task_id, result)` — mark done, unblock dependents
- `get_status()` — full board snapshot for parent agent
- Implementation: in-memory dict + asyncio.Lock (no file locks needed in Python)

---

### E. Capability Router (`src/core/capability_router.py`)

Classifies user intent and selects execution strategy.

**Two paths:**
1. Simple task (1-3 steps, single capability) → direct Main Agent execution
2. Complex task (4+ steps, multi-capability) → trigger Plan Mode → decompose into TaskBoard tasks → Fork/Subagent execution

**Capabilities registered:** search, coding, interpreter, file_ops. Extensible via config.

---

### F. Tool System (`src/tools/`)

**Registry** (`src/tools/registry.py`):

> Refs: [CC source — ToolSearch, shouldDefer, deferred loading](https://zhuanlan.zhihu.com/p/2022443175361388953) · [Hermes tools/registry.py self-registration pattern](https://hermes-agent.nousresearch.com/docs/developer-guide/architecture) · [learn-claude-code s02 tool use, s05 skill loading](https://github.com/shareAI-lab/learn-claude-code/blob/main/docs/zh/s02-tool-use.md)
- Self-registration at import time (Hermes pattern): each tool file calls `registry.register()` on import
- Deferred loading (CC pattern): `get_initial_tools()` returns names + short descriptions only; `tool_search(query)` returns full schemas for matching tools
- `get_active_schemas()` returns only tools whose schemas have been loaded

**Built-in tools** (minimum viable set):

| Tool | File | Permission level |
|------|------|-----------------|
| read_file | tools/file_ops.py | read_only |
| write_file | tools/file_ops.py | write |
| list_dir | tools/file_ops.py | read_only |
| search_files | tools/file_ops.py | read_only |
| bash | tools/coding.py | execute |
| python_repl | tools/interpreter.py | execute |
| web_search | tools/search.py | read_only |
| tool_search | tools/registry.py | read_only |
| enter_plan_mode | tools/plan_tools.py | read_only (deferred) |
| exit_plan_mode | tools/plan_tools.py | write |
| ask_user | tools/plan_tools.py | read_only |
| create_task | tools/task_tools.py | write |
| complete_task | tools/task_tools.py | write |

**MCP integration:** `src/tools/mcp_adapter.py` connects to external MCP servers via stdio or SSE transport. MCP tools are dynamically converted to the internal Tool schema and registered.

> Refs: [learn-claude-code s19 MCP plugin](https://github.com/shareAI-lab/learn-claude-code/blob/main/docs/zh/s19a-mcp-capability-layers.md) · [Hermes MCP tool — 2200 lines, dynamic conversion](https://hermes-agent.nousresearch.com/docs/developer-guide/tools-runtime)

---

### G. Permission Gate (`src/safety/permission_gate.py`)

Five-layer check chain, executed in order:

> Refs: [CC source — Zod → self-check → hooks → canUseTool, YOLO shadow classifier](https://zhuanlan.zhihu.com/p/2022433246449780672) · [B站视频 — YOLO mode shadow AI detail](https://bilibili.com/video/BV1ZB9EBmEAU) · [learn-claude-code s07 permission system](https://github.com/shareAI-lab/learn-claude-code/blob/main/docs/zh/s07-task-system.md)

1. **Schema validation** — tool call args match expected types (Pydantic)
2. **Tool self-check** — tool's own `validate()` method (e.g., bash checks for dangerous commands)
3. **Plan mode check** — if `permission_mode == "plan"`, block write/execute ops
4. **Hook check** — fire `pre_tool_use` hooks; any hook returning "deny" blocks
5. **Permission level** — classify tool into read_only/write/execute/destructive; auto-allow reads, ask user for writes, require confirmation for destructive

**YOLO mode:** Even with auto-approve, a shadow classifier (lightweight LLM call) screens for dangerous operations.

---

### H. Hook System (`src/safety/hook_system.py`)

Lifecycle events:

> Refs: [learn-claude-code s08 hook system](https://github.com/shareAI-lab/learn-claude-code/blob/main/docs/zh/s08-background-tasks.md) · [CC best practices — hooks for CI integration](https://code.claude.com/docs/en/best-practices)

| Event | Fires when | Common use |
|-------|-----------|-----------|
| session_start | New session begins | Load project-specific context |
| pre_tool_use | Before every tool execution | Validate, log, block dangerous ops |
| post_tool_use | After every tool execution | Run linter after write, log results |
| pre_compression | Before context compression | Save full context snapshot |
| post_compression | After context compression | Verify compression quality |
| plan_approved | User approves a plan | Log plan for audit trail |
| budget_warning | Budget < 20% remaining | Alert user, suggest wrapping up |
| session_end | Session terminates | Save session summary to L1 memory |

Hooks can be Python callables or paths to shell scripts. Loaded from `configs/hooks.yaml`.

---

### I. Context Engine (`src/context/engine.py`)

Five-layer progressive compression:

> Refs: [CC 两万字详解 — 5-layer compression, dynamic threshold formula, circuit breaker](https://zhuanlan.zhihu.com/p/2022443175361388953) · [Hermes context compression — dual system, 4-phase algorithm, structured summary template, before/after example](https://hermes-agent.nousresearch.com/docs/developer-guide/context-compression-and-caching) · [learn-claude-code s06 context compact](https://github.com/shareAI-lab/learn-claude-code/blob/main/docs/zh/s06-context-compact.md)

| Layer | Method | Cost | Trigger |
|-------|--------|------|---------|
| 1 | Truncate tool results > 3000 chars | Free | Always |
| 2 | Dedup file reads (same file, unchanged mtime → stub) | Free | Always |
| 3 | Summarize earliest 1/3 of turns via aux LLM | 1 LLM call | tokens > warn_threshold |
| 4 | Full rewrite: keep only last 5 turns as original | 1 LLM call | tokens still > compress_threshold |
| 5 | Hard truncate: system msgs + last 4 turns | Free | tokens > hard_cutoff |

**Dynamic thresholds:**
```
warn_threshold     = model_context_window - 40,000
compress_threshold = model_context_window - 33,000
hard_cutoff        = model_context_window - 23,000
```

**Circuit breaker:** LLM-based layers (3, 4) track consecutive failures. After 3 failures, skip to next layer. Resets on success.

**Compression boundary tracking:** The conversation zone maintains a `compression_cursor` — the index marking where "summarized" ends and "original" begins. Each compression pass moves this cursor rightward:

```
Turn 0: [orig][orig][orig][orig][orig][orig]   cursor=0
Turn 8: [SUMMARY      ][orig][orig][orig]      cursor=3 (first 3 turns summarized)
Turn 15:[SUMMARY              ][orig][orig]     cursor=5 (first 5 turns summarized)
Turn 20:[SUMMARY                    ][orig]     cursor=7 (only last turn is original)
```

The cursor is stored in `AgentState.compression_cursor` and persisted across turns within a session. The summary messages carry a `"compressed": true` flag so the assembler knows not to re-compress them.

---

### I-bis. Context Assembler (`src/context/assembler.py`)

The assembler orchestrates the full context window before each LLM call. It owns the **five-zone layout** and the **dynamic boundaries** between zones.

> Refs: [Hermes prompt assembly — system prompt construction from memory, skills, context files](https://hermes-agent.nousresearch.com/docs/developer-guide/prompt-assembly) · [CC source — layered context loading (CLAUDE.md full, MCP names only, schema deferred)](https://zhuanlan.zhihu.com/p/2022442135182406883) · [learn-claude-code s10 system prompt](https://github.com/shareAI-lab/learn-claude-code/blob/main/docs/zh/s10-team-protocols.md)

**Five zones (in order):**

| Zone | Content | Lifecycle | Cacheable |
|------|---------|-----------|-----------|
| 1. Static | System prompt + role + core rules | Frozen at session init | Yes — prompt cache hit every turn |
| 2. Tool schemas | Tool names (always) + full schemas (on demand) | Grows as tools are discovered via ToolSearch | Partially — stable prefix is cached |
| 3. Environment | Git status / data schema / dir tree | Replaced every turn (task-type-aware) | No |
| 4. Conversation | User msgs, assistant msgs, tool results | Grows, then compresses (5-layer engine) | No |
| 5. (Reserved) | Empty — space for model output | Never touched | N/A |

**Interface:**
```python
class ContextAssembler:
    def __init__(self, system_prompt: list[Message], model_context_window: int):
        # Freeze system prompt at init → maximize prompt cache hits
        self._frozen_system = list(system_prompt)  # never modified after init
        self._model_context_window = model_context_window
        self._output_reserve = 20_000
        self._safety_buffer = 13_000
    
    async def assemble(
        self,
        tool_registry: ToolRegistry,
        env_injector: EnvironmentInjector,
        conversation: list[Message],
        task_type: str,
    ) -> list[Message]:
        """
        Build the full message list for one LLM call.
        
        Returns messages in zone order: static → tools → env+conversation.
        The static zone is always identical (cache-friendly).
        """
        messages = []
        
        # Zone 1: Static (frozen)
        messages.extend(self._frozen_system)
        
        # Zone 2: Tool schemas (only discovered tools)
        tool_msg = self._build_tool_zone(tool_registry)
        messages.append(tool_msg)
        
        # Zone 3+4: Environment injected into conversation
        env_context = await env_injector.get_context(task_type)
        conversation = self._inject_env(conversation, env_context)
        
        # Check if compression needed
        usable_budget = (
            self._model_context_window 
            - self._output_reserve 
            - self._safety_buffer 
            - estimate_tokens(messages)  # static + tools already placed
        )
        
        if estimate_tokens(conversation) > usable_budget:
            conversation = await self._context_engine.compress(
                conversation, target_tokens=usable_budget
            )
        
        # Zone 4: Conversation
        messages.extend(conversation)
        
        return messages
    
    def _build_tool_zone(self, registry: ToolRegistry) -> Message:
        """
        Dynamic boundary: initial tools are names-only (~50 tok each).
        After ToolSearch, discovered tools expand to full schema (~300 tok each).
        Only active schemas are included.
        """
        schemas = registry.get_active_schemas()  # only loaded ones
        initial = registry.get_initial_tools()     # names + descriptions for unloaded
        return {"role": "system", "content": format_tools(schemas, initial)}
    
    def _inject_env(self, conversation: list[Message], env: str) -> list[Message]:
        """
        Inject environment context as prefix to the latest user message.
        NOT as a separate system message (that would grow messages[] every turn).
        """
        if conversation and conversation[-1]["role"] == "user" and env:
            conversation[-1] = {
                **conversation[-1],
                "content": f"[Environment]\n{env}\n\n{conversation[-1]['content']}"
            }
        return conversation
```

**Why freeze the static zone?**

The LLM provider computes KV cache for all input tokens. If the first N tokens are identical across calls, the provider can reuse the cached KV states — this is "prompt caching." Claude Code's source explicitly freezes the system prompt snapshot at session initialization for this reason. If you modify the system prompt mid-session (even adding a newline), the entire cache invalidates and every subsequent call pays full compute cost.

In our design, `_frozen_system` is set once in `__init__` and never touched. The tool zone grows (new schemas get appended), but the static zone prefix stays byte-identical → cache hit.

**Dynamic boundary illustration:**

```
Turn 1 (session start):
|===STATIC===|=TOOLS(names)=|=ENV=|===CONVERSATION===|...output reserve...|
                ^                   ^                   ^
            tool boundary      env boundary        compression threshold

Turn 5 (tools discovered via ToolSearch):
|===STATIC===|==TOOLS(names+schemas)==|=ENV=|==CONVERSATION(growing)==|.output.|
                      ^                                  ^
              tool boundary expanded              approaching threshold

Turn 12 (compression triggered):
|===STATIC===|==TOOLS==|=ENV=|[SUMMARY][orig][orig][orig]|..output..|
                                  ^         ^
                          compression    boundary moved right
                           cursor
```

---

### I-ter. Prompt Cache (`src/context/prompt_cache.py`)

Manages the cache-friendly prefix of the context window.

> Refs: [Hermes prompt caching — Anthropic cache breakpoints, system_and_3 strategy, TTL](https://hermes-agent.nousresearch.com/docs/developer-guide/context-compression-and-caching) · [CC source — Fork inherits parent cache, prefix byte matching](https://zhuanlan.zhihu.com/p/2022442135182406883)

**Strategy:**
- System prompt is frozen at session init (Zone 1)
- Tool zone prefix (names + descriptions for all tools) is stable — only full schemas get appended
- Fork agents inherit the parent's frozen system prompt → same prefix → shared cache
- Subagents build their own system prompt → independent cache

**Cache invalidation awareness:**
```python
class PromptCache:
    def __init__(self, frozen_system: list[Message]):
        self._prefix_hash = hash_messages(frozen_system)
    
    def is_cache_compatible(self, messages: list[Message]) -> bool:
        """Check if the current messages start with the cached prefix."""
        current_hash = hash_messages(messages[:len(self._frozen_system)])
        return current_hash == self._prefix_hash
    
    def on_fork(self) -> 'PromptCache':
        """Fork inherits the same prefix → cache sharing."""
        return PromptCache(self._frozen_system)  # same hash
    
    def on_subagent(self, new_system: list[Message]) -> 'PromptCache':
        """Subagent gets a new prefix → independent cache."""
        return PromptCache(new_system)  # different hash
```

---

### J. Memory System (`src/memory/`)

> Refs: [Hermes session storage — SQLite + FTS5 schema, WAL mode, session lineage](https://hermes-agent.nousresearch.com/docs/developer-guide/session-storage) · [Inside Hermes — 4-layer memory, periodic nudge persisting patterns](https://mranand.substack.com/p/inside-hermes-agent-how-a-self-improving) · [learn-claude-code s09 memory system](https://github.com/shareAI-lab/learn-claude-code/blob/main/docs/zh/s09-agent-teams.md)

| Layer | Storage | Content | Lifetime |
|-------|---------|---------|----------|
| L0 Working | In-memory (messages list) | Current conversation, active tools, todo list | Single session |
| L1 Episodic | SQLite + FTS5 | Session summaries, task outcomes, user preferences | Persistent |
| L2 Semantic | Qdrant (optional) | Vectorized knowledge, skill embeddings | Persistent |

**L1 schema** (`src/memory/episodic.py`):
```sql
CREATE TABLE sessions (id TEXT PRIMARY KEY, summary TEXT, outcome TEXT, created_at REAL, tags TEXT);
CREATE VIRTUAL TABLE sessions_fts USING fts5(summary, outcome, tags);
```

**Skill index** (`src/memory/skill_index.py`): maintains a registry of all skills in `skills/` with metadata for matching. Skills follow agentskills.io format.

---

### K. Learning Loop (`src/learning/`)

Three mechanisms:

> Refs: [Inside Hermes — learning loop, procedural memory, skill auto-creation and self-improvement](https://mranand.substack.com/p/inside-hermes-agent-how-a-self-improving) · [Hermes Agent README — periodic nudge, skill evolution, trajectory export](https://github.com/NousResearch/hermes-agent) · [hermes-agent-self-evolution — DSPy + GEPA skill optimization](https://github.com/NousResearch/hermes-agent-self-evolution) · [Hermes environments & trajectories](https://hermes-agent.nousresearch.com/docs/developer-guide/environments)

1. **Periodic Nudge** (`src/learning/nudge.py`): Every `NUDGE_INTERVAL` tool calls (default 15), aux LLM evaluates recent activity. Persists useful patterns to L1 memory.

2. **Skill Auto-Creation** (`src/learning/skill_creator.py`): After successful completion of a complex task, extract the workflow as a SKILL.md file in agentskills.io format. Register in skill index.

3. **Skill Self-Improvement** (`src/learning/skill_improver.py`): When a skill is used and the task succeeds, compare execution to skill definition. If the agent found a better approach, update the skill. Version-controlled with rollback on regression.

4. **Trajectory Recording** (`src/learning/trajectory.py`): Always-on recording of (user_msg, tool_calls, assistant_response) per turn. Export formats: ShareGPT JSON (for SFT), RL transitions (state, action, reward, next_state).

---

### L. Environment Injector (`src/core/env_injector.py`)

Injects task-type-aware context before each LLM call:

| Task type | Injected context | Max chars |
|-----------|-----------------|-----------|
| coding | git branch + recent commits + working tree status | 2000 |
| data_analysis | CSV/JSON column schemas in working directory | 2000 |
| file_ops | 2-level directory tree of working directory | 2000 |
| research | Recent search queries and results summary | 1500 |

Updated every turn, not just at session start.

---

### M. Eval Benchmark (`src/eval/`)

4 domains × 3 difficulty levels = 12 baseline tasks (extensible via `configs/eval_tasks.yaml`).

**Dual-channel evaluation:**
- Rule checks: code_executes, file_exists, output_matches_pattern
- LLM-as-Judge: quality, relevance, completeness scoring (temperature=0)

**Metrics:**
- `completion_rate` — tasks completed / total (per domain and overall)
- `avg_steps` — mean tool calls per task (efficiency)
- `recovery_rate` — tasks where agent hit an error but recovered / total errors
- `consistency_score` — 1 - normalized variance of completion rates across domains
- `skill_creation_count` — skills auto-created during benchmark run

---

### N. Provider System (`src/providers/`)

Model-agnostic design. Provider ABC with three implementations:

> Refs: [Hermes provider runtime resolution — 4-tier precedence (CLI → config → env → defaults), 3 API modes, credential pools](https://hermes-agent.nousresearch.com/docs/developer-guide/provider-runtime)

| Provider | API format | Models |
|----------|-----------|--------|
| DeepSeekProvider | OpenAI-compatible | deepseek-chat, deepseek-reasoner |
| OpenAICompatProvider | OpenAI chat.completions | Any OpenAI-compatible endpoint |
| AnthropicProvider | Anthropic messages | claude-sonnet, claude-opus (optional) |

Runtime resolution via `configs/local.yaml`. Hot-swap with no code change.

---

## Data flow: one complete turn

```
1. User types message
2. CLI captures → appends to conversation[]
3. context_assembler.assemble(tools, env_injector, conversation, task_type):
   ├── Zone 1: frozen system prompt (from cache — never changes)
   ├── Zone 2: tool schemas (names always + full schemas for discovered tools)
   ├── Zone 3: env_injector.get_context(task_type) → inject into latest user msg
   ├── Token budget check: usable = context_window - 33K - zones_1_2_tokens
   ├── Zone 4: if conversation > budget → context_engine.compress(conversation)
   │   ├── Layer 1: truncate tool results > 3000 chars
   │   ├── Layer 2: dedup unchanged file reads → stub
   │   ├── Layer 3: summarize earliest 1/3 via aux LLM (if circuit breaker closed)
   │   ├── Layer 4: full rewrite, keep last 5 turns (if still over)
   │   └── Layer 5: hard truncate to last 4 turns (emergency)
   └── Returns: [static] + [tools] + [env+conversation] = final messages[]
4. provider.stream(messages, tools.get_active_schemas())
   ├── chunk by chunk:
   │   ├── text token → yield Event("token")
   │   ├── tool_call_start → executor.prepare()
   │   └── tool_call_complete → permission_gate.check()
   │       ├── ALLOW → hooks.pre → execute → hooks.post → yield Event("tool_result")
   │       ├── ASK_USER → yield Event("permission_request") → wait
   │       └── BLOCK → yield Event("blocked")
5. trajectory.record_turn(user_msg, tool_calls, response)
6. budget.consume(1)
7. if stop_reason == "end_turn":
   ├── if turn_count % NUDGE_INTERVAL == 0: learning.nudge()
   ├── if task was complex and successful: learning.maybe_create_skill()
   └── yield Event("final_answer")
8. else: append tool_results to conversation[] → goto 3
```
