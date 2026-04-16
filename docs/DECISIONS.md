# Technical decisions

Each entry: what we chose → why → what we rejected → why not.

---

## D1: Language — Python (not TypeScript)

**Chose:** Python 3.11+ with full async/await

**Why:** All existing projects (太行 Agent, SAC RL, MLP surrogate, LoRA fine-tuning) are Python. Interview code review is in Python. The LLM/ML ecosystem (PyTorch, transformers, vllm) is Python-first.

**Rejected:** TypeScript — Claude Code's choice. Would require relearning the ecosystem. The project goal is demonstrating agent architecture understanding, not TypeScript skill.

---

## D2: Framework — From scratch (not LangGraph)

**Chose:** Custom async generator-based agent loop, no framework dependency.

**Why:** 太行 project already proves LangGraph proficiency (81K lines, 31 MCP tools). This project proves understanding of what LangGraph does under the hood. Interview differentiator: "I've built agents both ways — with a framework and without."

**Rejected:** LangGraph — declarative state graph is optimal for fixed workflows (like simulation pipelines) but too rigid for general-purpose agents where execution paths are fully dynamic. Also rejected: AutoGen, CrewAI — multi-agent frameworks that abstract away the loop internals we want to demonstrate.

---

## D3: Agent loop — Async generator (not sync while loop)

**Chose:** `async def agent_loop(...) -> AsyncGenerator[Event, None]`

**Why:** Async generators yield events mid-stream (tokens, tool results, permission requests) to the UI layer while continuing execution. Enables streaming tool execution — start tool prep while model is still outputting. Claude Code uses the same pattern (TypeScript `async function*`).

**Rejected:** Synchronous while loop (Hermes pattern) — blocks during API calls, cannot stream tokens to UI during execution, cannot parallelize tool calls.

---

## D4: Provider — Multi-provider with hot-swap (not single-vendor)

**Chose:** Provider ABC with DeepSeek (primary) + OpenAI-compatible (secondary). Runtime hot-swap via config.

**Why:** General-purpose agent should not be locked to one vendor. DeepSeek V3 is cost-effective and matches 太行 project. OpenAI-compatible interface covers Qwen, GLM, local models via Ollama/vLLM. Hermes proves this pattern works (200+ models).

**Rejected:** Anthropic-only (Claude Code pattern) — vendor lock-in, higher cost, cannot demonstrate model-agnostic design.

---

## D5: Memory — SQLite+FTS5 for L1 (not JSON flat files)

**Chose:** SQLite with FTS5 virtual table for episodic memory. Qdrant for L2 semantic memory.

**Why:** FTS5 enables full-text search across session history — critical for cross-session recall. SQLite is file-based (zero config), supports WAL mode for concurrent reads, and is the Hermes Agent's proven choice. JSON flat files (Claude Code's choice) are not searchable without loading everything into memory.

**Rejected:** PostgreSQL — too heavy for a personal project. JSON files (CC pattern) — no search capability. Pure Qdrant — vector search alone misses keyword-exact matches.

---

## D6: Context compression — 5-layer progressive with dynamic thresholds (not fixed)

**Chose:** Five compression layers (tool truncation → file dedup → early summarization → full rewrite → hard cutoff). Thresholds dynamically computed as `model_context_window - output_reserve - buffer`.

**Why:** Claude Code source analysis shows this exact pattern. Dynamic thresholds mean switching to a longer-context model automatically delays compression, preserving more reasoning history. Fixed thresholds would require manual reconfiguration per model.

**Rejected:** Single-layer summarization — too aggressive, loses important context. No compression (rely on large context) — fails on long tasks. Hermes's pluggable engine ABC is good architecture but their default implementation is single-layer.

---

## D7: Permissions — 4-layer chain + Plan mode enforcement (not prompt-based)

**Chose:** Schema validation → tool self-check → hook check → permission level classification. Plan mode blocks writes at the permission layer.

**Why:** Claude Code source analysis reveals the critical insight: prompt-based safety ("please don't write files during planning") is unreliable because models can ignore instructions. System-level blocking is 100% reliable. The 4-layer chain ensures no single point of failure.

**Rejected:** Prompt-only constraints — unreliable. Single approval gate (Hermes pattern) — catches dangerous commands but misses the Plan mode use case.

---

## D8: Skill format — agentskills.io (not proprietary)

**Chose:** agentskills.io open standard for skill definitions.

**Why:** Cross-platform compatibility (works with Hermes, Claude Code, Cursor). Community ecosystem of existing skills. Hermes proves auto-creation and self-improvement work well with this format.

**Rejected:** Proprietary format — no ecosystem benefit. Plain text recipes — not machine-parseable for auto-loading.

---

## D9: Evaluation — Built-in benchmark (not external)

**Chose:** Internal 4-domain × 3-difficulty benchmark with LLM-as-Judge + rule checks.

**Why:** No existing framework evaluates general-purpose agent cross-domain consistency. SWE-bench only covers coding. Our benchmark measures what the JD asks for: "generalization to untrained scenarios." Dual-channel evaluation (rules + LLM judge) mirrors our 太行 multimodal evaluation (keyword matching + manual review, Macro F1 39.5% → 97.5%).

**Rejected:** SWE-bench only — coding only, not general-purpose. No evaluation — can't quantify the "generalization" claim.

---

## D10: Trajectory export — ShareGPT + RL formats (not just logs)

**Chose:** Always-on trajectory recording. Export as ShareGPT JSON (for SFT fine-tuning) and RL transitions (state, action, reward, next_state).

**Why:** Seed JD mentions "推动通用人工智能" — training next-gen tool-calling models needs high-quality trajectory data. This positions the project as infrastructure, not just a product. Hermes Agent includes this (Atropos RL integration) and it's a clear differentiator over Claude Code which has no training data export.

**Rejected:** No export — misses the research infrastructure angle. Logs-only — not structured enough for direct use in training pipelines.

---

## D11: Tool loading — Deferred with ToolSearch (not eager)

**Chose:** Tool names + short descriptions loaded at session start. Full JSON schemas loaded on-demand via ToolSearch. Some tools (PlanMode) marked `should_defer` and not in initial list at all.

**Why:** With 15+ tools, full schemas in the system prompt consume 5-10K tokens. Deferred loading saves context space. ToolSearch forces the model to actively discover tools, demonstrating "meta-cognition." Claude Code source uses this exact pattern.

**Rejected:** Eager loading (Hermes pattern: all tools registered and exposed at import time) — wastes context when most tools won't be used in a given session.

---

## D12: Hook system — Lifecycle events (not ad-hoc)

**Chose:** 8 lifecycle events (session_start, pre/post_tool_use, pre/post_compression, plan_approved, budget_warning, session_end). Hooks are Python callables or shell scripts.

**Why:** Hooks make the agent a platform, not a tool. Users can customize behavior (auto-lint after writes, log all searches, block specific commands) without modifying core code. Claude Code source shows hooks are essential for CI/CD integration and enterprise adoption.

**Rejected:** No hooks (Hermes) — limits extensibility. Hardcoded behavior — not configurable.
