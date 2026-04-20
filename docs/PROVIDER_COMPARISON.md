# Provider layer — comparison against reference projects

> Scope: `src/providers/` only. Compares Yigent's multi-provider design against
> five reference projects and calls out concrete borrowings.
>
> Written after adding `AnthropicCompatProvider` (commit following 50 passing tests).

---

## Yigent's current design

- One ABC: `LLMProvider.stream_message(messages, model, tools, temperature) -> AsyncGenerator[StreamChunk]`.
- One unified event stream: `token | tool_call_start | tool_call_delta | tool_call_complete | done`.
- Three concrete providers:
  - `OpenAICompatProvider` — `openai` SDK + custom `base_url`. Covers OpenAI, DeepSeek, Qwen, MiniMax (OpenAI route), local vLLM/Ollama.
  - `DeepSeekProvider` — thin subclass, just defaults.
  - `AnthropicCompatProvider` — `anthropic` SDK + custom `base_url`. Covers native Claude API and MiniMax `/anthropic`. Translates OpenAI-dialect messages → Anthropic (system hoisting, tool_result blocks, tool_use blocks) and Anthropic SSE events → unified `StreamChunk`.
- Internal dialect: OpenAI-style (`role` / `content` / `tool_calls` / `tool_call_id`).
- `resolver.py` — string→class registry, config picks `name: <provider>`, supports `fallback` and `auxiliary` slots. Single-key per provider. No retry, no key rotation, no cross-provider failover.

---

## Reference projects

### 1. Claude Code Router (musistudio/claude-code-router) — 32.5k★

- **Shape:** local HTTP proxy (TypeScript, default port 3456). Translation logic split into a dependency package `@musistudio/llms` (~300★). Monorepo `cli/ + server/ + shared/ + ui/`.
- **Provider coverage:** ~10+ via a pluggable `Transformer` class. One transformer per provider (Anthropic, DeepSeek, Gemini, OpenRouter, Groq, Volcengine, SiliconFlow, …) plus utility transformers (`maxtoken`, `tooluse`, `reasoning`, `enhancetool`, `customparams`).
- **Translation contract:** four hooks per transformer — `transformRequestIn / transformRequestOut / transformResponseIn / transformResponseOut` — plus an `endPoint` string. Unified dialect is OpenAI-flavoured.
- **Streaming tool calls:** `SSEParserTransform → rewriteStream → SSESerializerTransform` pipeline. Server re-emits translated SSE. Fragile when providers deliver `input_json_delta` chunks out of order (open bug reports).
- **Routing:** 4-level hierarchy evaluated in order: `Router.default` → project config → custom JS (`CUSTOM_ROUTER_PATH`) → scenario categories (`default / background / think / longContext / webSearch / image`). `longContext` uses tiktoken cl100k_base threshold. `/model provider,model` for manual switch. **Per-request dynamic routing.**
- **Fallback:** none built-in. No key rotation, no 429 handling, no cross-provider failover. "Stack with LiteLLM" is the community answer.
- **Pointers:** `packages/server/src/utils/router.ts`, `packages/server/src/agents/`, `musistudio/llms/src/transformer/*.transformer.ts`.

### 2. OpenClaude (Gitlawb/openclaude) — 22.4k★

- **Shape:** fork of Claude Code, not a bridge. TypeScript + Bun. Internal dialect forced to OpenAI; everything else adapted client-side ("behaviour not identical across all backends" per README).
- **Interesting bit:** headless gRPC server (`src/proto/openclaude.proto`) streams text chunks, tool calls, and permission requests over gRPC instead of SSE.
- **Relevance to Yigent:** low. Opposite design (fork-and-rewrite), not directly reusable.
- Sibling projects: **antomix** (25★, reverse proxy via `ANTHROPIC_BASE_URL` override), **meridian** (Claude Max subscription bridge). Neither has scale.

### 3. LiteLLM (BerriAI/litellm)

- **Shape:** SDK (`litellm/`) + optional proxy (`litellm/proxy/`), same translation layer. Entry point `litellm.completion()` / `acompletion()` in `main.py`; `get_llm_provider()` in `utils.py` resolves model-string → provider; dispatch through `BaseLLMHTTPHandler` in `llms/custom_httpx/llm_http_handler.py`. Translation is **client-side**; proxy just adds auth/budgets/rate-limits.
- **Provider abstraction:** `BaseConfig` subclass per provider with `transform_request()` / `transform_response()`. Almost all via raw `httpx` — LiteLLM owns the wire format. ~100+ providers.
- **Anthropic bridge:** `llms/anthropic/chat/transformation.py` for non-streaming plus an experimental pass-through adapter exposing `/v1/messages` for any backend (`llms/anthropic/experimental_pass_through/adapters/`).
- **Streaming tool calls:** `CustomStreamWrapper` normalizes all providers to OpenAI format by **indexed-dict accumulation** — each chunk merged into a per-index buffer keyed by `tool_calls[i].index`, handles out-of-order chunks. For Anthropic output path (`AnthropicStreamWrapper`): tracks `current_content_block_type`, `current_content_block_index`, `chunk_queue` (deque for atomic stop→start transitions), `tool_name_mapping` (restores names truncated to OpenAI's 64-char limit).
- **Streaming regressions:** function `_translate_streaming_openai_chunk_to_anthropic` near line 1403 has been a recurring source of bugs (issues #25321, #25561, #22296). This is hard code even when done by experts.
- **Routing/fallback:** `litellm.Router` in `router.py`. `model_list` with multiple deployments per logical model (key rotation equivalent). `num_retries` + exponential backoff. `fallbacks=[{model: [list]}]`. 429 → immediate cooldown + skip to `order=2`. Known hole: `usage-based-routing-v2` ignores `retry_after` header.
- **Pointers:** `litellm/main.py`, `litellm/router.py`, `litellm/utils.py::get_llm_provider`, `litellm/llms/anthropic/chat/transformation.py`, `litellm/llms/anthropic/experimental_pass_through/adapters/streaming_iterator.py`, `litellm/litellm_core_utils/streaming_handler.py`.

### 4. Hermes Agent provider runtime

- **Shape:** single shared resolver (`hermes_cli/runtime_provider.py` + `hermes_cli/auth.py::resolve_provider()`) reused by CLI, gateway, cron, auxiliary tasks. Returns `(provider, api_mode, base_url, api_key, source, metadata)`.
- **`api_mode` is a first-class concept** — 3-way switch at the resolver layer:
  - `chat_completions` (OpenAI-compatible default)
  - `anthropic_messages` (native Anthropic wire format)
  - `codex_responses` (OpenAI Responses API)
  Wire format decoupled from provider identity — `(provider=openrouter, api_mode=anthropic_messages)` is valid.
- **Precedence (5 tiers):** CLI/runtime → `config.yaml` → env vars → provider defaults → auto-resolution. `config` beats env deliberately, to prevent shell state from silently overriding saved settings.
- **Credential pools:** per provider, strategies `fill_first / round_robin / least_used / random`. Flow: pick key → on 429 retry once → second 429 rotate → exhaustion → `fallback_model` → on 402 billing rotate + 24h cooldown → on 401 mark expired. **Scoped routing:** `OPENROUTER_API_KEY` only goes to `openrouter.ai`, no cross-leak.
- **3-tier fallback:** (1) pool rotation within provider, (2) primary-model fallback to another `(provider, model)`, (3) auxiliary task fallback (vision/compression/web-extract resolve independently).
- **Tests:** `tests/test_fallback_model.py`.

### 5. OpenRouter (hosted service)

- API is strict OpenAI `chat/completions` shape, plus an Anthropic-compatible endpoint at `/api/v1/messages`. Internal translation + routing is server-side. `Auto Exacto` re-orders providers per request using tool-call-quality signals. Provider failover is built in. Streaming SSE uses OpenAI shape regardless of backend — OpenRouter owns translation for ~60 providers.
- **Value as reference:** the wire-level contract. Not the code structure (closed).

---

## Key contrasts

### Where Yigent is simpler / better

1. **Translation lives where it belongs.** Yigent keeps Anthropic translation inside `AnthropicCompatProvider`. No separate transformer package (CCR's `musistudio/llms`), no bolted-on pass-through adapter (LiteLLM's experimental `/v1/messages`). One class, one responsibility. Adding a provider means implementing `stream_message()`, not learning a 4-method transformer contract.
2. **Async generator streaming from day one.** LiteLLM's `CustomStreamWrapper` is a retrofit on a sync SDK — hence the long tail of streaming-tool-call bugs (#4495, #5063, #12463, #25321, #25561). Yigent's explicit `tool_call_start / _delta / _complete` events make boundaries explicit instead of reconstructed.
3. **No proxy hop.** CCR and antomix want a localhost:3456 proxy and eat SSE serialization twice. Yigent talks to provider SDKs directly.

### Where Yigent is missing

1. **No credential pools.** Hermes's 4-strategy pool with scoped routing is the single biggest gap. Even with 2 keys on the same provider, pools give free reliability.
2. **No cross-provider fallback.** `configs/default.yaml` has a `fallback` slot, but `resolver.py` only uses it on **init failure**, not on runtime 5xx/429. LiteLLM Router and Hermes both auto-route on failure.
3. **No 429 / 402 / 401 handling.** Every reference project handles these; Yigent only has timeouts. One 429 loop burns a session.
4. **Static routing.** Provider is picked once at session start. CCR's per-scenario routing (`default / background / think / longContext`) is genuinely useful — the periodic-nudge aux-LLM call should hit a cheap model, not the main one. Yigent's design doc calls for this (`auxiliary` slot) but execution-path routing is absent.
5. **No "API mode" concept.** Yigent conflates wire format with provider class. MiniMax exposes both OpenAI and Anthropic routes — today you pick the route by picking the provider class (coarse). Hermes's `(provider, api_mode)` tuple is cleaner.
6. **Internal-dialect choice is debatable.** Yigent standardizes on OpenAI-style internally, matching CCR/LiteLLM. Right call for the current provider mix (OpenAI-compat dominates). But Hermes goes the other way (Anthropic canonical) which preserves `thinking` blocks and `content_block` semantics on round-trips. Worth revisiting if Yigent ever cares about Claude extended thinking.

### Things I'd steal (in order of ROI)

1. **Credential pool + scoped routing** (from Hermes). Config: `keys: [key1, key2, ...]` + `strategy: round_robin`. On 429/402/401 rotate key, update cooldowns. ~150 lines in `resolver.py` + a new `KeyPool` class. Unit-testable with fake responses carrying different status codes. Biggest free reliability win.
2. **Scenario-based routing** (from CCR). Add `routes: {default, background, long_context, thinking}` in config; map `task_type` → route key in the capability router. Periodic nudge hits `background`. Plan-mode deliberation hits `thinking`. ~50 lines. Makes cost/quality knobs explicit and visible in config.
3. **Indexed tool-call accumulator** (from LiteLLM's `CustomStreamWrapper`). My current `AnthropicCompatProvider` keys `_ToolUseAccumulator` by `event.index` from Anthropic's SDK, which assumes in-order block delivery. MiniMax's Anthropic route has been observed to reorder. Safer to key by `tool_use_id` and only emit `tool_call_complete` when `content_block_stop` for that index arrives. Small change inside the provider, no API break.

### Things I'd not steal

- **4-hook transformer interface** (musistudio/llms). Overkill for 3 providers. Revisit only if coverage reaches ~10.
- **Router + BaseConfig + Adapter + Proxy 4-layer stack** (LiteLLM). Right for a library selling itself as "drop-in for 100+ providers"; wrong for a harness that wants one honest abstraction.
- **gRPC as IPC boundary** (OpenClaude). Irrelevant for a Python in-process harness. Yigent's async generator already serves the UI directly.

---

## Sources

- [claude-code-router (musistudio)](https://github.com/musistudio/claude-code-router)
- [claude-code-router CLAUDE.md](https://github.com/musistudio/claude-code-router/blob/main/CLAUDE.md)
- [claude-code-router Transformers docs](https://musistudio.github.io/claude-code-router/docs/server/config/transformers/)
- [musistudio/llms](https://github.com/musistudio/llms)
- [LiteLLM ARCHITECTURE.md](https://github.com/BerriAI/litellm/blob/main/ARCHITECTURE.md)
- [LiteLLM AnthropicStreamWrapper](https://github.com/BerriAI/litellm/blob/main/litellm/llms/anthropic/experimental_pass_through/adapters/streaming_iterator.py)
- [LiteLLM Router docs](https://docs.litellm.ai/docs/routing)
- [LiteLLM streaming regression #25321](https://github.com/BerriAI/litellm/issues/25321)
- [LiteLLM streaming fallback hole #22296](https://github.com/BerriAI/litellm/issues/22296)
- [Hermes Agent provider runtime](https://hermes-agent.nousresearch.com/docs/developer-guide/provider-runtime)
- [Hermes Agent credential pools](https://hermes-agent.nousresearch.com/docs/user-guide/features/credential-pools)
- [Hermes Agent fallback providers](https://hermes-agent.nousresearch.com/docs/user-guide/features/fallback-providers)
- [OpenClaude](https://github.com/Gitlawb/openclaude)
- [OpenRouter tool-calling](https://openrouter.ai/docs/guides/features/tool-calling)
- [OpenRouter provider routing](https://openrouter.ai/docs/guides/routing/provider-selection)
