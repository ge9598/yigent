# Day 1 + Day 2 学习笔记

12 个源文件。每节先讲「这个模块/类在系统里扮演什么角色」，再讲内部的技术细节和踩过的坑。

---

## Day 1：类型系统 + Provider 层

### 1. `src/core/types.py` — 全系统共享类型

**它是干什么的**：定义跨模块流转的数据对象。Agent Loop 里流出的 `Event`、Provider 吐出的 `StreamChunk`、模型调用的 `ToolCall`、工具返回的 `ToolResult`、LLM 见到的 `Message`、工具的 `ToolSchema`、权限枚举 —— 全部在这一个文件里。任何两个模块要交换数据，都通过这些类型。

这里有 **12 个类 + 2 个枚举**，按用途分三组：

| 组 | 类 | 表达形式 |
|---|---|---|
| 对外协议 | `Message`, `ToolCallDict`, `FunctionCall` | TypedDict |
| 运行时对象 | `ToolCall`, `ToolResult`, `StreamChunk`, 7 个 Event 类, `ToolContext`, `ToolDefinition` | dataclass |
| 配置/契约 | `ToolSchema` | Pydantic BaseModel |

三种表达形式不是随性 —— 每种匹配一类场景：

```python
# TypedDict — 需要是「字典本身」
class Message(TypedDict, total=False):
    role: str
    content: str | None
    tool_calls: list[ToolCallDict]
```

`Message` 最终要直接塞进 OpenAI SDK 的 `messages=[...]` 参数里，那个接口吃 dict。用 dataclass 每次都得 `asdict()` 转一层，白搭 CPU。TypedDict 就是「带类型标注的 dict」，本质是 dict，零开销。

```python
# dataclass — 运行时轻量对象
@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    status: Literal["pending", "running", "completed", "failed"] = "pending"
```

`Event`, `ToolCall`, `ToolResult` 每一次对话都会产生大量实例。dataclass 就是个 `__init__` + `__eq__` + `__repr__` 的自动生成，没有校验开销。用 Pydantic 每次 `ToolCall(...)` 都跑一遍字段校验 —— 在这种场景浪费。

```python
# Pydantic — 需要校验的外部输入
class ToolSchema(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]
    permission_level: PermissionLevel = PermissionLevel.READ_ONLY
```

`ToolSchema` 的 `parameters` 是 JSON Schema，要保证结构正确。`AgentConfig` 从 YAML 加载，要过校验。这些是信任边界 —— 外部数据进来必须校验一次。Pydantic 就是为这个设计的。

**踩过的坑**：`ToolContext` 字段类型引用了 `PlanMode`、`AgentConfig`、`ToolRegistry`，而这三个在 types.py 加载时还没定义（它们在 Layer 3、Layer 5 的文件里）。直接写会循环依赖。

解决：

```python
from __future__ import annotations       # 让所有 annotation 变成延迟求值的字符串
from typing import TYPE_CHECKING

if TYPE_CHECKING:                         # 这个 block 运行时不执行，只给 mypy 看
    from src.core.config import AgentConfig
    from src.core.plan_mode import PlanMode
    from src.tools.registry import ToolRegistry

@dataclass
class ToolContext:
    plan_mode: PlanMode                   # 不报错 — 字符串形式的类型标注
    config: AgentConfig
    registry: ToolRegistry
```

这是 Python 3.10+ 的标准套路。`__future__ annotations` 改变 PEP 563 的行为，所有函数/dataclass 的类型标注变成字符串，只在被显式访问（比如 `typing.get_type_hints()`）时才 resolve。`TYPE_CHECKING` 永远 `False` —— 运行时那块 import 不执行。

---

### 2. `src/core/config.py` — 配置加载管道

**它是干什么的**：把 YAML 配置文件变成类型安全的 `AgentConfig` 对象。系统每处需要配置（provider 地址、budget 大小、权限策略、搜索 API key）都从这个对象读。启动时加载一次、传递到各处。

**核心类结构**：`AgentConfig` 是顶层，下面嵌套 11 个 section 模型：

```
AgentConfig
├── ProviderSection   → 主/副/辅助 provider 三组
├── AgentSection      → max_iterations, nudge_interval
├── ContextSection    → 压缩阈值参数
├── PlanModeSection   → 保存目录
├── ToolsSection      → deferred 列表、timeouts
├── MemorySection     → SQLite 路径、Qdrant 配置
├── LearningSection   → skills/trajectories 目录
├── EvalSection       → 评估配置
├── PermissionsSection→ auto_allow / require_approval / always_block 列表
├── UISection         → CLI 显示选项
└── SearchSection     → Tavily / DuckDuckGo 选择 + API key
```

全 Pydantic。每个字段都有默认值 —— 没有 `local.yaml` 也能跑。

**加载流程**四步：

```
default.yaml 读成 dict
  ↓  _deep_merge(local.yaml 的 dict)
  ↓  _expand_env_vars 递归展开 ${VAR}
  ↓  AgentConfig.model_validate(final_dict)
AgentConfig 实例
```

**细节 1：环境变量展开**。default.yaml 里写 `api_key: "${DEEPSEEK_API_KEY}"`。PyYAML 读出来是字面字符串 `"${DEEPSEEK_API_KEY}"`。需要正则扫一遍：

```python
_ENV_VAR_PATTERN = re.compile(r"\$\{(\w+)\}")

def _expand_env_vars(value):
    if isinstance(value, str):
        return _ENV_VAR_PATTERN.sub(
            lambda m: os.environ.get(m.group(1), ""), value)
    if isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env_vars(i) for i in value]
    return value
```

递归处理 —— YAML 嵌套结构里任何位置的字符串都可能有变量引用。

**细节 2：深度合并**。`local.yaml` 只会覆盖个别字段（比如 Tavily key）：

```yaml
# local.yaml
search:
  tavily_api_key: "tvly-..."
```

如果用 `dict.update()`，整个 `search` section 被覆盖 —— default 里的 `provider: tavily`、`timeout: 15` 全丢了。必须递归合并：

```python
def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, val in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = _deep_merge(merged[key], val)
        else:
            merged[key] = val
    return merged
```

**踩过的坑：YAML 注释 section 变成 None**。default.yaml 里 `hooks:` 下面全部注释掉：

```yaml
hooks:
  # pre_tool_use:
  #   - "scripts/check_dangerous_commands.sh"
```

PyYAML 解析的结果是 `{"hooks": None}`。Pydantic 看到 None 传给 `dict[str, Any]` 字段立刻 ValidationError。

修法：加一个 `model_validator(mode="before")`，在 Pydantic 正式校验前过滤掉 None：

```python
@model_validator(mode="before")
@classmethod
def _strip_none_values(cls, data):
    if isinstance(data, dict):
        return {k: v for k, v in data.items() if v is not None}
    return data
```

`mode="before"` 很关键 —— 必须在字段校验之前跑，不然 None 已经进入字段校验管道了。

---

### 3. `src/core/stubs.py` — Phase 2 占位

**它是干什么的**：为 Phase 2 才会实现的模块提供「空壳」实现，让 Phase 1 的代码能用完整的接口签名写出来。里面 4 个类：

- `NullContextAssembler` — 会被 `src/context/assembler.py` 替换（5-zone 上下文组装）
- `NullHookSystem` — 会被 `src/safety/hook_system.py` 替换（生命周期 hooks）
- `NullLearningLoop` — 会被 `src/learning/nudge.py` 替换（周期性 nudge + 技能创建）
- `NullTrajectoryRecorder` — 会被 `src/learning/trajectory.py` 替换（轨迹记录）

所有方法都是 no-op：

```python
class NullHookSystem:
    async def fire(self, event_name: str, **data: Any) -> None:
        pass
```

**为什么不直接在 Phase 1 不带这些参数**：agent_loop 是核心，签名定下来就尽量别改。方案对比：

| 方案 | 后果 |
|---|---|
| Phase 1 签名不带 hooks/learning/trajectory | Phase 2 加回来时 agent_loop 全改 |
| Phase 1 传 `None`，运行时判空 | 到处 `if hooks is not None: await hooks.fire(...)`，丑 |
| **Null Object** | agent_loop 写 `await hooks.fire(...)`，Phase 2 一行不改 |

这是 "Null Object Pattern"（Martin Fowler 经典补丁）。代价是 Phase 1 多这 50 行占位，收益是 Phase 2 落地时 agent_loop 零改动。

---

### 4. `src/core/iteration_budget.py` — 迭代次数预算

**它是干什么的**：给整个 agent 会话设一个硬上限，防止模型陷入死循环烧钱。默认 90 次迭代（一次 LLM 调用+工具执行算一次）。多 agent 场景下，父 agent 通过 `allocate(n)` 把预算切一块给子 agent。

`IterationBudget` 这一个类，接口简洁：

```python
class IterationBudget:
    async def consume(self, n: int = 1) -> int: ...
    async def allocate(self, n: int) -> IterationBudget: ...
    @property
    def remaining(self) -> int: ...
    @property
    def is_warning(self) -> bool: ...
    @property
    def is_exhausted(self) -> bool: ...
```

**`consume` 和 `allocate` 的语义区别**：

- `consume(n)`：自己花掉 n 次。典型调用是 agent loop 每轮结束 `await budget.consume(1)`。
- `allocate(n)`：从当前预算切出 n 次，给子 agent 一个独立的 `IterationBudget(total=n)`。父预付 —— 子 agent 后续 `consume` 不再回到父。

两种语义：「共享池」（子消费影响父） vs 「父预付」（子独立）。选后者因为简单可预测：父 agent 调 `allocate(20)` 之后立刻知道自己还剩多少，不会被子 agent 的行为再影响。

**并发安全**。在 async 代码里，两个协程并发调 `consume(5)`：

```python
# 时间线可能是这样：
coro A: read spent=10
coro B: read spent=10           ← 同时读到相同值
coro A: write spent=15
coro B: write spent=15          ← 丢失 A 的写入，实际应该是 20
```

必须用锁：

```python
async def consume(self, n: int = 1) -> int:
    async with self._lock:                    # asyncio.Lock，不是 threading.Lock
        if self._spent + n > self._total:
            self._spent = self._total
            raise BudgetExhausted(...)
        self._spent += n
        return self.remaining
```

`asyncio.Lock` 和 `threading.Lock` 不一样 —— 前者是协程让渡调度，不涉及操作系统线程。在单线程事件循环里用它就对了。

**踩过的坑：`is_warning` 的边界**。设计是「剩余 ≤ 20% 时报警」。最早写成：

```python
return self.remaining < self._total * 0.2
```

测试：total=5, remaining=1。`1 < 5 * 0.2 = 1 < 1.0 = False`。但这正是 20%、正该报警的时候 —— 边界值被漏掉了。

修正：

```python
return 0 < self.remaining <= self._total * 0.2
```

前半部分 `0 < remaining` 排除 exhausted 的情况（exhausted 不等于 warning，它们在视图上是不同状态）。后半部分 `<=` 包含边界值。

**还有个坑：exhausted 时把 spent 锁死**。`consume` 检测到会超的时候：

```python
if self._spent + n > self._total:
    self._spent = self._total     # ← 不加这行会出事
    raise BudgetExhausted(...)
```

如果不加这行，`_spent` 还是超之前的值。调用者 catch 异常后继续用，`remaining` 还是正的 —— 明明已经宣告耗尽了。这是防御性编程：让无效状态无法表达。

---

### 5. `src/providers/base.py` + `openai_compat.py` + `deepseek.py` + `resolver.py` — Provider 层

**它是干什么的**：把不同厂商的 LLM API 统一成一个接口，让 agent loop 不需要关心底层用 DeepSeek 还是 OpenAI 还是 Qwen。

**四个文件的分工**：

- `base.py` — 定义抽象基类 `LLMProvider`，只有一个方法 `stream_message()`，返回 `AsyncGenerator[StreamChunk, None]`
- `openai_compat.py` — OpenAI-兼容接口的实现。OpenAI 本身、DeepSeek、Qwen、Moonshot、本地 vLLM 都走这里
- `deepseek.py` — `OpenAICompatProvider` 的薄子类，只改 base_url 和默认 model
- `resolver.py` — 从 `AgentConfig` 拿到 provider 配置 → 实例化对应类 → 返回。带 fallback 逻辑

**继承层次**：

```
LLMProvider (ABC, 只定义契约)
  └── OpenAICompatProvider (真正的实现)
       └── DeepSeekProvider (改几个默认值)
```

以后加新 provider：Qwen 就是另一个子类（改 base_url），Anthropic 是另一个独立实现（不兼容 OpenAI 格式，继承 LLMProvider）。

**ABC vs Protocol 的选择**。Python 里定义契约有两种方式：

```python
# ABC：运行时强制
class LLMProvider(ABC):
    @abstractmethod
    async def stream_message(self, ...): ...

# Protocol：结构鸭子类型，只静态检查
class LLMProvider(Protocol):
    async def stream_message(self, ...) -> AsyncGenerator[...]: ...
```

ABC 胜在「运行时契约」—— 漏了方法的子类实例化立刻 TypeError。Protocol 只有 mypy/pyright 会报错，运行时不拦。对会被多人扩展的契约（未来可能加 AnthropicProvider、GeminiProvider），ABC 的硬检查更可靠。

**OpenAICompatProvider 的核心：流式 tool call 解析**。这是 Day 1 最难的一段。

`stream_message()` 要把 OpenAI 的 SSE 流转成统一的 `StreamChunk` 事件序列。一次 LLM 响应会产生几类 chunk：

```
StreamChunk(type="token", data="Hello")         # 文本 token
StreamChunk(type="tool_call_start", data={...}) # 新工具调用开始
StreamChunk(type="tool_call_delta", data={...}) # 工具参数增量
StreamChunk(type="tool_call_complete", data=ToolCall(...))  # 工具调用完整解析
StreamChunk(type="done", data="stop")           # 流结束
```

**难点：tool call 的 `arguments` 是分块到达的 JSON 字符串片段**。OpenAI 返回的 streaming 格式里：

```json
chunk 1: {"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"read_file","arguments":"{\"pa"}}]}}
chunk 2: {"delta":{"tool_calls":[{"index":0,"function":{"arguments":"th\": \"x.py\"}"}}]}}
chunk 3: {"finish_reason":"tool_calls"}
```

单个 chunk 的 arguments 是 `{"pa` —— **不是合法 JSON**，`json.loads` 会抛异常。必须按 `index` 累积所有片段，等 `finish_reason == "tool_calls"` 再解析。

实现用 accumulator dict：

```python
@dataclass
class _ToolCallAccumulator:
    index: int
    id: str = ""
    name: str = ""
    arguments_buffer: str = ""

accumulators: dict[int, _ToolCallAccumulator] = {}

async for chunk in stream:
    delta = chunk.choices[0].delta
    if delta.tool_calls:
        for tc_delta in delta.tool_calls:
            idx = tc_delta.index
            if idx not in accumulators:
                # 首次出现 → 记 id/name，发 tool_call_start 事件
                acc = _ToolCallAccumulator(index=idx)
                accumulators[idx] = acc
                if tc_delta.id: acc.id = tc_delta.id
                if tc_delta.function and tc_delta.function.name:
                    acc.name = tc_delta.function.name
                yield StreamChunk(type="tool_call_start", ...)
            # 累积 arguments 片段（首次也要进来）
            if tc_delta.function and tc_delta.function.arguments:
                accumulators[idx].arguments_buffer += tc_delta.function.arguments
    
    if chunk.choices[0].finish_reason == "tool_calls":
        # 按 index 排序 yield，保证 LLM 的顺序 = 执行的顺序
        for acc in sorted(accumulators.values(), key=lambda a: a.index):
            tool_call = self._parse_accumulated(acc)     # 此时才 json.loads
            yield StreamChunk(type="tool_call_complete", data=tool_call)
```

**四个容易错的点**：

1. **按 `index` 索引，不是 `id`**。id 只在首个 chunk 出现；index 每个 chunk 都有。用 id 索引会遗漏后续 chunk。
2. **首次 chunk 也可能带 arguments 片段**。OpenAI 有时 id/name/arguments 首个 chunk 就一起给了一部分。不能「首次只记 id/name，后续才累积 arguments」。
3. **累积到 `finish_reason` 才解析**。中途的 buffer 是残缺 JSON。
4. **解析失败的降级**：
   ```python
   try:
       arguments = json.loads(acc.arguments_buffer)
   except json.JSONDecodeError:
       logger.warning(...)
       arguments = {"_raw": acc.arguments_buffer}
   ```
   直接抛异常会导致整轮对话失败。降级成 `{"_raw": ...}` 保留原文，让上层决定怎么处理。

**Resolver 的 fallback 逻辑**：

```python
def resolve_provider(config: AgentConfig) -> LLMProvider:
    try:
        return _build_provider(name=section.name, api_key=section.api_key, ...)
    except Exception:
        if section.fallback is None:
            raise
        logger.warning("Primary failed, trying fallback")
    return _build_provider(name=fb.name, ...)
```

Primary provider 失败（比如 DeepSeek key 没配）自动走 fallback。production 部署重要 —— 主服务挂了不至于系统崩溃。

---

## Day 2：工具系统 + Plan Mode

### 6. `src/tools/registry.py` — 工具注册中心

**它是干什么的**：所有工具（read_file、bash、web_search 等 11 个）都在这里登记。Agent 启动时模型看到哪些工具、工具怎么被找到、工具的完整 schema 什么时候进入 LLM 的 `tools` 参数 —— 全由这个类决定。

`ToolRegistry` 是这整个系统的**元数据**，不存储任何业务逻辑，只管「哪些工具存在、哪些被激活了、query 能匹配哪些」。

**三个模式捏一起**：self-registration + deferred loading + ToolSearch。

**模式 A — self-registration**（Hermes 风格）

每个工具文件在模块顶层自己注册：

```python
# file_ops.py
from src.tools.registry import register, ToolDefinition

async def _read_file_handler(path: str, ...) -> str:
    ...

register(ToolDefinition(
    name="read_file",
    description="...",
    handler=_read_file_handler,
    schema=ToolSchema(...),
))
```

`src/tools/__init__.py` 只做 import：

```python
from .registry import ToolRegistry, get_registry, register
from . import file_ops, coding, interpreter, search, plan_tools
```

`import file_ops` 时 Python 执行整个模块文件，顶层 `register(...)` 自动调用。加新工具只要在 `__init__.py` 加一行 import，注册自动完成。不需要手动维护工具列表。

**模式 B — deferred loading**（Claude Code 风格）

工具有两种状态，`_activated: set[str]` 记录哪些激活了：

```python
def register(self, tool: ToolDefinition) -> None:
    self._tools[tool.name] = tool
    if not tool.schema.deferred:
        self._activated.add(tool.name)         # 非 deferred 立即激活
```

`get_active_schemas()` 返回激活的工具的完整 schema —— 这是传给 LLM 的 `tools` 参数。`get_initial_tools()` 返回所有非 deferred 工具的名字+描述 —— 用于系统提示文本。Deferred 工具两个里都不出现，LLM 不知道它们存在。

**为什么要 defer**：
- Phase 1 11 个工具，schema 总计 3-5K token。还好。
- 真实场景 50+ 工具，schema 30K+ token，占掉系统提示半壁江山。
- 工具太多模型 attention 分散，选错工具的概率上升。
- Deferred 让 LLM 「按需发现」—— 演示「meta-cognition」，大多数时候只激活真需要的几个。

Phase 1 只 `enter_plan_mode` 一个 deferred 工具，演示机制。Phase 2/3 可以把更多工具标 deferred。

**模式 C — ToolSearch 本身是一个工具**

ToolSearch 在 registry.py 底部自己注册：

```python
async def _tool_search_handler(ctx: ToolContext, query: str) -> str:
    schemas = ctx.registry.tool_search(query)
    # 格式化返回给 LLM
    return "\n".join([f"• {s.name}: {s.description}" for s in schemas])

register(ToolDefinition(
    name="tool_search",
    handler=_tool_search_handler,
    schema=ToolSchema(name="tool_search", ...),
    needs_context=True,
))
```

`tool_search(query)` 在 registry 内部做模糊匹配 + 激活：

```python
def tool_search(self, query: str, limit: int = 10) -> list[ToolSchema]:
    q = query.lower().strip()
    if not q:
        return []
    matches = []
    for t in self._tools.values():
        if q in t.name.lower() or q in t.description.lower():
            self._activated.add(t.name)      # ← 关键：激活它
            matches.append(t.schema)
    return matches
```

模型调用 `tool_search("plan")` 后：
- 内部 `_activated` 加入 `enter_plan_mode`
- 下一轮 LLM 调用的 `tools` 参数里就有 `enter_plan_mode` 的 schema
- 模型现在能调用它了

ToolSearch 是**元工具**，递归自指：用工具来发现工具。这是 Claude Code 最巧妙的设计之一。

**模块级单例的取舍**：

```python
_default_registry = ToolRegistry()

def get_registry() -> ToolRegistry:
    return _default_registry

def register(tool: ToolDefinition) -> None:
    _default_registry.register(tool)
```

单例的代价是测试里状态污染（两个测试都 `register()` 同名工具会打架）。但 self-registration 模式天然要求单例 —— 模块被 import 时的 register 调用需要有目标。折中：类本身 `ToolRegistry` 可独立实例化，测试里 `ToolRegistry()` 建新的。默认单例只是方便。

---

### 7. `src/core/types.py::ToolContext` — 工具运行时依赖注入

**它是干什么的**：工具 handler 需要访问的「运行时依赖」（PlanMode 实例、当前 AgentConfig、用户回调、working directory...）都打包进这个 dataclass。handler 签名里第一个参数是 `ToolContext`，工具需要什么从里面取。

```python
@dataclass
class ToolContext:
    plan_mode: PlanMode              # enter_plan_mode/exit_plan_mode 用
    registry: ToolRegistry           # tool_search 用（查其他工具）
    config: AgentConfig              # web_search 用（取 Tavily key）
    working_dir: Path                # 给所有需要 cwd 的工具
    user_callback: UserCallback | None = None   # ask_user 用
    session_id: str | None = None    # plan_mode 保存文件名要用
```

**为什么要有这个类**：

问题背景 —— `plan_tools.py` 里 `enter_plan_mode` handler 要调 `plan_mode.enter()`。但：

1. `plan_mode` 实例是 CLI 启动时才创建的
2. `plan_tools.py` 在 CLI 启动前 import（触发 self-registration）
3. import 时 plan_mode 还不存在

四种解法对比：

| 方案 | 问题 |
|---|---|
| 模块级全局 + `bind_plan_mode(pm)` | 全局状态、测试难清理 |
| 每个 handler 签名都带 ctx | 简单工具（read_file）被迫加不用的参数 |
| 两种 handler 签名共存 | **选这个** |
| 用 ContextVar 传递 | 过度设计 |

最终：`ToolDefinition.needs_context: bool = False`，handler 按需带 ctx。

```python
# 简单工具 — 不需要 ctx
async def read_file_handler(path: str, offset=0, limit=2000) -> str: ...

register(ToolDefinition(name="read_file", handler=read_file_handler, ...))
# needs_context 默认 False

# 需要 ctx 的工具
async def enter_plan_mode_handler(ctx: ToolContext) -> str:
    ctx.plan_mode.enter(...)

register(ToolDefinition(
    name="enter_plan_mode",
    handler=enter_plan_mode_handler,
    needs_context=True,
))
```

Executor 调用时按 flag 分支：

```python
handler, needs_ctx = registry.get_handler(name)
if needs_ctx:
    result = await handler(ctx, **args)
else:
    result = await handler(**args)
```

**Phase 1 里 needs_context=True 的工具**：`tool_search`（要访问 registry）、`enter_plan_mode`、`exit_plan_mode`、`ask_user`（要调 user_callback）、`web_search`（要 API key）。其他 6 个工具纯粹的 stateless 操作，不需要 ctx。

---

### 8. `src/tools/file_ops.py` — 4 个文件操作工具

**它是干什么的**：提供 read_file、write_file、list_dir、search_files。这是 agent 最基础的「感知和动作」—— 能看文件内容、能写文件、能浏览目录、能搜代码。Phase 1 的 demo 跑通几乎必然用到这些。

**4 个工具的职责**：

- `read_file(path, offset=0, limit=2000)` — 读文本文件，带行号。拒绝二进制。支持分页（大文件可以一次读一段）。
- `write_file(path, content)` — 覆盖写入。父目录自动创建。原子操作。
- `list_dir(path=".", depth=2)` — 递归列目录，树形展示。跳过 `.git` 等垃圾目录。
- `search_files(pattern, path=".", glob="**/*")` — 正则搜索文件内容，max 50 条匹配。跳过二进制文件。

**细节 1：二进制检测**

```python
def _is_binary_file(path: Path) -> bool:
    with open(path, "rb") as f:
        chunk = f.read(1024)
    if b"\x00" in chunk:
        return True
    try:
        chunk.decode("utf-8")
    except UnicodeDecodeError:
        return True
    return False
```

两个启发：
- **null 字节**：C 字符串规范禁止，文本文件几乎不会有。`.pyc`, `.exe`, `.png` 都会有。
- **UTF-8 解码失败**：非 UTF-8 编码的文本（比如 GBK 中文）也会触发，但为了代码简单接受这个误伤（Phase 1 假设 UTF-8 世界）。

只检查前 1KB 避免大文件全读。

**细节 2：原子写入**

```python
p = Path(path)
p.parent.mkdir(parents=True, exist_ok=True)
tmp = p.with_suffix(p.suffix + ".tmp")
with open(tmp, "w", encoding="utf-8") as f:
    f.write(content)
os.replace(tmp, p)               # 原子替换
```

直接 `open(p, "w")` 的风险：写到一半进程被 kill → 文件被截断只剩一半内容。`os.replace` 在 POSIX 和 Windows 上都是原子的 —— 要么完整替换，要么失败保留原样，永远不会有「半个文件」的中间态。

**细节 3：行号格式**

```python
numbered = [f"{i + offset + 1:>6}\t{line.rstrip(chr(10))}" for i, line in enumerate(selected)]
```

`{:>6}` 右对齐 6 位数字，够处理 10万行以内的文件。用 tab 分隔。模型看到这个格式能精确知道每行的行号（后面 edit_file 工具需要这种精度）。

---

### 9. `src/tools/coding.py` — bash 执行工具

**它是干什么的**：提供一个通用的 shell 命令执行工具。Agent 要编译代码、跑测试、查 git 状态、装包 —— 都走它。

**跨平台的核心难题**：在 Windows 上执行 `echo "hello"` vs `echo hello` 行为不同（引号规则不同），`ls` vs `dir`，路径 `/` vs `\`。如果放任模型自己对付 `cmd` 和 bash 的差异，它会频繁出错。

解决：**强制用 bash**。Windows 上用 Git-for-Windows 自带的 bash.exe：

```python
def _find_bash() -> str | None:
    if sys.platform == "win32":
        bash = shutil.which("bash")
        # 关键：跳过 WSL 的 System32 下的 bash.exe
        if bash and "System32" not in bash:
            return bash
        for candidate in (
            r"C:\Program Files\Git\bin\bash.exe",
            r"C:\Program Files (x86)\Git\bin\bash.exe",
        ):
            if os.path.exists(candidate):
                return candidate
        return None
    return shutil.which("bash")
```

**踩过的坑：WSL bash.exe**

Windows 10+ 自带 WSL，安装 WSL 后 `C:\Windows\System32\bash.exe` 会在 PATH 里。`shutil.which("bash")` 优先找到它。但它启动的是 WSL Linux 子系统 —— 可能要密码、路径映射成 `/mnt/c/`、没装 python 等各种诡异问题。我们要的是 git-bash，不是 WSL。

用「路径里不含 System32」这个启发过滤掉 WSL 的 bash。Git-for-Windows 永远装在 `Program Files\Git\` 下，不会在 System32。

**子进程调用方式**：

```python
proc = await asyncio.create_subprocess_exec(
    *argv,
    stdout=asyncio.subprocess.PIPE,
    stderr=asyncio.subprocess.STDOUT,    # stderr 合并到 stdout
)
```

`create_subprocess_shell` vs `create_subprocess_exec`：
- `shell` 用系统默认 shell（Windows 是 cmd），我们要的是 bash，不用
- `exec` 显式指定 argv

`stderr=STDOUT` 合并两个流到一个管道。模型看到错误消息和正常输出混在一起，像真实终端那样，自然解读。

**超时和清理**：

```python
try:
    stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
except asyncio.TimeoutError:
    try:
        proc.kill()
    except ProcessLookupError:
        pass
    await proc.wait()                    # 等收尸，避免僵尸进程
    return f"Error: timed out after {timeout}s"
```

`wait_for` 超时后，proc 还在跑。必须 `kill()` + `wait()`。`wait()` 不收的话 Windows 上会留僵尸进程（asyncio 后台有警告）。

**危险命令检测**（Phase 1 只警告不阻断）：

```python
_DANGEROUS_PATTERNS = [
    re.compile(r"\brm\s+-rf?\s+/(?:\s|$)"),       # rm -rf /
    re.compile(r":\(\)\s*\{.*\|\s*:.*\}"),         # fork bomb
    re.compile(r"\bdd\s+if=/dev/(?:zero|random|urandom)\s+of=/dev/(?:sd|hd|nvme)"),
    re.compile(r"\bmkfs\."),
    re.compile(r">\s*/dev/(?:sd|hd|nvme)"),
]
```

Phase 1 检测到这些只输出 warning header，还是执行。真正的阻断在 Phase 2 的 permission gate 第二层（tool self-check）—— 届时会拒绝执行并要求用户显式确认。

---

### 10. `src/tools/interpreter.py` — Python REPL

**它是干什么的**：让 agent 能直接运行 Python 代码片段。比 bash 轻量，没有 shell 转义问题，数据处理场景常用。

只有一个工具 `python_repl(code, timeout=30)`。**Stateless** —— 每次调用开一个全新的 Python 子进程执行完退出。第一次 `x = 5` 第二次看不到 `x`。

**关键决策 1：tempfile 而非 `python -c`**

```python
fd, tmp_path = tempfile.mkstemp(suffix=".py", text=True)
try:
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(code)
    proc = await asyncio.create_subprocess_exec(sys.executable, tmp_path, ...)
finally:
    os.unlink(tmp_path)
```

为啥不用 `python -c "code"`：多行代码 + 嵌套引号在 shell 里转义是噩梦。模型写 `print("hello \"world\"")` 一进 shell 就炸。写 tempfile 永远正确。

`mkstemp` 返回 fd + path，`os.fdopen(fd, "w")` 包装成文件对象写入。完成后 `os.unlink` 清理。`try/finally` 保证 tempfile 永远删 —— 哪怕 wait_for 超时或用户 Ctrl+C。

**关键决策 2：无状态**

持久化 REPL 状态（保留命名空间）要 pickle 整个 globals dict，边界 case 多：无法 pickle 的对象（lambda、打开的文件、线程锁）全要特殊处理。Phase 1 先保守 —— 每次新进程，纯粹、可预测。

需要状态的场景可以在一次 `python_repl` 调用里塞多行代码（把所有逻辑打包一起）。

---

### 11. `src/tools/search.py` — web_search 工具

**它是干什么的**：让 agent 能查互联网信息。查文档、查库用法、查报错信息 —— 基础能力。

一个工具 `web_search(query, num_results=5)`，两个后端：

1. **Tavily**（优先）— Anthropic 和 LangChain 的默认推荐，专为 LLM 场景设计
2. **DuckDuckGo instant answer**（fallback）— 免 key，但效果弱

**为什么 Tavily**：

- 返回 `answer` 字段（一句话总结） + `results` 列表
- 每条 result 有 `title`/`url`/`content`/`score`，其中 `content` 是已经裁到 300 字的关键片段
- 天然适合给 LLM 消化。DuckDuckGo 的 instant answer API 经常返回空结果

**为什么还留 DDG fallback**：

```python
if cfg.provider == "tavily":
    if not cfg.tavily_api_key:
        return "Error: Tavily selected but api_key not set..."
    try:
        return await _tavily_search(...)
    except httpx.HTTPError as e:
        logger.warning("Tavily failed (%s), falling back to DDG", e)
        return await _duckduckgo_search(...)
return await _duckduckgo_search(...)
```

Tavily 挂了 / 配额用光 / 网络问题 → 整个 agent 不能就因此死掉。DDG 能当应急救火。

**API key 存哪里**：

- `default.yaml` 里写 `tavily_api_key: "${TAVILY_API_KEY}"` —— 要求先 `export TAVILY_API_KEY=...`，本地开发烦
- `configs/local.yaml` 明文 key，`.gitignore` 屏蔽这个文件 —— 本地最方便

Phase 1 选后者。正式部署才用环境变量。

---

### 12. `src/core/plan_mode.py` — Plan Mode 状态机

**它是干什么的**：实现「先规划再执行」的三阶段循环。Agent 遇到复杂任务时先进入 plan 模式，只能读不能写，先把方案想清楚、跟用户确认，然后退出 plan 模式开始真正执行。

`PlanMode` 这一个类，管理会话级状态（每个会话一个实例）：

```python
class PlanMode:
    ALLOWED_TOOLS: frozenset[str] = frozenset({
        "read_file", "list_dir", "search_files",
        "web_search", "tool_search", "ask_user",
        "exit_plan_mode",            # 特殊：write 权限但白名单放行
    })
    
    def enter(self, session_id: str) -> None: ...
    def exit(self, approved: bool, plan_content: str = "") -> str: ...
    
    @property
    def is_active(self) -> bool: ...
    
    def is_tool_allowed(self, tool_name: str) -> bool:
        if not self._active:
            return True              # 非 plan 模式：全放行
        return tool_name in self.ALLOWED_TOOLS
```

**核心设计原则：permissions are architecture, not suggestions**

Plan mode 禁写**不是**通过在 system prompt 里加一句「请不要修改文件」。那种 prompt 层的约束：

- 模型会忘
- 模型会狡辩（「我觉得这个写操作是必要的...」）
- 某些上下文会让模型忽略（jailbreak、对抗性提示）

正确做法：在 **权限层** 拦截。当 plan_mode.is_active == True：

```
模型输出：调用 write_file
  ↓
Executor 拿到 tool call
  ↓
Executor 问 plan_mode.is_tool_allowed("write_file") → False
  ↓
Executor 直接返回 ToolResult(is_error=True, content="blocked: plan mode active")
  ↓
write_file handler 根本没执行
```

模型无法绕过，因为阻断是系统级的，不是 prompt 级的。这是整个项目的核心哲学（DECISIONS.md D7）。

**特殊放行 `exit_plan_mode`**：

这个工具的 `permission_level` 是 WRITE（它确实要写 `plans/*.md`）。但如果不放行，Plan mode 就是单程票 —— 进去出不来。白名单特殊包含它。

这类「例外」我选择显式写在 `ALLOWED_TOOLS` 里，比在 executor 里搞特判好 —— 规则集中在一个 set 里，读代码的人一眼看到。

**保存格式**：

```python
path = self._save_dir / f"{session_id}_{ts}.md"
path.write_text(f"# Plan\n\nSession: {session_id}\nCreated: {ts}\n\n---\n\n{plan_content}")
```

文件名有 timestamp，避免同会话多次规划互相覆盖。

---

### 13. `src/tools/plan_tools.py` — 3 个特殊工具

**它是干什么的**：提供 plan mode 的进出口 + ask_user 机制。这三个工具需要访问 `PlanMode` 实例和用户回调，所以都 `needs_context=True`。

三个工具：

- `enter_plan_mode()` — **唯一 deferred 的工具**。模型必须先 `tool_search("plan")` 发现它才能调用。
- `exit_plan_mode(approved, plan_content)` — 退出 plan mode。approved=True 保存 plan_content 到文件。
- `ask_user(question)` — 向用户抛问题并等回答。用 `ctx.user_callback` 回调 CLI。

**为什么 enter_plan_mode 要 deferred**：

Plan mode 是重操作 —— 进去之后写操作全阻断，要完整走完「规划 → 审批 → 退出」流程。如果这工具在初始列表里直接可用，模型看到简单任务也会被吸引去用它（工具多的时候模型会「贪」）。

Deferred 机制强制模型先明确表达「我需要规划」—— 调用 `tool_search("plan")` 本身就是一次语义决策。发现 enter_plan_mode 存在后再用，这一层门槛避免误触发。

**ask_user 的回调机制**：

```python
async def _ask_user_handler(ctx: ToolContext, question: str) -> str:
    if ctx.user_callback is None:
        return "Error: user interaction not available (running headless?)"
    return await ctx.user_callback(question)
```

`user_callback` 在 CLI 启动时注入到 ToolContext。CLI 里的实现大致是：

```python
async def user_callback(question: str) -> str:
    print(f"\n[Agent asks]: {question}\n")
    return await prompt_session.prompt_async("Your answer: ")
```

headless 运行（比如 API 模式、eval 模式）时 callback 为 None，`ask_user` 返回错误。这让工具在所有场景下都能安全调用。

**timeout=300 的取舍**：

```python
schema=ToolSchema(
    name="ask_user",
    timeout=300,          # 5 分钟
    ...
)
```

其他工具 timeout 都是 10-60 秒，ask_user 给了 5 分钟 —— 人类回答要时间。但这意味着如果用户忘了回复，agent 会卡 5 分钟。Phase 1 这是合理权衡；Phase 2 可能加一个「pending 检测 + 更长超时 + 提醒」机制。

---

## 三条通用工程教训

### A. 循环依赖的标准解法

本项目 `ToolContext` 要引用 `PlanMode`、`ToolRegistry`、`AgentConfig`，这三者又间接依赖 types.py。形成环。

```python
from __future__ import annotations        # PEP 563
from typing import TYPE_CHECKING

if TYPE_CHECKING:                          # 运行时永远 False
    from src.core.config import AgentConfig
    from src.core.plan_mode import PlanMode
    from src.tools.registry import ToolRegistry
```

两个开关组合：

- `__future__ annotations` — 所有函数/类的 annotation 变成字符串，不在 class 定义时 resolve
- `TYPE_CHECKING` guarded import — 只给 mypy/pyright 看，运行时不执行

这套解法 Python 3.10+ 标准。记住它 —— 写大一点的项目必然遇到循环依赖。

### B. 分层依赖 DAG，层内并发实现

Day 1+2 的实现顺序严格按层：

```
Layer 0:  types.py, config.py, stubs.py
Layer 1:  iteration_budget.py, providers/base.py
Layer 2:  providers/{openai_compat, deepseek, resolver}.py
Layer 3:  tools/registry.py
Layer 4:  tools/{file_ops, coding, interpreter, search}.py
Layer 5:  core/plan_mode.py
Layer 6:  tools/plan_tools.py
```

层级是严格 DAG —— Layer N 只能依赖 Layer < N。Layer 2 三个文件互相不依赖，可以并发写（实际也是 3 个 Write 调用一起发）。

好处：写到某个文件发现要 import 的东西不存在，只有两种可能：层次设计错了，或者代码里有循环依赖。永远不会有「我这层应该独立，结果依赖还没实现」的惊喜。

### C. 验证驱动，不是单元测试

Day 1、Day 2 结束都是用 `python -c "..."` 跑一段集成验证脚本，不写 pytest 文件。原因：

1. Phase 1 接口还在快速迭代，pytest 测试写完明天就过时
2. 集成验证是「端到端跑通」的黑盒证据，比「每个函数按设计工作」的白盒证据更能证明「这玩意真的能用」
3. 三周交付 Phase 1 的时间压力下，维护 pytest 文件成本太高

真正的 pytest 单元测试留到 Phase 3（Week 3）—— 那时接口定型、模块稳定、写测试不会明天就过时。

这不是说测试不重要，而是说**测试粒度和时机要匹配代码成熟度**。

---

## 收尾：我们现在有什么

一个**能被调用的工具库**（11 个工具，含 deferred 机制示例） + 一个**能被驱动的 Provider**（流式解析完整） + 一个**能切换状态的 Plan Mode**（系统级强制） + 一套**依赖注入机制**（ToolContext）。

Day 3 把它们串成 agent —— async generator ReAct loop + 并行 Streaming Executor + Rich CLI。

---

## 延伸阅读

- **Null Object Pattern**：Martin Fowler 的 [Refactoring to Patterns](https://www.google.com/search?q=null+object+pattern+fowler)
- **PEP 525 Async Generators** + **PEP 563 Postponed Evaluation of Annotations**
- **Claude Code 源码分析**（CLAUDE.md 里有链接）
- **Anthropic Advanced Tool Use**：[anthropic.com/engineering/advanced-tool-use](https://www.anthropic.com/engineering/advanced-tool-use)
- **Tavily API Docs**：[docs.tavily.com](https://docs.tavily.com)
