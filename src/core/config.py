"""Configuration loader — Pydantic models matching configs/default.yaml."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Nested config models
# ---------------------------------------------------------------------------

class ProviderConfig(BaseModel):
    name: str = "deepseek"
    api_key: str = ""
    keys: list[str] = Field(default_factory=list)
    strategy: str = "round_robin"
    cooldown_seconds: float = 60.0
    base_url: str = "https://api.deepseek.com/v1"
    model: str = "deepseek-chat"

    def effective_keys(self) -> list[str]:
        """Return the key list. ``keys`` wins over ``api_key`` when both are set."""
        if self.keys:
            return list(self.keys)
        if self.api_key:
            return [self.api_key]
        return []


class ProviderSection(BaseModel):
    name: str = "deepseek"
    api_key: str = ""
    keys: list[str] = Field(default_factory=list)
    strategy: str = "round_robin"
    cooldown_seconds: float = 60.0
    base_url: str = "https://api.deepseek.com/v1"
    model: str = "deepseek-chat"
    routes: dict[str, dict[str, str]] = Field(default_factory=dict)
    fallback: ProviderConfig | None = None
    auxiliary: ProviderConfig | None = None

    def effective_keys(self) -> list[str]:
        """Return the key list. ``keys`` wins over ``api_key`` when both are set."""
        if self.keys:
            return list(self.keys)
        if self.api_key:
            return [self.api_key]
        return []


class AgentSection(BaseModel):
    max_iterations: int = 90
    nudge_interval: int = 15
    fork_budget_allocation: int = 20


class ContextSection(BaseModel):
    output_reserve: int = 20_000
    buffer: int = 13_000
    max_compression_failures: int = 3


class PlanModeSection(BaseModel):
    save_dir: str = "plans/"
    auto_trigger_complexity: str = "complex"


class ToolTimeouts(BaseModel):
    bash: int = 60
    python_repl: int = 30
    web_search: int = 15
    default: int = 30


class ToolsSection(BaseModel):
    deferred: list[str] = Field(default_factory=lambda: ["enter_plan_mode"])
    timeouts: ToolTimeouts = Field(default_factory=ToolTimeouts)


class SemanticMemory(BaseModel):
    enabled: bool = False
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    collection: str = "agent_memory"


class MemorySection(BaseModel):
    sqlite_path: str = "data/sessions.db"
    semantic: SemanticMemory = Field(default_factory=SemanticMemory)


class LearningSection(BaseModel):
    skills_dir: str = "skills/"
    trajectories_dir: str = "trajectories/"
    auto_create_skills: bool = True
    auto_improve_skills: bool = True
    trajectory_recording: bool = True


class EvalSection(BaseModel):
    tasks_file: str = "configs/eval_tasks.yaml"
    results_dir: str = "benchmarks/"


class PermissionsSection(BaseModel):
    auto_allow: list[str] = Field(
        default_factory=lambda: [
            "read_file", "list_dir", "search_files", "web_search", "tool_search",
        ]
    )
    require_approval: list[str] = Field(
        default_factory=lambda: ["write_file", "edit_file", "bash", "python_repl"]
    )
    always_block: list[str] = Field(default_factory=lambda: ["delete_file"])
    yolo_mode: bool = False


class UISection(BaseModel):
    streaming: bool = True
    show_tool_calls: bool = True
    show_permission_prompts: bool = True
    debug: bool = False


class SearchSection(BaseModel):
    """Web search config. Tavily is the preferred provider."""
    provider: str = "tavily"  # "tavily" | "duckduckgo"
    tavily_api_key: str = ""
    max_results: int = 5
    timeout: int = 15


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

class AgentConfig(BaseModel):
    provider: ProviderSection = Field(default_factory=ProviderSection)
    agent: AgentSection = Field(default_factory=AgentSection)
    context: ContextSection = Field(default_factory=ContextSection)
    plan_mode: PlanModeSection = Field(default_factory=PlanModeSection)
    tools: ToolsSection = Field(default_factory=ToolsSection)
    memory: MemorySection = Field(default_factory=MemorySection)
    learning: LearningSection = Field(default_factory=LearningSection)
    eval: EvalSection = Field(default_factory=EvalSection)
    hooks: dict[str, Any] = Field(default_factory=dict)
    permissions: PermissionsSection = Field(default_factory=PermissionsSection)
    ui: UISection = Field(default_factory=UISection)
    search: SearchSection = Field(default_factory=SearchSection)

    @model_validator(mode="before")
    @classmethod
    def _strip_none_values(cls, data: Any) -> Any:
        """YAML parses commented-out sections as None — drop them so defaults apply."""
        if isinstance(data, dict):
            return {k: v for k, v in data.items() if v is not None}
        return data
    permissions: PermissionsSection = Field(default_factory=PermissionsSection)
    ui: UISection = Field(default_factory=UISection)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

_ENV_VAR_PATTERN = re.compile(r"\$\{(\w+)\}")


def _expand_env_vars(value: Any) -> Any:
    """Recursively expand ${ENV_VAR} references in strings."""
    if isinstance(value, str):
        def _replacer(match: re.Match) -> str:
            var_name = match.group(1)
            env_val = os.environ.get(var_name, "")
            if not env_val:
                pass  # missing env var → empty string, config validation catches it
            return env_val
        return _ENV_VAR_PATTERN.sub(_replacer, value)
    if isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env_vars(item) for item in value]
    return value


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base. Override wins on conflicts."""
    merged = dict(base)
    for key, val in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = _deep_merge(merged[key], val)
        else:
            merged[key] = val
    return merged


def load_config(
    default_path: str | Path = "configs/default.yaml",
    local_path: str | Path = "configs/local.yaml",
) -> AgentConfig:
    """Load config from default YAML, overlay local YAML, expand env vars."""
    default_path = Path(default_path)
    local_path = Path(local_path)

    if not default_path.exists():
        return AgentConfig()

    with open(default_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if local_path.exists():
        with open(local_path, "r", encoding="utf-8") as f:
            local_data = yaml.safe_load(f) or {}
        data = _deep_merge(data, local_data)

    data = _expand_env_vars(data)
    return AgentConfig.model_validate(data)
