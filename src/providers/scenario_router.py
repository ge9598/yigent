"""Scenario router — pick provider+model per task type.

Route keys follow CCR convention:
  - default: general-purpose
  - background: low-priority cheap work (compression, nudge, skill creation)
  - long_context: long-context workloads beyond the primary provider's limit
  - thinking: reasoning-heavy tasks using a thinking/reasoner model

Unknown task types fall back to ``default``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import LLMProvider


class ScenarioRouter:
    """Route requests to a provider+model based on task type."""

    def __init__(
        self,
        providers: dict[str, "LLMProvider"],
        routes: dict[str, dict[str, str]],
    ) -> None:
        if "default" not in routes:
            raise ValueError("ScenarioRouter must define a 'default' route")
        for task_type, route in routes.items():
            provider_name = route.get("provider")
            if provider_name not in providers:
                raise ValueError(
                    f"Route {task_type!r} references unknown provider "
                    f"{provider_name!r}. Known: {list(providers)}"
                )
            if "model" not in route:
                raise ValueError(
                    f"Route {task_type!r} missing 'model' field"
                )
        self._providers = dict(providers)
        self._routes = {k: dict(v) for k, v in routes.items()}

    def select(self, task_type: str) -> tuple["LLMProvider", str]:
        """Return (provider, model) for a task type. Falls back to default."""
        route = self._routes.get(task_type, self._routes["default"])
        provider = self._providers[route["provider"]]
        model = route["model"]
        return provider, model

    def list_routes(self) -> dict[str, dict[str, str]]:
        """Return a deep copy of the routes for introspection."""
        return {k: dict(v) for k, v in self._routes.items()}
