"""Failure-tracking circuit breaker for LLM-based compression layers.

Each compression layer that calls the auxiliary LLM (layers 3 and 4) wraps
its call through a CircuitBreaker. After ``threshold`` consecutive failures
the breaker trips and ``is_open`` returns True; the engine then skips that
layer and tries the next. A successful call resets the count.

Why per-layer breakers (not global):
    Layer 3 (early-turn summarization) and layer 4 (full rewrite) call the
    aux LLM with very different prompt sizes. A 4xx on layer 4 doesn't
    necessarily mean layer 3 will fail too. Independent counters give the
    engine more options before falling back to the free hard-truncate layer.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CircuitBreaker:
    """Trips open after N consecutive failures; closes again on first success."""

    threshold: int = 3
    failures: int = 0

    @property
    def is_open(self) -> bool:
        return self.failures >= self.threshold

    def record_success(self) -> None:
        self.failures = 0

    def record_failure(self) -> None:
        self.failures += 1

    def reset(self) -> None:
        self.failures = 0
