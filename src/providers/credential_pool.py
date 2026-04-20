"""Multi-key credential pool with 4 rotation strategies and 429 cooldown.

Strategies:
    round_robin — cursor advances each acquire, skipping cooling keys
    fill_first  — stick to the first available key until it errors
    least_used  — pick the available key with minimum usage (tie → insertion order)
    random      — uniform random pick among available keys (deterministic with seed)

Error policy (user-confirmed 2026-04-20):
    HTTP 429 → key enters cooldown for `cooldown_seconds` (default 60s)
    any other status → rotate to next key, do NOT cool

No permanent invalidation — bad keys are removed manually via `remove_key`.
"""

from __future__ import annotations

import random as _random
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass


_STRATEGIES = {"round_robin", "fill_first", "least_used", "random"}


@dataclass
class _KeyState:
    usage: int = 0
    cooldown_until: float = 0.0  # monotonic timestamp; 0 = not cooling


class CredentialPool:
    """Thread-safe pool of API keys with configurable rotation and 429 cooldown."""

    def __init__(
        self,
        keys: list[str],
        strategy: str = "round_robin",
        cooldown_seconds: float = 60.0,
        seed: int | None = None,
    ) -> None:
        if not keys:
            raise ValueError("CredentialPool needs at least one key")
        if strategy not in _STRATEGIES:
            raise ValueError(f"Unknown strategy {strategy!r}; expected one of {sorted(_STRATEGIES)}")

        self._lock = threading.Lock()
        self._states: OrderedDict[str, _KeyState] = OrderedDict(
            (k, _KeyState()) for k in keys
        )
        self._strategy = strategy
        self._cooldown_seconds = cooldown_seconds
        self._rr_cursor = 0  # round_robin cursor
        self._sticky: str | None = None  # fill_first current key
        self._rng = _random.Random(seed)

    # -- public API ----------------------------------------------------------

    def acquire(self) -> str:
        """Return a usable key according to the current strategy.

        Raises:
            RuntimeError: if every key is currently in cooldown.
        """
        with self._lock:
            available = self._available_keys()
            if not available:
                raise RuntimeError("CredentialPool: all keys in cooldown")

            if self._strategy == "round_robin":
                key = self._pick_round_robin(available)
            elif self._strategy == "fill_first":
                key = self._pick_fill_first(available)
            elif self._strategy == "least_used":
                key = self._pick_least_used(available)
            elif self._strategy == "random":
                key = self._rng.choice(available)
            else:  # pragma: no cover — guarded at init and set_strategy
                raise ValueError(f"Unknown strategy {self._strategy!r}")

            self._states[key].usage += 1
            return key

    def mark_error(self, key: str, status: int) -> None:
        """Record an error against `key`. 429 triggers cooldown; others just rotate."""
        with self._lock:
            state = self._states.get(key)
            if state is None:
                return  # key already removed; nothing to do

            if status == 429:
                state.cooldown_until = time.monotonic() + self._cooldown_seconds

            # For fill_first: any error (429 or not) advances past the sticky
            # key to the next one in insertion order.
            if self._strategy == "fill_first" and self._sticky == key:
                keys_list = list(self._states.keys())
                start = keys_list.index(key) + 1
                self._sticky = None
                for offset in range(len(keys_list)):
                    candidate = keys_list[(start + offset) % len(keys_list)]
                    if candidate == key:
                        continue
                    self._sticky = candidate
                    break

            # For round_robin: no action needed on error.
            #   - 429: key enters cooldown; `_available_keys()` filters it out
            #     on the next acquire, so the cursor naturally walks past it.
            #   - non-429: the cursor was already advanced past this key by the
            #     acquire() that returned it (or a later acquire() has already
            #     moved further). Resetting the cursor here would regress it.

    def add_key(self, key: str) -> None:
        """Add a new key to the pool (no-op if already present)."""
        with self._lock:
            if key not in self._states:
                self._states[key] = _KeyState()

    def remove_key(self, key: str) -> None:
        """Remove `key` from the pool. Raises ValueError on removing the last key."""
        with self._lock:
            if key not in self._states:
                return
            if len(self._states) == 1:
                raise ValueError("CredentialPool: cannot remove last key")
            del self._states[key]
            if self._sticky == key:
                self._sticky = None
            # Clamp round_robin cursor into range.
            if self._rr_cursor >= len(self._states):
                self._rr_cursor = 0

    def set_strategy(self, strategy: str) -> None:
        """Swap strategy at runtime. Resets sticky/cursor bookkeeping."""
        if strategy not in _STRATEGIES:
            raise ValueError(f"Unknown strategy {strategy!r}; expected one of {sorted(_STRATEGIES)}")
        with self._lock:
            self._strategy = strategy
            self._sticky = None
            self._rr_cursor = 0

    def list_keys(self) -> list[str]:
        """Return keys in insertion order (snapshot)."""
        with self._lock:
            return list(self._states.keys())

    def status(self) -> dict[str, dict[str, float | int | bool]]:
        """Return per-key status snapshot: usage, cooling, seconds_until_ready."""
        with self._lock:
            now = time.monotonic()
            return {
                key: {
                    "usage": state.usage,
                    "cooling": state.cooldown_until > now,
                    "seconds_until_ready": max(0.0, state.cooldown_until - now),
                }
                for key, state in self._states.items()
            }

    # -- internals -----------------------------------------------------------

    def _available_keys(self) -> list[str]:
        now = time.monotonic()
        return [k for k, s in self._states.items() if s.cooldown_until <= now]

    def _pick_round_robin(self, available: list[str]) -> str:
        keys_list = list(self._states.keys())
        n = len(keys_list)
        # Walk from cursor forward; return first available.
        for offset in range(n):
            idx = (self._rr_cursor + offset) % n
            candidate = keys_list[idx]
            if candidate in available:
                self._rr_cursor = (idx + 1) % n
                return candidate
        # Unreachable: caller already checked `available` is non-empty.
        raise RuntimeError("CredentialPool: all keys in cooldown")  # pragma: no cover

    def _pick_fill_first(self, available: list[str]) -> str:
        if self._sticky is not None and self._sticky in available:
            return self._sticky
        # Pick first available in insertion order.
        for key in self._states:
            if key in available:
                self._sticky = key
                return key
        raise RuntimeError("CredentialPool: all keys in cooldown")  # pragma: no cover

    def _pick_least_used(self, available: list[str]) -> str:
        # Iterate in insertion order so ties resolve to earliest-inserted.
        best: str | None = None
        best_usage = -1
        for key in self._states:
            if key not in available:
                continue
            usage = self._states[key].usage
            if best is None or usage < best_usage:
                best = key
                best_usage = usage
        if best is None:
            raise RuntimeError("CredentialPool: all keys in cooldown")  # pragma: no cover
        return best
