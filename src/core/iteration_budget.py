"""Iteration budget — truly shared across parent and child agents.

Per CLAUDE.md ("IterationBudget is shared across parent and all child
agents") and ARCHITECTURE.md §A. Implementation: every IterationBudget
instance holds a reference to the same ``_SharedCounter``; consume() on
parent or any child decrements the same number. Optional ``local_cap``
on a child handle additionally limits how much THIS handle is allowed to
consume from the shared pool, useful for "this fork shouldn't burn more
than 30 iterations of the shared 90".
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass


class BudgetExhausted(Exception):
    """Raised when the iteration budget is fully consumed."""


@dataclass
class _SharedCounter:
    """The actual shared state — a single counter both parent and children
    decrement against. Wrapped in IterationBudget handles for the public API."""
    total: int
    spent: int = 0
    lock: asyncio.Lock | None = None

    def __post_init__(self) -> None:
        if self.lock is None:
            self.lock = asyncio.Lock()


class IterationBudget:
    """Public handle to a shared iteration budget.

    Multiple IterationBudget instances can share the same underlying
    ``_SharedCounter`` — see ``allocate()``. ``local_cap`` adds a per-handle
    ceiling on consumption (None means no extra limit beyond the shared total).
    """

    def __init__(
        self,
        total: int = 90,
        enable_warning: bool = True,
        *,
        _counter: _SharedCounter | None = None,
        local_cap: int | None = None,
    ) -> None:
        if _counter is None:
            _counter = _SharedCounter(total=total)
        self._counter = _counter
        self._enable_warning = enable_warning
        self._local_cap = local_cap
        self._local_spent = 0

    # -- properties ----------------------------------------------------------

    @property
    def total(self) -> int:
        """For child handles with a local_cap, total is the cap (what THIS
        handle can spend). For unbounded handles, it's the shared total."""
        if self._local_cap is not None:
            return self._local_cap
        return self._counter.total

    @property
    def shared_total(self) -> int:
        """The shared pool total, regardless of local_cap."""
        return self._counter.total

    @property
    def remaining(self) -> int:
        """Remaining budget visible to THIS handle.

        Equals min(shared_remaining, local_cap_remaining). The minimum is
        what determines whether the next consume() succeeds.
        """
        shared_remaining = max(0, self._counter.total - self._counter.spent)
        if self._local_cap is None:
            return shared_remaining
        local_remaining = max(0, self._local_cap - self._local_spent)
        return min(shared_remaining, local_remaining)

    @property
    def is_warning(self) -> bool:
        """True when remaining <= 20% of total (but not exhausted) and warning is enabled."""
        if not self._enable_warning:
            return False
        if self.total == 0:
            return False
        return 0 < self.remaining <= self.total * 0.2

    @property
    def is_exhausted(self) -> bool:
        return self.remaining <= 0

    # -- mutations -----------------------------------------------------------

    async def try_consume(self, n: int = 1) -> bool:
        """Soft consume: returns True if successful, False if budget exhausted."""
        async with self._counter.lock:
            if self._counter.spent + n > self._counter.total:
                return False
            if self._local_cap is not None and self._local_spent + n > self._local_cap:
                return False
            self._counter.spent += n
            self._local_spent += n
            return True

    async def consume(self, n: int = 1) -> int:
        """Consume *n* iterations. Returns remaining. Raises on exhaustion."""
        async with self._counter.lock:
            if self._counter.spent + n > self._counter.total:
                self._counter.spent = self._counter.total
                raise BudgetExhausted(
                    f"Shared budget exhausted: {self._counter.total} iterations used"
                )
            if self._local_cap is not None and self._local_spent + n > self._local_cap:
                raise BudgetExhausted(
                    f"Local cap reached: this handle limited to {self._local_cap}"
                )
            self._counter.spent += n
            self._local_spent += n
            return self.remaining

    async def allocate(self, n: int) -> IterationBudget:
        """Return a CHILD handle that shares the same underlying counter,
        with an additional local cap of ``n``.

        Unlike the old pre-deduct model, this does NOT charge ``n`` against
        the shared pool upfront — the child can consume up to ``n`` of the
        shared budget, but if the parent burns through the shared total
        first, the child's consume() will see exhaustion too. That's the
        point: parent and children share a single budget.
        """
        async with self._counter.lock:
            # Sanity: can't promise more than the shared pool can deliver.
            if n > self._counter.total - self._counter.spent:
                raise BudgetExhausted(
                    f"Cannot allocate {n}: only "
                    f"{self._counter.total - self._counter.spent} remaining in shared pool"
                )
        return IterationBudget(
            total=self._counter.total,
            enable_warning=self._enable_warning,
            _counter=self._counter,  # SAME counter — that's the sharing
            local_cap=n,
        )

    # -- display -------------------------------------------------------------

    def __repr__(self) -> str:
        return f"IterationBudget({self.remaining}/{self.total})"
