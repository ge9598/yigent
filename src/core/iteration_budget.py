"""Shared iteration budget across parent and child agents."""

from __future__ import annotations

import asyncio


class BudgetExhausted(Exception):
    """Raised when the iteration budget is fully consumed."""


class IterationBudget:
    """Track and allocate iteration counts for agent loops.

    The budget is shared: parent allocates a slice to each child via
    ``allocate(n)``. The child's consumption does NOT further deduct from
    the parent — the parent already paid upfront.
    """

    def __init__(self, total: int = 90, enable_warning: bool = True) -> None:
        self._total = total
        self._spent = 0
        self._lock = asyncio.Lock()
        self._enable_warning = enable_warning

    # -- properties ----------------------------------------------------------

    @property
    def total(self) -> int:
        return self._total

    @property
    def remaining(self) -> int:
        return max(0, self._total - self._spent)

    @property
    def is_warning(self) -> bool:
        """True when remaining <= 20% of total (but not exhausted) and warning is enabled."""
        if not self._enable_warning:
            return False
        if self._total == 0:
            return False
        return 0 < self.remaining <= self._total * 0.2

    @property
    def is_exhausted(self) -> bool:
        return self.remaining <= 0

    # -- mutations -----------------------------------------------------------

    async def try_consume(self, n: int = 1) -> bool:
        """Soft consume: returns True if successful, False if budget exhausted."""
        async with self._lock:
            if self._spent + n > self._total:
                return False
            self._spent += n
            return True

    async def consume(self, n: int = 1) -> int:
        """Consume *n* iterations. Returns remaining. Raises on exhaustion."""
        async with self._lock:
            if self._spent + n > self._total:
                self._spent = self._total
                raise BudgetExhausted(
                    f"Budget exhausted: {self._total} iterations used"
                )
            self._spent += n
            return self.remaining

    async def allocate(self, n: int) -> IterationBudget:
        """Carve out a child budget of *n* iterations.

        Atomically checks availability and deducts from this budget.
        The returned child budget is independent — its consumption
        does not further deduct from this parent.
        """
        async with self._lock:
            if n > self.remaining:
                raise BudgetExhausted(
                    f"Cannot allocate {n}: only {self.remaining} remaining"
                )
            self._spent += n
        return IterationBudget(total=n)

    # -- display -------------------------------------------------------------

    def __repr__(self) -> str:
        return f"IterationBudget({self.remaining}/{self._total})"
