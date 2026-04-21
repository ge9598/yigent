"""Tests for IterationBudget — truly shared (Unit 10) semantics."""

from __future__ import annotations

import pytest

from src.core.iteration_budget import BudgetExhausted, IterationBudget


@pytest.mark.asyncio
async def test_allocate_does_not_pre_deduct_shared_pool():
    """Unit 10 — allocate() returns a shared handle, not a pre-charged slice.
    Parent's remaining is unchanged immediately after allocate()."""
    parent = IterationBudget(total=100, enable_warning=False)
    child = await parent.allocate(20)
    # The child has access to up to 20 of the shared pool, but nothing has
    # been spent yet — both still see 100 remaining (child clamped to 20 by cap).
    assert parent.remaining == 100
    assert child.total == 20  # local cap
    assert child.remaining == 20  # min(shared_remaining=100, cap_remaining=20)


@pytest.mark.asyncio
async def test_child_consumption_DOES_decrement_parent():
    """Unit 10 — sharing means parent SEES child's consumption.
    Old behaviour (independent counters) is gone."""
    parent = IterationBudget(total=100, enable_warning=False)
    child = await parent.allocate(20)
    await child.consume(5)
    assert child.remaining == 15  # cap remaining = 20 - 5
    assert parent.remaining == 95  # shared pool also decremented


@pytest.mark.asyncio
async def test_parent_consumption_visible_to_child():
    """Sanity: shared sharing goes both ways — parent burning the pool
    leaves less for the child."""
    parent = IterationBudget(total=100, enable_warning=False)
    child = await parent.allocate(50)
    await parent.consume(70)
    # Shared pool has 30 left. Child's local cap says 50, but shared takes
    # priority (min of the two).
    assert parent.remaining == 30
    assert child.remaining == 30


@pytest.mark.asyncio
async def test_local_cap_limits_child_consumption():
    """The local_cap on a child handle prevents it from spending more than
    its allocation, even if the shared pool has more."""
    parent = IterationBudget(total=100, enable_warning=False)
    child = await parent.allocate(10)
    await child.consume(10)
    assert child.is_exhausted
    with pytest.raises(BudgetExhausted, match="Local cap"):
        await child.consume(1)
    # Parent paid for the 10 the child spent; still has 90 left to spend.
    assert parent.remaining == 90


@pytest.mark.asyncio
async def test_shared_pool_exhaustion_propagates_to_all_handles():
    """Once the shared total is hit, EVERY handle reports exhausted."""
    parent = IterationBudget(total=10, enable_warning=False)
    child = await parent.allocate(8)  # cap within pool size
    await parent.consume(10)
    assert parent.is_exhausted
    assert child.is_exhausted
    with pytest.raises(BudgetExhausted):
        await child.consume(1)


@pytest.mark.asyncio
async def test_allocate_cannot_exceed_shared_remaining():
    parent = IterationBudget(total=10, enable_warning=False)
    with pytest.raises(BudgetExhausted, match="Cannot allocate"):
        await parent.allocate(20)


@pytest.mark.asyncio
async def test_multiple_allocations_share_one_pool():
    """Two child handles spending the SAME shared pool — total spending
    can't exceed the parent's total."""
    parent = IterationBudget(total=100, enable_warning=False)
    a = await parent.allocate(30)
    b = await parent.allocate(40)
    # Both are open caps over the same pool. Pool still has 100.
    assert parent.remaining == 100
    await a.consume(10)
    await b.consume(20)
    assert parent.remaining == 70
    assert a.remaining == 20  # local cap 30, used 10
    assert b.remaining == 20  # local cap 40, used 20


@pytest.mark.asyncio
async def test_allocate_zero_is_immediately_exhausted_handle():
    parent = IterationBudget(total=10, enable_warning=False)
    child = await parent.allocate(0)
    assert parent.remaining == 10  # pool unchanged
    assert child.total == 0
    assert child.is_exhausted


@pytest.mark.asyncio
async def test_total_property_reports_local_cap_when_set():
    """For a child handle with a local cap, `.total` is the cap (what the
    handle can spend), not the shared pool size."""
    parent = IterationBudget(total=100, enable_warning=False)
    child = await parent.allocate(15)
    assert child.total == 15
    assert child.shared_total == 100
    assert parent.total == 100
    assert parent.shared_total == 100
