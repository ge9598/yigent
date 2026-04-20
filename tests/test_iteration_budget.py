"""Tests for IterationBudget allocate() behavior."""

from __future__ import annotations

import pytest

from src.core.iteration_budget import BudgetExhausted, IterationBudget


@pytest.mark.asyncio
async def test_allocate_decrements_parent_upfront():
    parent = IterationBudget(total=100, enable_warning=False)
    child = await parent.allocate(20)
    # Parent paid upfront — immediately down 20
    assert parent.remaining == 80
    # Child has its own 20
    assert child.total == 20
    assert child.remaining == 20


@pytest.mark.asyncio
async def test_child_consumption_does_not_further_deduct_parent():
    parent = IterationBudget(total=100, enable_warning=False)
    child = await parent.allocate(20)
    await child.consume(5)
    assert child.remaining == 15
    assert parent.remaining == 80  # unchanged by child's consume


@pytest.mark.asyncio
async def test_child_exhaustion_is_independent():
    parent = IterationBudget(total=100, enable_warning=False)
    child = await parent.allocate(10)
    await child.consume(10)
    assert child.is_exhausted
    with pytest.raises(BudgetExhausted):
        await child.consume(1)
    # Parent still has 90
    assert parent.remaining == 90


@pytest.mark.asyncio
async def test_allocate_cannot_exceed_parent_remaining():
    parent = IterationBudget(total=10, enable_warning=False)
    with pytest.raises(BudgetExhausted, match="Cannot allocate"):
        await parent.allocate(20)


@pytest.mark.asyncio
async def test_multiple_allocations_chain():
    parent = IterationBudget(total=100, enable_warning=False)
    a = await parent.allocate(30)
    b = await parent.allocate(40)
    assert parent.remaining == 30
    assert a.remaining == 30
    assert b.remaining == 40


@pytest.mark.asyncio
async def test_allocate_zero_is_noop():
    parent = IterationBudget(total=10, enable_warning=False)
    child = await parent.allocate(0)
    assert parent.remaining == 10
    assert child.total == 0
    assert child.is_exhausted  # 0-total budget is immediately exhausted
