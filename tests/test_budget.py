import pytest
from src.core.iteration_budget import IterationBudget, BudgetExhausted


@pytest.mark.asyncio
async def test_try_consume_returns_true_when_ok():
    b = IterationBudget(10)
    assert await b.try_consume(3) is True
    assert b.remaining == 7


@pytest.mark.asyncio
async def test_try_consume_returns_false_when_exhausted():
    b = IterationBudget(5)
    await b.consume(5)
    assert await b.try_consume(1) is False


@pytest.mark.asyncio
async def test_warning_disabled():
    b = IterationBudget(10, enable_warning=False)
    await b.consume(9)
    assert not b.is_warning


@pytest.mark.asyncio
async def test_warning_enabled_default():
    b = IterationBudget(10)
    await b.consume(9)
    assert b.is_warning  # 1 <= 10*0.2=2
