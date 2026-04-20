"""Tests for CircuitBreaker."""

from __future__ import annotations

from src.context.circuit_breaker import CircuitBreaker


def test_starts_closed() -> None:
    cb = CircuitBreaker(threshold=3)
    assert cb.is_open is False
    assert cb.failures == 0


def test_opens_at_threshold() -> None:
    cb = CircuitBreaker(threshold=3)
    cb.record_failure()
    cb.record_failure()
    assert cb.is_open is False
    cb.record_failure()
    assert cb.is_open is True


def test_success_resets_counter() -> None:
    cb = CircuitBreaker(threshold=3)
    cb.record_failure()
    cb.record_failure()
    cb.record_success()
    assert cb.failures == 0
    assert cb.is_open is False


def test_explicit_reset() -> None:
    cb = CircuitBreaker(threshold=2)
    cb.record_failure()
    cb.record_failure()
    assert cb.is_open is True
    cb.reset()
    assert cb.is_open is False


def test_threshold_one() -> None:
    cb = CircuitBreaker(threshold=1)
    cb.record_failure()
    assert cb.is_open is True
