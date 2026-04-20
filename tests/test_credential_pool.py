"""Tests for CredentialPool — multi-key rotation with 429 cooldown."""

from __future__ import annotations

import time

import pytest

from src.providers.credential_pool import CredentialPool


def test_round_robin_rotation():
    pool = CredentialPool(keys=["k1", "k2", "k3"], strategy="round_robin")
    assert pool.acquire() == "k1"
    assert pool.acquire() == "k2"
    assert pool.acquire() == "k3"
    assert pool.acquire() == "k1"


def test_fill_first_sticks_until_error():
    pool = CredentialPool(keys=["k1", "k2", "k3"], strategy="fill_first")
    assert pool.acquire() == "k1"
    assert pool.acquire() == "k1"
    pool.mark_error("k1", status=500)
    assert pool.acquire() == "k2"
    assert pool.acquire() == "k2"


def test_least_used_picks_minimum():
    pool = CredentialPool(keys=["k1", "k2", "k3"], strategy="least_used")
    assert pool.acquire() == "k1"
    assert pool.acquire() == "k2"
    assert pool.acquire() == "k3"
    # All have usage=1; tie → k1 (insertion order)
    assert pool.acquire() == "k1"


def test_random_stays_within_pool():
    pool = CredentialPool(keys=["k1", "k2"], strategy="random", seed=42)
    for _ in range(20):
        assert pool.acquire() in {"k1", "k2"}


def test_429_cooldown_excludes_key_temporarily():
    pool = CredentialPool(keys=["k1", "k2"], strategy="round_robin", cooldown_seconds=0.1)
    assert pool.acquire() == "k1"
    pool.mark_error("k1", status=429)
    assert pool.acquire() == "k2"
    assert pool.acquire() == "k2"
    time.sleep(0.15)
    next_keys = {pool.acquire() for _ in range(4)}
    assert "k1" in next_keys


def test_non_429_rotates_but_does_not_cool():
    pool = CredentialPool(keys=["k1", "k2"], strategy="round_robin", cooldown_seconds=10)
    assert pool.acquire() == "k1"
    pool.mark_error("k1", status=500)
    assert pool.acquire() == "k2"
    assert pool.acquire() == "k1"


def test_all_keys_cooling_raises():
    pool = CredentialPool(keys=["k1", "k2"], strategy="round_robin", cooldown_seconds=10)
    pool.mark_error("k1", status=429)
    pool.mark_error("k2", status=429)
    with pytest.raises(RuntimeError, match="all keys in cooldown"):
        pool.acquire()


def test_single_key_works():
    pool = CredentialPool(keys=["k1"], strategy="fill_first")
    assert pool.acquire() == "k1"
    assert pool.acquire() == "k1"


def test_empty_keys_raises_at_init():
    with pytest.raises(ValueError, match="at least one key"):
        CredentialPool(keys=[], strategy="round_robin")


def test_unknown_strategy_raises():
    with pytest.raises(ValueError, match="Unknown strategy"):
        CredentialPool(keys=["k1"], strategy="bogus")


def test_add_remove_key():
    pool = CredentialPool(keys=["k1"], strategy="round_robin")
    pool.add_key("k2")
    acquired = {pool.acquire() for _ in range(4)}
    assert acquired == {"k1", "k2"}
    pool.remove_key("k1")
    assert pool.acquire() == "k2"


def test_remove_last_key_raises():
    pool = CredentialPool(keys=["k1"], strategy="round_robin")
    with pytest.raises(ValueError, match="cannot remove last key"):
        pool.remove_key("k1")


def test_set_strategy_at_runtime():
    pool = CredentialPool(keys=["k1", "k2", "k3"], strategy="round_robin")
    pool.acquire()
    pool.set_strategy("fill_first")
    first = pool.acquire()
    assert pool.acquire() == first


def test_round_robin_cursor_does_not_regress_on_late_error():
    """Out-of-order mark_error must not rewind the cursor."""
    pool = CredentialPool(keys=["k1", "k2", "k3"], strategy="round_robin")
    assert pool.acquire() == "k1"
    assert pool.acquire() == "k2"
    # Late arriving error from the k1 request
    pool.mark_error("k1", status=500)
    # Cursor was at position 2 — next acquire must advance to k3, not rewind to k2
    assert pool.acquire() == "k3"
    assert pool.acquire() == "k1"


def test_mark_error_on_unknown_key_is_noop():
    pool = CredentialPool(keys=["k1"], strategy="round_robin")
    # Should not raise, should not affect state
    pool.mark_error("never_added", status=429)
    assert pool.acquire() == "k1"
