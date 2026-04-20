"""Tests for PromptCache prefix-tracking helper."""

from __future__ import annotations

from src.context.prompt_cache import PromptCache, hash_messages


def test_hash_is_stable_for_same_content() -> None:
    msgs = [{"role": "system", "content": "hello"}]
    assert hash_messages(msgs) == hash_messages(list(msgs))


def test_hash_changes_when_content_changes() -> None:
    a = [{"role": "system", "content": "hello"}]
    b = [{"role": "system", "content": "hello!"}]
    assert hash_messages(a) != hash_messages(b)


def test_cache_compatible_when_prefix_matches() -> None:
    frozen = [{"role": "system", "content": "you are X"}]
    cache = PromptCache(frozen)
    assert cache.is_cache_compatible(frozen + [{"role": "user", "content": "hi"}])


def test_cache_incompatible_when_prefix_changes() -> None:
    frozen = [{"role": "system", "content": "you are X"}]
    cache = PromptCache(frozen)
    bad = [{"role": "system", "content": "you are Y"}]
    assert cache.is_cache_compatible(bad) is False


def test_cache_incompatible_when_messages_too_short() -> None:
    frozen = [{"role": "system", "content": "a"}, {"role": "system", "content": "b"}]
    cache = PromptCache(frozen)
    assert cache.is_cache_compatible([frozen[0]]) is False


def test_fork_inherits_prefix_hash() -> None:
    frozen = [{"role": "system", "content": "shared"}]
    parent = PromptCache(frozen)
    child = parent.on_fork()
    assert child.prefix_hash == parent.prefix_hash


def test_on_subagent_creates_independent_cache():
    from src.context.prompt_cache import PromptCache

    parent = PromptCache(frozen_system=[{"role": "system", "content": "parent"}])
    sub = parent.on_subagent([{"role": "system", "content": "sub"}])
    assert sub.prefix_hash != parent.prefix_hash


def test_on_fork_preserves_hash():
    from src.context.prompt_cache import PromptCache

    parent = PromptCache(frozen_system=[{"role": "system", "content": "parent"}])
    fork = parent.on_fork()
    assert fork.prefix_hash == parent.prefix_hash
