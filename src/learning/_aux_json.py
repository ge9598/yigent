"""Tolerant JSON parser for aux-LLM responses (audit C8).

Aux LLMs (cheap models powering nudge / skill_creator / skill_improver / llm_judge)
return JSON in inconsistent shapes — fenced (```json ... ```), raw,
embedded in prose, or sometimes the literal string ``"null"``.

Each module used to maintain its own near-identical ``_parse_response`` with
a regex constant and try/except chain. This module unifies that into one
function plus a shared compiled regex.

Per-module ``_parse_response`` then becomes a thin shape validator on top of
the resulting ``dict`` / ``None``.
"""

from __future__ import annotations

import json
import re
from typing import Any

# Greedy match — captures the first balanced-looking object. Module-local
# callers can substitute a tighter regex if they need to (the function takes
# the regex as a parameter).
_DEFAULT_JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}", re.DOTALL)


def parse_aux_json(
    text: str,
    block_re: re.Pattern[str] | None = None,
) -> dict[str, Any] | None:
    """Parse aux-LLM output into a dict, or return None on failure.

    Tolerates:
        - leading/trailing whitespace
        - a single Markdown code fence around the JSON (``` ``` or ```json ... ```)
        - the literal ``null`` (treated as "nothing to return")
        - JSON wrapped in surrounding prose (extracted via ``block_re``)

    Returns ``None`` when the text is empty, ``"null"``, malformed, or
    parses to something other than a dict.
    """
    if not text:
        return None
    s = text.strip()
    # Strip a single code fence if present.
    if s.startswith("```"):
        s = s.strip("`")
        # after stripping backticks, the block may start with "json\n"
        if s.lower().startswith("json"):
            s = s[4:].lstrip()
    s = s.strip()
    if not s or s == "null":
        return None

    # Try direct parse, then fall back to regex extraction.
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        regex = block_re or _DEFAULT_JSON_BLOCK_RE
        m = regex.search(s)
        if m is None:
            return None
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None

    if not isinstance(obj, dict):
        return None
    return obj
