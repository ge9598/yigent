"""Structured prompt for periodic nudge.

The aux LLM is asked to look at a recent slice of the trajectory and emit a
JSON object describing one non-obvious pattern worth persisting to L1
memory, or ``null`` if nothing is worth saving.

Schema:
```json
{"topic": "kebab-case-slug", "hook": "one-line index description",
 "body": "markdown content"}
```

or literally the four characters ``null`` to skip.

The prompt deliberately resists sycophancy — it lists "bad" signals (what
NOT to save) as explicit antipatterns. This mirrors the user's personal
memory rules in CLAUDE.md.
"""

from __future__ import annotations

NUDGE_SYSTEM_PROMPT = """\
You are a memory curator for an AI agent. Your job is to look at a recent \
slice of the agent's activity and decide whether there is ONE pattern \
worth saving to long-term memory.

Rules:
- Save ONLY non-obvious, reusable observations. Examples: a user preference \
that contradicts defaults, a project convention the agent just discovered, \
a gotcha that would save time next session.
- Do NOT save: ephemeral task state, code-derivable facts, summaries of \
what just happened, generic "try X next time" advice.
- Be terse. A good memory is 1-3 sentences, not an essay.
- Prefer saving nothing (null) over saving something weak.

Output format:
- If nothing is worth saving, respond with exactly: null
- Otherwise respond with valid JSON only (no prose, no code fences):
{"topic": "<kebab-case-slug>", "hook": "<one-line hook for index>", \
"body": "<markdown body, 1-3 sentences>"}
"""


NUDGE_USER_TEMPLATE = """\
Here are the last {n_turns} turns of the session:

{trajectory_text}

Decide: is there ONE observation worth persisting? Reply with JSON or null.
"""


def format_turns(turns: list) -> str:
    """Render a list of TurnRecords as a compact text block."""
    lines: list[str] = []
    for t in turns:
        if t.user_msg is not None:
            content = t.user_msg.get("content") or ""
            lines.append(f"[turn {t.turn_index} — user] {content.strip()}")
        asst = (t.assistant_msg.get("content") or "").strip()
        if asst:
            lines.append(f"[turn {t.turn_index} — assistant] {asst}")
        for tc in t.tool_calls:
            args_preview = str(tc.arguments)[:200]
            lines.append(
                f"[turn {t.turn_index} — tool_call] {tc.name}({args_preview})"
            )
        for tr in t.tool_results:
            preview = (tr.content or "")[:200].replace("\n", " ")
            suffix = " [ERROR]" if tr.is_error else ""
            lines.append(
                f"[turn {t.turn_index} — tool_result]{suffix} {tr.name}: {preview}"
            )
    return "\n".join(lines) if lines else "(no turns yet)"
