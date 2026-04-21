"""Structured summary prompt template (Hermes pattern).

Used by ``CompressionEngine`` layers 3+4 instead of free-prose summaries.
Forced sections improve recall quality for the next turn — the model can
locate user goals, blockers, and pending work without re-reading the
condensed transcript verbatim.

Reused by Phase 3 ``skill_creator`` to extract structured workflow
descriptions from successful conversations — the same template fits both
"compress this for the model" and "describe this so the next session can
replay it" use cases.

Sections (in order):
  Goal           — the user's overarching intent
  Constraints    — explicit must-have / must-not requirements
  Progress       — Done / In progress / Blocked
  Key Decisions  — choices that shaped subsequent work
  Relevant Files — paths the model touched or should know about
  Next Steps     — what's still open
  Critical Context — anything else the next turn must remember
"""

from __future__ import annotations


SUMMARY_SYSTEM_PROMPT = """You compress agent conversations into structured \
summaries so a future turn can resume work without re-reading the original \
transcript. You MUST output the following sections in order, each with its \
header. Be specific — name files, IDs, error strings; don't paraphrase. \
Skip sections that have no content (do NOT write 'N/A').

# Goal
What the user is trying to accomplish (1-2 sentences).

# Constraints
Explicit must-have / must-not requirements stated by the user, separated by \
hyphens.

# Progress
## Done
- ... (one bullet per completed step)
## In Progress
- ... (one bullet per active step)
## Blocked
- ... (one bullet per blocker, with the blocking reason)

# Key Decisions
Choices made that shape subsequent work (architecture, library, approach). \
Each entry: decision + one-line rationale.

# Relevant Files
Paths the agent touched or that the next turn should know about. Bullet \
list; include line ranges or symbols where useful.

# Next Steps
Open work items, in priority order.

# Critical Context
Anything else the next turn would lose if dropped — error messages, \
intermediate results, environment quirks.
"""


def render_user_prompt(transcript: str) -> str:
    """Build the user-side prompt that wraps the transcript."""
    return f"Summarize this transcript using the required sections:\n\n{transcript}"
