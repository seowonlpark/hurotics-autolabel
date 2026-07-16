"""Phase 0 smoke agent.

Its only job is to prove the wiring: DOMAIN_NOTES reached the prompt, tools fire,
the hook logs them, the cost lands on disk. It is deleted at Phase 1.
"""

from agents.base import MODEL_CHEAP, AgentSpec

SMOKE_AGENT = AgentSpec(
    name="smoke",
    system_prompt=(
        "You are a wiring test for an IMU data pipeline. Answer only from the DOMAIN NOTES "
        "provided below. Be terse. Do not speculate."
    ),
    allowed_tools=["Read", "Glob"],
    model=MODEL_CHEAP,
    max_turns=5,
)

SMOKE_PROMPT = (
    "Using only the DOMAIN NOTES in your system prompt, answer in three short lines:\n"
    "1. Which IMU channels are untrusted, and what is used instead?\n"
    "2. Why must sampling rate be measured to sub-Hz precision?\n"
    "3. List the files in the repo root using Glob.\n"
)
