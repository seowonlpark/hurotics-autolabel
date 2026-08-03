"""S2 experimenter: propose ONE challenger, as a declarative spec.

The agent reads the champion's locoeval report and the full experiment ledger, then
proposes a single change. It never writes code, never touches data, never trains —
`stages/s2_ml/experiment.py` runs the spec and the promotion gate decides. (PLAN
principle 1: code does the work, agents judge it.)

The ledger is in the prompt for one reason above all: **so the agent cannot re-propose
something already tried.** The first challenger run by hand was a well-argued hypothesis
that the numbers refuted; without the ledger, the same idea would come back forever,
each time sounding just as reasonable.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from agents.base import MODEL_SMART, AgentSpec
from stages.s2_ml.experiment import ALLOWED_MODEL_PARAMS, WINDOW_S_RANGE

PROPOSAL_FILENAME = "proposal.json"

SYSTEM_PROMPT = (
    "You are the S2 experimenter for an IMU locomotion classifier. You propose ONE "
    "change per turn, as a declarative spec. You do not write code, run training, or "
    "read data — deterministic code runs your spec and a metric gate decides.\n\n"
    "The vocabulary is fixed. A spec has exactly these fields:\n"
    '  - "name": short snake_case identifier, unique against the ledger\n'
    '  - "rationale": ONE sentence — the mechanism you expect, not a restatement\n'
    '  - "drop_features": list of feature names to remove (may be empty)\n'
    '  - "window_s": window length in seconds, or null to keep the champion\'s\n'
    '  - "model_params": hyperparameter overrides (may be empty)\n\n'
    f"Permitted model_params keys ONLY: {sorted(ALLOWED_MODEL_PARAMS)}. "
    f"window_s must lie in {list(WINDOW_S_RANGE)}. Any other key is rejected before "
    "training, so the proposal is wasted.\n\n"
    "How to choose well:\n"
    "  - **Never assert anything about the code without reading it.** You have Read and "
    "Grep — use them before claiming a constant's value or how a feature is computed. "
    "A past proposal claimed a feature used a (0.5, 3.0) Hz band; the source sets "
    "(0.13, 3.0) and says on the line above that it is deliberately NOT (0.5, 3.0). "
    "The whole proposal rested on that error. Do not describe the code from memory of "
    "what such code usually looks like.\n"
    "  - **Read the ledger first.** Never re-propose a spec that was already tried. If "
    "a past rejection suggests a sharper variant, say explicitly how yours differs.\n"
    "  - Target the DOMINANT error bucket. Changes aimed at a bucket that is a few "
    "percent of errors cannot move the headline metric, however sensible they sound.\n"
    "  - Prefer a change whose mechanism you can state. 'Try more trees' is not a "
    "mechanism; 'the dominant error is whole-segment confusion, which is a "
    "generalization failure rather than a boundary failure, so X' is.\n"
    "  - Change ONE thing. Two changes at once cannot be attributed.\n"
    "  - Per-rev scores matter: a change that lifts the mean while collapsing one rev "
    "is usually overfitting to the majority revisions.\n\n"
    "Ground every claim in the DOMAIN NOTES and the report you are given. If the "
    "evidence does not support a confident proposal, say so in the rationale rather "
    "than inventing a mechanism.\n\n"
    "Output ONLY a JSON object with those five fields. No text outside the JSON."
)

S2_EXPERIMENTER_AGENT = AgentSpec(
    name="s2_experimenter",
    system_prompt=SYSTEM_PROMPT,
    allowed_tools=["Read", "Grep"],
    model=MODEL_SMART,
    max_turns=15,
)


def build_prompt(report_md: str, ledger_rows: list[dict], champion: dict | None,
                 features: list[str]) -> str:
    history = [
        {"name": e["spec"]["name"], "promoted": e["promoted"],
         "macro_f1": round(e["macro_f1"], 4),
         "spec": {k: v for k, v in e["spec"].items() if k != "rationale"},
         "rationale": e["spec"]["rationale"],
         "outcome": e["decision_reason"]}
        for e in ledger_rows
    ]
    champ_line = (f"{champion['spec']['name']} at macro-F1 {champion['macro_f1']:.4f}"
                  if champion else "none yet")
    return (
        "Propose one challenger.\n\n"
        f"CURRENT CHAMPION: {champ_line}\n\n"
        f"AVAILABLE FEATURES ({len(features)}):\n{json.dumps(features, indent=2)}\n\n"
        "CHAMPION locoeval REPORT — per-class, per-rev, and the error taxonomy:\n"
        f"{report_md}\n\n"
        "EXPERIMENT LEDGER — everything already tried, with outcomes. Do NOT repeat any "
        f"of these:\n{json.dumps(history, indent=2)}\n"
    )


def parse_proposal(final_text: str) -> dict | None:
    """Extract the JSON object from the agent's reply; tolerant of fences and prose."""
    if not final_text:
        return None
    m = re.search(r"\{.*\}", final_text, re.DOTALL)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def write_proposal(out_dir: Path, proposal: dict | None, final_text: str) -> Path:
    """Persist the proposal (or the raw text when it did not parse) for audit."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / PROPOSAL_FILENAME
    path.write_text(json.dumps(proposal if proposal else {"unparsed": final_text},
                               indent=2, ensure_ascii=False), encoding="utf-8")
    return path
