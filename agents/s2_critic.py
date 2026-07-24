# S2 critic: review a proposed challenger before it runs (read-only)
# sees the ledger so it can catch repeats of rejected ideas; does not decide promotion
# -- experiment.decide() does that on the metric afterwards. see README.

from __future__ import annotations

import json
from pathlib import Path

from agents.base import MODEL_SMART, AgentSpec, extract_json_object

REVIEW_FILENAME = "critic_review.json"

SYSTEM_PROMPT = (
    "You are the S2 critic for an IMU locomotion classifier. You review ONE proposed "
    "experiment before it is run. You never run it, never modify it, and never decide "
    "promotion - a metric gate does that afterwards. You decide only whether running it "
    "is worth the cycle.\n\n"
    "Return exactly one verdict:\n"
    "  - approve: the proposal is new, its mechanism is plausible, and it targets "
    "something that could actually move the metric.\n"
    "  - reject: it repeats an entry in the ledger, contradicts an established DOMAIN "
    "NOTES finding, changes more than one thing at once, or targets an error bucket too "
    "small to matter. Say which, concretely.\n"
    "  - revise: the idea is worth running but the spec is wrong - name the specific "
    "field to change and why.\n\n"
    "Checks to apply, in order:\n"
    "  0. **VERIFY THE PREMISE - use your tools.** If the rationale asserts anything "
    "about the code (a constant's value, how a feature is computed, what a stage does), "
    "Read or Grep the source and confirm it before anything else. A proposal built on a "
    "false premise is `reject`, however sound the reasoning downstream. This has already "
    "happened: a proposal claimed a feature used a (0.5, 3.0) Hz band when "
    "`stages/s2_ml/features.py` sets (0.13, 3.0) and says on the line above that it is "
    "deliberately NOT (0.5, 3.0). It was approved because nobody checked. The ledger "
    "stores rationales, so an unverified premise that happens to win becomes recorded "
    "fact - that is the failure this check exists to prevent. If a claim cannot be "
    "verified from the repo, say `revise` and name the claim.\n"
    "  1. **Novelty.** Compare against every ledger entry. A spec that differs only "
    "cosmetically from a rejected one IS a repeat - cite the entry by name.\n"
    "  2. **Attribution.** One change at a time. Two at once cannot be attributed to "
    "either.\n"
    "  3. **Leverage.** Does it target the dominant error bucket? A change aimed at a "
    "bucket worth a few percent of errors cannot move the headline metric.\n"
    "  4. **Evidence.** Does the rationale state a mechanism, and does the report "
    "support it? Reject confident mechanisms the numbers do not show.\n"
    "  5. **Consistency.** Does it contradict a DOMAIN NOTES finding? Note that "
    "signal-based sagittal-axis detection and naive decimation are already measured "
    "dead ends.\n\n"
    "Be terse and specific. Vague approval is worse than no review - it launders a bad "
    "proposal as a checked one.\n\n"
    'Output ONLY a JSON object: {"verdict": "approve"|"reject"|"revise", '
    '"reasons": [<short strings>], "repeat_of": <ledger entry name or null>, '
    '"confidence": <0.0-1.0>}. No text outside the JSON.'
)

S2_CRITIC_AGENT = AgentSpec(
    name="s2_critic",
    system_prompt=SYSTEM_PROMPT,
    allowed_tools=["Read", "Grep"],
    model=MODEL_SMART,
    max_turns=10,
    # same footprint as the experimenter it reviews (also used by s2_critic_probe)
    domain_sections=("4", "5", "6", "7", "9", "10", "11"),
)


# assemble the review prompt: proposal, current champion, and the full ledger
def build_prompt(proposal: dict, ledger_rows: list[dict], report_md: str,
                 champion: dict | None) -> str:
    history = [
        {"name": e["spec"]["name"], "promoted": e["promoted"],
         "macro_f1": round(e["macro_f1"], 4), "spec": e["spec"],
         "outcome": e["decision_reason"]}
        for e in ledger_rows
    ]
    champ_line = (f"{champion['spec']['name']} at macro-F1 {champion['macro_f1']:.4f}"
                  if champion else "none yet")
    return (
        "Review this proposed experiment.\n\n"
        f"PROPOSAL:\n{json.dumps(proposal, indent=2)}\n\n"
        f"CURRENT CHAMPION: {champ_line}\n\n"
        "CHAMPION locoeval REPORT (per-class, per-rev, error taxonomy):\n"
        f"{report_md}\n\n"
        "EXPERIMENT LEDGER - check the proposal against every entry for repeats:\n"
        f"{json.dumps(history, indent=2)}\n"
    )


# pull the verdict JSON out of the agent's final text; None if absent or malformed
def parse_review(final_text: str) -> dict | None:
    return extract_json_object(final_text)


# persist the review; an unparseable review becomes a revise, never an approve. `name` lets a
# second (post-revision) critique write to a distinct file so the first is not overwritten.
def write_review(out_dir: Path, review: dict | None, final_text: str,
                 name: str = REVIEW_FILENAME) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = review if review else {
        "verdict": "revise",
        "reasons": ["critic output did not parse; refusing to approve by default"],
        "repeat_of": None, "confidence": 0.0, "unparsed": final_text,
    }
    path = out_dir / name
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
