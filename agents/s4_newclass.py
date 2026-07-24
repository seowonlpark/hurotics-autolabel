# S4 new-class discovery agent: propose classes the {stand, walk} taxonomy misses -- and NOTHING
# more. the deterministic core (stages/s4_fusion/newclass.py) already isolated the candidate
# spans: STAND-labeled windows the physics reads as MOVING but not clean gait. this agent does
# the one thing code cannot -- look at a span's physics profile + the S3 figure and say "these
# look like the same THING, and here is what it is" (a ramp, a sit, a repositioning). it is a
# PROPOSER under a hard governance contract (DOMAIN_NOTES Section 11.2): it never adds a class, never
# edits a label, never touches the S2 model / S3 rule / fuser. code validates cluster mass and
# provenance, then routes every proposal to needs_human. read-only (Read/Grep).

from __future__ import annotations

import json
from pathlib import Path

from agents.base import MODEL_SMART, AgentSpec, extract_json_array, gate_and_write

PROPOSALS_FILENAME = "new_class_proposals.jsonl"
# cumulative cross-run record at the stable S4 dir; every run appends its supported proposals
# and reads them back so a later run does not re-propose a class an earlier run already surfaced
NEWCLASS_LEDGER = "new_class_ledger.jsonl"

CONFIDENCE_LEVELS = ("low", "medium", "high")

SYSTEM_PROMPT = (
    "You are the S4 new-class discovery analyst for an IMU locomotion pipeline. The pipeline "
    "classifies each window as STAND or WALK. That 2-class taxonomy is deliberately narrow, and "
    "the fusion has surfaced a residue it has no word for: windows a human LABELED 'standing' "
    "that the physics reads as MOVING (grav_stab < 0.5) yet NOT clean alternating gait. Your job "
    "is to look at these candidate spans and propose whether some of them form a coherent NEW "
    "class - a ramp, a repositioning shuffle, a sit-to-stand, a bilateral squat (both knees bend "
    "together - legs IN PHASE, unlike walking's antiphase), etc.\n\n"
    "GOVERNANCE - this is not negotiable (DOMAIN NOTES Section 11.2, non-negotiable #1):\n"
    "  - You PROPOSE. You never adopt a class, never relabel a window, never change the model, "
    "the swap rule, or the fuser. A human decides whether a class is real. Code routes every "
    "proposal you make to needs_human regardless of how confident you are.\n"
    "  - A class is structure that RECURS. A single odd span is a mislabel or noise, not a "
    "category. Only group spans into a class when they share a physics signature AND appear "
    "across more than one rev (subject/day). Code will reject a proposal that does not clear "
    ">= 4 spans across >= 2 revs as 'insufficient_evidence' - so do not pad, and do not invent "
    "a class from one trial.\n\n"
    "Your evidence (Read/Grep only - you cannot see raw signal):\n"
    "  - runs/s4_fusion/new_class_candidates.json - every candidate span with its physics "
    "profile (grav_stab, antiphase, periodicity, interleg_offset, swap_count means) and the "
    "figure path. This is your primary evidence; read it first.\n"
    "  - runs/s3_physics/plots/trial_<rev>_t<trial>.png - the physics figure (leg angles + "
    "label band + swap verdict). Read the figures for the spans you want to group, to SEE the "
    "shape (in-phase vs antiphase, ramp vs oscillation).\n\n"
    "How to read the profile (DOMAIN NOTES Section 10, Section 11):\n"
    "  - antiphase > 0 means the legs move in OPPOSITION (walking-like); antiphase <= 0 means "
    "they move TOGETHER (a squat/sit/ramp - the taxonomy has no word for this, and it is the "
    "most interesting signal here).\n"
    "  - grav_stab -> 0 means the tilt is sweeping (moving), -> 1 means held still. periodicity "
    "-> 1 means rhythmic. A class with low antiphase, low grav_stab, low periodicity is moving "
    "but not gait and not rhythm - a transition or posture change.\n"
    "  - A metric that moves does not confirm the story (Section 11.3). For every class, say in "
    "`caveat` what would FALSIFY it - what else these spans could be (e.g. 'could just be "
    "mislabeled brief walks' or 'could be sensor drift, not a posture').\n\n"
    "HARD RULE (code rejects otherwise): every class you propose must cite span_refs, each "
    "{rev, trial, t_start_s, t_end_s} overlapping a real candidate span from the JSON.\n\n"
    "Output ONLY a JSON array of proposals, each:\n"
    '  {"class_name": short snake_case name, "description": one sentence on the posture/motion, '
    '"span_refs": [{"rev","trial","t_start_s","t_end_s"}, ...], '
    '"physics_signature": one sentence on the shared profile that defines it, '
    '"distinguishes_from_stand_and_walk": one sentence, '
    f'"statement": your claim, "confidence": one of {list(CONFIDENCE_LEVELS)}, '
    '"caveat": what would falsify this class}\n'
    "Prefer one or two well-evidenced proposals over many thin ones. If nothing recurs "
    "coherently, output an empty array []. No prose outside the JSON array."
)

S4_NEWCLASS_AGENT = AgentSpec(
    name="s4_newclass",
    system_prompt=SYSTEM_PROMPT,
    allowed_tools=["Read", "Grep"], # Read renders the S3 PNGs + json; Grep slices the candidates
    model=MODEL_SMART,
    max_turns=30,
    # labels + swap rule + methodology (incl. 11.2 invented-category) + fusion/new-class
    domain_sections=("5", "10", "11", "12"),
)


# classes earlier runs already proposed (and code found supported), compacted for the prompt;
# empty string when there is no history, so the first run's prompt is unchanged
def _prior_block(prior: list[dict]) -> str:
    if not prior:
        return ""
    seen = [{"class_name": p.get("class_name"), "description": p.get("description"),
             "run": p.get("run")} for p in prior]
    return (
        "CLASSES ALREADY PROPOSED AND SUPPORTED BY EARLIER RUNS - do NOT re-propose these. "
        "Propose only a class not already below, unless you are refining one with materially "
        "new spans, in which case say which prior class you are refining and how:\n"
        f"{json.dumps(seen, indent=2)}\n\n"
    )


# assemble the prompt: the corpus signature, the top candidate spans, the files to read, and
# the classes earlier runs already surfaced
def build_prompt(bundle: dict, plot_dir: Path, prior: list[dict] | None = None) -> str:
    sig = bundle["corpus_signature"]
    top = bundle["spans"][:16]
    return (
        "Propose the classes (if any) these candidate spans form.\n\n"
        f"{_prior_block(prior or [])}"
        f"CORPUS SIGNATURE: {sig['n_spans']} spans / {sig['n_windows']} windows across "
        f"{sig['n_revs']} revs {sig['revs']}. Mean profile {sig['mean_profile']}. A class needs "
        f">= {sig['min_spans_for_support']} spans across >= {sig['min_revs_for_support']} revs "
        "to be 'supported' (code checks this; you still propose, a human still decides).\n\n"
        "CANDIDATE SPANS (busiest first; full set + profiles in "
        "runs/s4_fusion/new_class_candidates.json):\n"
        f"{json.dumps(top, indent=2)}\n\n"
        f"Physics figures are in {plot_dir.resolve()} as trial_<rev>_t<trial>.png. Read the JSON "
        "in full, then Read the figures for spans you want to group, and write your proposals as "
        "a JSON array with span_refs. Empty array [] if nothing recurs coherently."
    )


# pull the JSON array out of the reply; tolerant of fences and prose
def parse_proposals(final_text: str) -> list[dict] | None:
    return extract_json_array(final_text)


# validate every proposal (cluster mass + provenance) and write one JSONL line each, plus the
# raw text alongside. every line is needs_human. returns (path, n_supported, n_insufficient).
# a proposal "passes" on cluster mass (validation.support == "supported"), not the plain
# validation.ok the other stages use -- hence the is_ok override.
def write_proposals(out_dir: Path, proposals: list[dict] | None, spans: list[dict],
                    final_text: str, ledger_path: Path | None = None,
                    run_id: str = "") -> tuple[Path, int, int]:
    from stages.s4_fusion.newclass import validate_proposal

    return gate_and_write(
        out_dir, proposals, lambda p: validate_proposal(p, spans), final_text,
        out_name=PROPOSALS_FILENAME,
        is_ok=lambda v: v.get("validation", {}).get("support") == "supported",
        ledger_path=ledger_path, run_id=run_id)
