# S4 fusion agent: judge the fuser, don't run it (read-only)
# the deterministic fuser already made every call and flagged the disagreements (S2 vs S3).
# this agent does the one thing code cannot: look at WHY the two models disagree on a given
# window and say what it means -- is it a label problem, an S2 error the physics caught, an S3
# error, or genuine ambiguity? it never re-labels and never changes the fusion; it characterises
# the LOW-confidence cases so a human knows what the abstentions are made of (PLAN S4). code
# validates window-level provenance before recording, exactly as S3 does.

from __future__ import annotations

import json
import re
from pathlib import Path

from agents.base import MODEL_SMART, AgentSpec

REVIEW_FILENAME = "fusion_review.jsonl"

CONFIDENCE_LEVELS = ("low", "medium", "high")
# what a disagreement window turns out to be -- the classification the agent must choose from
CLASSES = ("label_problem", "s2_error_physics_caught", "s3_error", "genuine_ambiguity")

SYSTEM_PROMPT = (
    "You are the S4 fusion analyst for an IMU locomotion pipeline. Two models predict each "
    "window: S2 (a learned random forest) and S3 (a model-free physics 'swap rule'). A "
    "deterministic fuser already combined them and flagged every window where they DISAGREE as "
    "LOW-confidence (it abstains there). Your job is to explain what those disagreements ARE.\n\n"
    "Measured facts you must not re-derive (DOMAIN NOTES Section 12):\n"
    "  - When S2 and S3 AGREE, they are 98% correct. Disagreement is where the risk lives.\n"
    "  - S3's WALKING verdict is reliable (~78-99% truly walking); its STANDING verdict is NOT "
    "(it under-calls slow gait as standing). So a disagreement is usually one of: S3 wrongly "
    "calling slow walking 'standing', OR S2 wrongly calling a still posture 'walking', OR a "
    "label that is itself wrong (a labeled STANDING bout that is actually a moving ramp).\n\n"
    "Your evidence (Read/Grep only - you cannot see raw signal):\n"
    "  - runs/s4_fusion/disagreements.json - trials ranked by how many LOW windows they hold\n"
    "  - runs/s4_fusion/fused_windows.csv - per window: s2_pred, s3, fused_label, true, "
    "confidence. Grep a trial's rows to see each disagreement (s2_pred vs s3 vs true).\n"
    "  - runs/s3_physics/plots/trial_<rev>_t<trial>.png - the physics figure for that trial "
    "(leg angles + label band + swap verdict). Read it to see what the body is doing.\n\n"
    "For each disagreement case you examine, classify it as one of "
    f"{list(CLASSES)} and give window-level provenance. HARD RULE (code rejects otherwise): "
    "every finding needs {rev, trial, t_start_s, t_end_s} with t_end_s > t_start_s.\n\n"
    "How to think (DOMAIN NOTES Section 11.3, Section 11.4):\n"
    "  - A metric that moves does not confirm the story you tell about it. Say what would "
    "FALSIFY each classification in `caveat`.\n"
    "  - If a labeled region's physics contradicts its label, set `label_audit` true - that "
    "routes it to the human label-review path (precedent: labeled STANDING that was a ramp).\n"
    "  - Judge whether the fuser did the right thing: `fusion_verdict` is 'abstain_correct' "
    "(disagreement is genuine, abstaining is right), 's2_should_win', or 's3_should_win'.\n"
    "  - Prefer few, well-evidenced findings over many thin ones. Earn 'high' confidence.\n\n"
    "Output ONLY a JSON array of objects, each with:\n"
    '  {"rev","trial","t_start_s","t_end_s","classification": one of '
    f'{list(CLASSES)}, '
    '"statement": one sentence, "fusion_verdict": one of '
    '["abstain_correct","s2_should_win","s3_should_win"], '
    f'"label_audit": true|false, "confidence": one of {list(CONFIDENCE_LEVELS)}, '
    '"caveat": what would falsify it}\n'
    "No prose outside the JSON array."
)

S4_FUSION_AGENT = AgentSpec(
    name="s4_fusion",
    system_prompt=SYSTEM_PROMPT,
    allowed_tools=["Read", "Grep"], # Read renders the S3 PNGs + json; Grep slices the fused csv
    model=MODEL_SMART,
    max_turns=30,
)


# assemble the prompt: the fusion headline, the ranked disagreement cases, the files to read
def build_prompt(metrics: dict, plot_dir: Path) -> str:
    top = metrics["disagreement_cases"][:8]
    return (
        "Characterise the fusion's disagreement cases.\n\n"
        f"FUSION SUMMARY: fused macro-F1 {metrics['fused']['macro_f1']} vs S2 alone "
        f"{metrics['s2_alone']['macro_f1']}; acting on HIGH+MED confidence covers "
        f"{metrics['acting_on_confidence']['coverage']} of windows at accuracy "
        f"{metrics['acting_on_confidence']['accuracy']}. Confidence tiers: "
        f"{json.dumps({k: v['accuracy'] for k, v in metrics['calibration'].items()})}\n\n"
        "DISAGREEMENT TRIALS (most LOW-confidence windows first; `low_accuracy` = how often the "
        "fused label was right on those windows):\n"
        f"{json.dumps(top, indent=2)}\n\n"
        f"Physics figures are in {plot_dir.resolve()} as trial_<rev>_t<trial>.png. Grep "
        "runs/s4_fusion/fused_windows.csv for a trial's rows to see each disagreement, and Read "
        "the matching figure. Then write your findings as a JSON array with window provenance."
    )


# pull the JSON array out of the reply; tolerant of fences and prose
def parse_review(final_text: str) -> list[dict] | None:
    if not final_text:
        return None
    m = re.search(r"\[.*\]", final_text, re.DOTALL)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, list) else None


# enforce the provenance gate on one finding: real window + valid classification/confidence.
# returns the finding with a validation block; ok=False means flagged, never silently kept.
def validate_finding(f: dict) -> dict:
    reasons: list[str] = []
    if not isinstance(f, dict) or not f.get("statement"):
        reasons.append("no statement")
    if f.get("classification") not in CLASSES:
        reasons.append(f"classification not in {list(CLASSES)}")
    if f.get("confidence") not in CONFIDENCE_LEVELS:
        reasons.append(f"confidence not in {list(CONFIDENCE_LEVELS)}")
    if not f.get("rev") or f.get("trial") is None:
        reasons.append("no rev/trial")
    try:
        if not float(f["t_end_s"]) > float(f["t_start_s"]):
            reasons.append("t_end_s not after t_start_s")
    except (KeyError, TypeError, ValueError):
        reasons.append("missing/invalid time window")
    return {**f, "validation": {"ok": not reasons, "reasons": reasons}}


# validate all, write one JSONL line per finding + the raw text alongside. returns (path, n_ok,
# n_flagged).
def write_review(out_dir: Path, findings: list[dict] | None,
                 final_text: str) -> tuple[Path, int, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / REVIEW_FILENAME
    (out_dir / "fusion_review_raw.txt").write_text(final_text or "", encoding="utf-8")

    if not findings:
        path.write_text("", encoding="utf-8")
        return path, 0, 0

    validated = [validate_finding(f) for f in findings]
    n_ok = sum(1 for v in validated if v["validation"]["ok"])
    with path.open("w", encoding="utf-8") as fh:
        for v in validated:
            fh.write(json.dumps(v, ensure_ascii=False) + "\n")
    return path, n_ok, len(validated) - n_ok
