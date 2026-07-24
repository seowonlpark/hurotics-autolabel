# S4 fusion agent (read-only): judge the fuser, don't run it. characterise WHY S2 and S3 disagree on
# a window (label problem, S2 error physics caught, S3 error, genuine ambiguity) so a human knows what
# the abstentions are made of (PLAN S4). code validates window-level provenance before recording.

from __future__ import annotations

import json
from pathlib import Path

from agents.base import MODEL_SMART, AgentSpec, extract_json_array, gate_and_write

REVIEW_FILENAME = "fusion_review.jsonl"
# cumulative cross-run record at the stable S4 dir; every run appends passers and reads it back
FINDINGS_LEDGER = "fusion_findings_ledger.jsonl"

CONFIDENCE_LEVELS = ("low", "medium", "high")
# what a disagreement window turns out to be; the classification the agent must choose from
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
    "(disagreement is genuine, abstaining is right), 's2_should_win', or 's3_should_win'. Code "
    "CHECKS this against the labels on the windows you cite - if you say 's2_should_win', the "
    "labels there had better show S2 scoring above the fused call, or your verdict is stamped "
    "'contradicted'. A finding that cites no scored window is dropped. Point at real disagreement "
    "windows and let the ground truth back you.\n"
    "  - Prefer few, well-evidenced findings over many thin ones. Earn 'high' confidence - a "
    "'high' the labels contradict is flagged.\n\n"
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
    # labels, taxonomy, swap verdicts, methodology, fusion
    domain_sections=("5", "7", "10", "11", "12"),
)


# prior findings earlier runs recorded, compacted for the prompt; empty string when no history
def _prior_block(prior: list[dict]) -> str:
    if not prior:
        return ""
    seen = [{"rev": p.get("rev"), "trial": p.get("trial"),
             "t_start_s": p.get("t_start_s"), "t_end_s": p.get("t_end_s"),
             "classification": p.get("classification"), "statement": p.get("statement"),
             "run": p.get("run")} for p in prior]
    return (
        "DISAGREEMENT WINDOWS ALREADY CHARACTERISED BY EARLIER RUNS - do NOT re-characterise "
        "these same windows. Spend your turns on cases not below, or overturn one only with new "
        "window-level evidence and say what changed:\n"
        f"{json.dumps(seen, indent=2)}\n\n"
    )


# assemble the prompt: fusion headline, ranked disagreement cases, files to read, prior windows
def build_prompt(metrics: dict, plot_dir: Path, prior: list[dict] | None = None) -> str:
    top = metrics["disagreement_cases"][:8]
    return (
        "Characterise the fusion's disagreement cases.\n\n"
        f"{_prior_block(prior or [])}"
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
    return extract_json_array(final_text)


# which of two accuracies-on-truth is higher; the primitive under a checked fusion_verdict
def _cmp(alt: float, fused: float) -> str:
    if alt > fused:
        return "corroborated"
    if alt < fused:
        return "contradicted"
    return "inconclusive"


# check the agent's fusion_verdict against the labels on the cited windows: 's2/s3_should_win' hold
# only if that model scored higher than the fused label; 'abstain_correct' only if no alternative call
# beat it.
def _check_verdict(verdict: str | None, s2_acc: float, s3_acc: float | None,
                   fused_acc: float) -> str:
    if verdict == "s2_should_win":
        return _cmp(s2_acc, fused_acc)
    if verdict == "s3_should_win":
        return "unverifiable" if s3_acc is None else _cmp(s3_acc, fused_acc)
    if verdict == "abstain_correct":
        better = s2_acc > fused_acc or (s3_acc is not None and s3_acc > fused_acc)
        return "contradicted" if better else "corroborated"
    return "unverifiable"


# measure a finding against the fused windows it cites: how the models actually scored there, so the
# agent's verdict rests on a measured base. n_matched=0 means it cites no scored window (unverifiable).
def measure_finding(finding: dict, rows: list[dict]) -> dict:
    try:
        t0, t1 = float(finding["t_start_s"]), float(finding["t_end_s"])
    except (KeyError, TypeError, ValueError):
        return {"n_matched": 0, "verdict_check": "unverifiable",
                "note": "finding has no usable time window"}
    sel = [r for r in rows if t0 <= r["t_start_s"] < t1]
    n = len(sel)
    if n == 0:
        return {"n_matched": 0, "verdict_check": "unverifiable",
                "note": "cites no scored window in this trial"}
    s2_acc = round(sum(r["s2_pred"] == r["true"] for r in sel) / n, 3)
    fused_acc = round(sum(r["fused_label"] == r["true"] for r in sel) / n, 3)
    s3_rows = [r for r in sel if r["s3_class"] is not None]
    s3_acc = (round(sum(r["s3_class"] == r["true"] for r in s3_rows) / len(s3_rows), 3)
              if s3_rows else None)
    return {
        "n_matched": n,
        "n_disagreement": sum(1 for r in sel if r["is_low"]),
        "s2_accuracy": s2_acc, "s3_accuracy": s3_acc, "fused_accuracy": fused_acc,
        "verdict_check": _check_verdict(finding.get("fusion_verdict"), s2_acc, s3_acc, fused_acc),
    }


# enforce the provenance gate on one finding: real window + valid classification/confidence. when
# `rows` is given, also measure against ground truth, fail a finding that cites no scored window, and
# flag a 'high' the labels contradict. ok=False means flagged, never silently kept.
def validate_finding(f: dict, rows: list[dict] | None = None) -> dict:
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

    out = {**f}
    if rows is not None:
        m = measure_finding(f, rows)
        out["measurement"] = m
        flags: list[str] = []
        if m["n_matched"] == 0:
            # a finding on no scored window is not grounded; keep it out of the record
            reasons.append("cites no scored window (finding not grounded in the fused table)")
        if m.get("verdict_check") == "contradicted" and f.get("confidence") == "high":
            flags.append("high confidence contradicted by ground truth")
        out["confidence_flags"] = flags

    out["validation"] = {"ok": not reasons, "reasons": reasons}
    return out


# validate all, write one JSONL line per finding; passers also go to the cross-run ledger when one is
# given. `windows` maps (rev, trial) -> fused row dicts, used to measure each finding. returns (path,
# n_ok, n_flagged).
def write_review(out_dir: Path, findings: list[dict] | None, final_text: str,
                 ledger_path: Path | None = None, run_id: str = "",
                 windows: dict | None = None) -> tuple[Path, int, int]:
    def validate(f: dict) -> dict:
        rows = None if windows is None else windows.get((f.get("rev"), _trial_key(f.get("trial"))))
        return validate_finding(f, [] if rows is None else rows)

    return gate_and_write(
        out_dir, findings, validate if windows is not None else validate_finding, final_text,
        out_name=REVIEW_FILENAME, ledger_path=ledger_path, run_id=run_id)


# normalise a trial id to an int key when possible, so int/str/float spellings share a bucket
def _trial_key(trial: object) -> object:
    try:
        return int(trial)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return trial
