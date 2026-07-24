# S3 deterministic core: compute anchor table, run rate-invariance audit, draw plots. no agent, no
# judgement. outputs (anchors.csv, rate_audit.json, plots/) are the interface the hypothesis agent
# reads; rate_audit.json satisfies the PLAN S3 gate (Section 10).

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from stages.s2_ml.dataset import STAND, WALK, Trial, load_dataset
from stages.s2_ml.experiment import resolve_champion
from stages.s2_ml.features import WindowSpec
from stages.s3_physics.anchors import ANCHOR_NAMES, WALKING, trial_anchors
from stages.s3_physics.plots import plot_all
from stages.s3_physics.rate_audit import (
    AUDIT_FACTOR, AUDIT_TOL, CANONICAL_HZ, audit_anchors,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
S3_OUT_DIR = REPO_ROOT / "runs" / "s3_physics"
S2_RUN_DIR = REPO_ROOT / "runs" / "s2_ml"


# champion's window, so S3's per-window grid matches the S2 OOF grid the fusion joins on (a
# mismatched window collapses the inner join). defaults to WindowSpec() when no champion exists yet.
def champion_windowspec(s2_dir: Path = S2_RUN_DIR) -> WindowSpec:
    return resolve_champion(s2_dir, require=False)[0]

ANCHORS_CSV = "anchors.csv"
AUDIT_JSON = "rate_audit.json"
DISAGREEMENT_JSON = "disagreement.json"
PLOTS_SUBDIR = "plots"


# the per-window anchor table for the whole corpus, one row per window
def build_anchor_table(trials: list[Trial], spec: WindowSpec | None = None) -> pd.DataFrame:
    spec = spec or WindowSpec()
    frames = [a for t in trials if not (a := trial_anchors(t, spec)).empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# physics cause of one window's disagreement with its human label (Section 10.7, Section 11), so the
# reviewer can triage the flag rather than see a bare boolean. None when verdict and label agree.
def disagree_reason(label: object, verdict: str, antiphase: float, swap_count: int) -> str | None:
    if label == WALK and verdict != WALKING:
        if antiphase <= 0:
            return "in_phase_not_gait" # legs together: physics right, likely a label issue
        if swap_count <= 1:
            return "single_hump"       # alternates but <2 crossings even after the grow
        return "sub_threshold"         # antiphase swing that never commits both +/-delta bands
    if label == STAND and verdict == WALKING:
        return "stand_reads_walking"   # label-audit: standing stretch the physics reads as gait
    return None


# per-trial physics-vs-label disagreement, ranked so the agent looks at the worst trials first.
# ranked on the stride-adaptive verdict (Section 10.6) so slow-gait window artifacts don't drown real
# label issues. each row carries a `reasons` histogram (Section 10.7) showing why a trial ranks.
def label_disagreement(table: pd.DataFrame) -> list[dict]:
    rows = []
    for (rev, trial), g in table.groupby(["rev", "trial"], sort=True):
        walk_lab = g[g["label"] == WALK]
        stand_lab = g[g["label"] == STAND]
        walk_not_walking = float((walk_lab["swap_verdict_adaptive"] != WALKING).mean()) if len(walk_lab) else 0.0
        stand_is_walking = float((stand_lab["swap_verdict_adaptive"] == WALKING).mean()) if len(stand_lab) else 0.0
        reasons = g["disagree_reason"].dropna().value_counts().to_dict() if "disagree_reason" in g else {}
        rows.append({
            "rev": rev, "trial": int(trial), "n_windows": int(len(g)),
            "walk_labeled_not_walking": round(walk_not_walking, 3),
            "stand_labeled_is_walking": round(stand_is_walking, 3),
            "disagreement": round(max(walk_not_walking, stand_is_walking), 3),
            "reasons": {k: int(v) for k, v in reasons.items()},
            "plot": f"{PLOTS_SUBDIR}/trial_{rev}_t{int(trial)}.png",
        })
    return sorted(rows, key=lambda r: r["disagreement"], reverse=True)


# trials S3 may see: train + val only. lockbox stays SEALED (Section 7), since S3's plots feed an agent
# whose hypotheses reach DOMAIN_NOTES, and reading the lockbox would spend that independence.
def load_analysis_trials() -> list[Trial]:
    return [t for t in load_dataset() if t.split != "lockbox"]


# run the whole deterministic core into out_dir; returns the audit report
def run(out_dir: Path = S3_OUT_DIR, spec: WindowSpec | None = None) -> dict:
    spec = spec or champion_windowspec()  # match the champion/OOF grid the fusion joins on
    out_dir.mkdir(parents=True, exist_ok=True)
    trials = load_analysis_trials() # lockbox sealed (Section 7)

    # purge stale figures so a previous run's outputs can't leak to the agent, which reads every png
    plots_dir = out_dir / PLOTS_SUBDIR
    if plots_dir.exists():
        for png in plots_dir.glob("*.png"):
            png.unlink()

    table = build_anchor_table(trials, spec)
    if not table.empty: # Section 10.7: annotate each window's disagreement cause for the reviewer/agent
        table["disagree_reason"] = [
            disagree_reason(lab, v, ap, sc) for lab, v, ap, sc in zip(
                table["label"], table["swap_verdict_adaptive"],
                table["antiphase"], table["swap_count"])]
    table.to_csv(out_dir / ANCHORS_CSV, index=False)

    disagreement = label_disagreement(table)
    (out_dir / DISAGREEMENT_JSON).write_text(json.dumps(disagreement, indent=2),
                                             encoding="utf-8")

    report = audit_anchors(trials, spec)
    audit = {
        "rate_hz": CANONICAL_HZ,
        "decimated_hz": CANONICAL_HZ / AUDIT_FACTOR,
        "factor": AUDIT_FACTOR,
        "tol": AUDIT_TOL,
        "gate": "no anchor feature without a rate-invariance verdict (PLAN S3)",
        "anchors": report,
    }
    (out_dir / AUDIT_JSON).write_text(json.dumps(audit, indent=2), encoding="utf-8")

    plot_all(trials, report, out_dir / PLOTS_SUBDIR, spec)

    audit["disagreement"] = disagreement
    return audit


# run the deterministic core and print the audit verdicts + output locations
def main() -> None:
    parser = argparse.ArgumentParser(description="S3 physics deterministic core.")
    parser.add_argument("--out", type=Path, default=S3_OUT_DIR)
    parser.add_argument("--window-s", type=float, default=None,
                        help="window length in seconds (default: champion window)")
    args = parser.parse_args()

    spec = WindowSpec(window_s=args.window_s) if args.window_s else None  # None => champion window
    audit = run(args.out, spec)

    n_rows = sum(1 for _ in (args.out / ANCHORS_CSV).open(encoding="utf-8")) - 1
    r0 = audit["anchors"][ANCHOR_NAMES[0]]
    print(f"[s3] {n_rows:,} windows -> {args.out / ANCHORS_CSV}")
    print(f"[s3] rate audit @ {audit['rate_hz']:.0f} vs {audit['decimated_hz']:.0f} Hz "
          f"({r0['n_windows_walking']:,} walking windows), tol {audit['tol']:g}:")
    for a in ANCHOR_NAMES:
        r = audit["anchors"][a]
        flag = "ok " if r["verdict"] == "invariant" else "!! "
        print(f"    {flag}{a:14} {r['metric']} delta={r['median_delta']:.4f}  {r['verdict']}")
    print(f"[s3] plots -> {args.out / PLOTS_SUBDIR}")


if __name__ == "__main__":
    main()
