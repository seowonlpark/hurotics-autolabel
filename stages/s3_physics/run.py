# S3 deterministic core: compute the anchor table, run the rate-invariance audit, draw the
# plots. no agent, no judgement -- this is everything the physics stage can assert with code
# before Claude reads a single figure. the outputs (anchors.csv, rate_audit.json, plots/) are
# the interface: the hypothesis agent reads them and writes hypotheses.jsonl, and the PLAN S3
# gate ("no anchor without a rate-invariance verdict") is satisfied by rate_audit.json here.
# see README / PLAN S3 / DOMAIN_NOTES §10.

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from stages.s2_ml.dataset import STAND, WALK, Trial, load_dataset
from stages.s2_ml.features import WindowSpec
from stages.s3_physics.anchors import ANCHOR_NAMES, WALKING, trial_anchors
from stages.s3_physics.plots import plot_all
from stages.s3_physics.rate_audit import (
    AUDIT_FACTOR, AUDIT_TOL, CANONICAL_HZ, audit_anchors,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
S3_OUT_DIR = REPO_ROOT / "runs" / "s3_physics"

ANCHORS_CSV = "anchors.csv"
AUDIT_JSON = "rate_audit.json"
DISAGREEMENT_JSON = "disagreement.json"
PLOTS_SUBDIR = "plots"


# the per-window anchor table for the whole corpus, one row per window
def build_anchor_table(trials: list[Trial], spec: WindowSpec | None = None) -> pd.DataFrame:
    spec = spec or WindowSpec()
    frames = [a for t in trials if not (a := trial_anchors(t, spec)).empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# per-trial physics-vs-label disagreement, ranked. two directions the swap rule (§10) and
# the human label can part ways: a WALK-labeled window the physics does NOT call walking
# (slow-cadence tail, §10.3), and a STAND-labeled window it DOES (a mislabeled ramp, the S4
# label-audit precedent). ranked on the STRIDE-ADAPTIVE verdict (§10.6): the fixed-window
# verdict flags slow-gait windows the rule simply can't resolve, drowning real label issues in
# window artifacts -- the adaptive call resolves those, so what remains ranked is more likely a
# genuine label problem. this is where the agent should look first, so code ranks it rather
# than making the agent read all 44 figures blind.
def label_disagreement(table: pd.DataFrame) -> list[dict]:
    rows = []
    for (rev, trial), g in table.groupby(["rev", "trial"], sort=True):
        walk_lab = g[g["label"] == WALK]
        stand_lab = g[g["label"] == STAND]
        walk_not_walking = float((walk_lab["swap_verdict_adaptive"] != WALKING).mean()) if len(walk_lab) else 0.0
        stand_is_walking = float((stand_lab["swap_verdict_adaptive"] == WALKING).mean()) if len(stand_lab) else 0.0
        rows.append({
            "rev": rev, "trial": int(trial), "n_windows": int(len(g)),
            "walk_labeled_not_walking": round(walk_not_walking, 3),
            "stand_labeled_is_walking": round(stand_is_walking, 3),
            "disagreement": round(max(walk_not_walking, stand_is_walking), 3),
            "plot": f"{PLOTS_SUBDIR}/trial_{rev}_t{int(trial)}.png",
        })
    return sorted(rows, key=lambda r: r["disagreement"], reverse=True)


# load the trials S3 is allowed to see: train + val only. the lockbox (rev8, rev13) stays
# SEALED (§7) -- S3 is physics enrichment, but its plots feed an agent whose hypotheses reach
# DOMAIN_NOTES and the human's understanding, so letting it read the lockbox would spend that
# independence just as surely as training on it would. the swap rule was validated against a
# held-out rev8 once (§10) and that verdict is recorded; the ongoing stage does not re-open it.
def load_analysis_trials() -> list[Trial]:
    return [t for t in load_dataset() if t.split != "lockbox"]


# run the whole deterministic core into out_dir; returns the audit report
def run(out_dir: Path = S3_OUT_DIR, spec: WindowSpec | None = None) -> dict:
    spec = spec or WindowSpec()
    out_dir.mkdir(parents=True, exist_ok=True)
    trials = load_analysis_trials() # lockbox sealed (§7)

    # purge stale figures so a previous run's outputs (e.g. lockbox trials) cannot leak to
    # the agent, which reads every plots/*.png
    plots_dir = out_dir / PLOTS_SUBDIR
    if plots_dir.exists():
        for png in plots_dir.glob("*.png"):
            png.unlink()

    table = build_anchor_table(trials, spec)
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

    spec = WindowSpec(window_s=args.window_s) if args.window_s else WindowSpec()
    audit = run(args.out, spec)

    n_rows = sum(1 for _ in (args.out / ANCHORS_CSV).open(encoding="utf-8")) - 1
    r0 = audit["anchors"][ANCHOR_NAMES[0]]
    print(f"[s3] {n_rows:,} windows -> {args.out / ANCHORS_CSV}")
    print(f"[s3] rate audit @ {audit['rate_hz']:.0f} vs {audit['decimated_hz']:.0f} Hz "
          f"({r0['n_windows_walking']:,} walking windows), tol {audit['tol']:g}:")
    for a in ANCHOR_NAMES:
        r = audit["anchors"][a]
        flag = "ok " if r["verdict"] == "invariant" else "!! "
        print(f"    {flag}{a:14} {r['metric']} Δ={r['median_delta']:.4f}  {r['verdict']}")
    print(f"[s3] plots -> {args.out / PLOTS_SUBDIR}")


if __name__ == "__main__":
    main()
