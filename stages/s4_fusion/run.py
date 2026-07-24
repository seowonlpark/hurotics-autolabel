# S4 fusion deterministic core: join the S2 out-of-fold labels with the S3 physics verdicts, apply the
# fuser, and score it. no agent, no judgement. writes the fused per-window table, a fused-vs-S2 report,
# the calibration, and the ranked disagreement cases. evaluates on train+val (both inputs non-lockbox).

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from stages.s2_ml.dataset import STAND, WALK
from stages.s2_ml.oof import OOF_CSV, S2_OUT_DIR, champion_oof
from stages.s3_physics.anchors import STANDING, WALKING
from stages.s3_physics.run import S3_OUT_DIR, ANCHORS_CSV
from stages.s4_fusion.fuse import HIGH, LOW, MED, FusionPolicy, derive_policy, fuse

REPO_ROOT = Path(__file__).resolve().parents[2]
S4_OUT_DIR = REPO_ROOT / "runs" / "s4_fusion"

FUSED_CSV = "fused_windows.csv"
REPORT_MD = "fusion_report.md"
METRICS_JSON = "fusion.json"
DISAGREEMENT_JSON = "disagreements.json"

JOIN_KEYS = ["rev", "trial", "segment", "t_start_ms"]


# the aligned per-window table: S2 out-of-fold label+proba joined to the S3 adaptive verdict,
# on the shared window grid. reads the S2 OOF artifact, regenerating it if absent.
def load_aligned(s2_dir: Path = S2_OUT_DIR, s3_dir: Path = S3_OUT_DIR) -> pd.DataFrame:
    oof_path = s2_dir / OOF_CSV
    oof = pd.read_csv(oof_path) if oof_path.exists() else champion_oof(s2_dir)
    anchors = pd.read_csv(s3_dir / ANCHORS_CSV)[JOIN_KEYS + ["swap_verdict_adaptive"]]
    # both inputs are non-lockbox by construction: S2 OOF is train-only, S3 seals the lockbox
    m = oof.merge(anchors, on=JOIN_KEYS, how="inner")
    return m.rename(columns={"label": "true", "swap_verdict_adaptive": "s3"})


# apply the fuser row-wise; adds fused_label, confidence, abstain. policy=None is the frozen 2-class
# default; the lockbox path passes NO policy on purpose, never one re-derived on the sealed revs (12.5).
def apply_fusion(df: pd.DataFrame, policy: FusionPolicy | None = None) -> pd.DataFrame:
    res = [fuse(int(r.s2_pred), float(r.s2_proba), r.s3, policy) for r in df.itertuples()]
    df = df.copy()
    df["fused_label"] = [f.label for f in res]
    df["confidence"] = [f.confidence for f in res]
    df["abstain"] = [f.abstain for f in res]
    return df


# per-class recall on the kept subset; None when the class is absent -- recall is undefined
# then, not zero, and reporting 0 (or a bare nan) would misstate a rev/tier that simply holds
# no windows of that class
def _recall(t: np.ndarray, p: np.ndarray, cls: int) -> float | None:
    mask = t == cls
    return round(float((p[mask] == cls).mean()), 4) if mask.any() else None


# macro-F1 / accuracy / per-class recall / coverage for pred against true, over an optional
# keep mask (the windows a controller would actually act on). metrics are None on an empty
# subset (e.g. a tier where everything abstained) rather than nan
def score(true: np.ndarray, pred: np.ndarray, keep: np.ndarray | None = None) -> dict:
    keep = np.ones(len(true), bool) if keep is None else keep
    t, p = true[keep], pred[keep]
    return {
        "coverage": round(float(keep.mean()), 4) if len(keep) else 0.0,
        "macro_f1": (round(float(f1_score(t, p, labels=[STAND, WALK], average="macro",
                                          zero_division=0)), 4) if t.size else None),
        "accuracy": round(float((t == p).mean()), 4) if t.size else None,
        "stand_recall": _recall(t, p, STAND),
        "walk_recall": _recall(t, p, WALK),
    }


# per-tier calibration: how often each confidence tier is actually correct (does HIGH mean high?)
def calibration(df: pd.DataFrame) -> dict:
    out = {}
    for tier in (HIGH, MED, LOW):
        g = df[df["confidence"] == tier]
        if len(g):
            out[tier] = {"n": int(len(g)), "share": round(len(g) / len(df), 4),
                         "accuracy": round(float((g["fused_label"] == g["true"]).mean()), 4)}
    return out


# per-rev breakdown: S2 alone vs fused (full coverage) vs fused acting-on-confidence, on the joined
# windows. the fused label only beats S2 on every rev once the abstention is applied.
def per_rev(df: pd.DataFrame) -> list[dict]:
    rows = []
    for rev, g in df.groupby("rev", sort=True):
        t = g["true"].to_numpy(int)
        keep = ~g["abstain"].to_numpy(bool)
        rows.append({
            "rev": rev, "n": int(len(g)),
            "s2": score(t, g["s2_pred"].to_numpy(int)),
            "fused": score(t, g["fused_label"].to_numpy(int)),
            "acting": score(t, g["fused_label"].to_numpy(int), keep=keep),
        })
    return rows


# rank trials by their share of LOW-confidence (disagreement) windows -- the cases the agent
# reviews, the fusion analogue of S3's physics-vs-label ranking
def disagreement_cases(df: pd.DataFrame) -> list[dict]:
    rows = []
    for (rev, trial), g in df.groupby(["rev", "trial"], sort=True):
        low = g[g["confidence"] == LOW]
        if not len(low):
            continue
        rows.append({
            "rev": rev, "trial": int(trial), "n_windows": int(len(g)),
            "n_low": int(len(low)), "low_share": round(len(low) / len(g), 3),
            "low_accuracy": round(float((low["fused_label"] == low["true"]).mean()), 3),
        })
    return sorted(rows, key=lambda r: r["n_low"], reverse=True)


# render the per-rev table: S2 vs fused (full) vs acting-on-confidence, macro-F1 / stand-recall
def render_per_rev(rows: list[dict]) -> str:
    L = ["## Per-rev breakdown (leave-one-rev-out; the deployment question)", "",
         "F1 = macro-F1, st = stand-recall, cov = coverage. Acting = fused with abstain-on-LOW.", "",
         "| rev | n | S2 F1 / st | fused F1 / st | acting F1 / acc / st / cov |",
         "|---|---|---|---|---|"]
    for r in rows:
        s, f, a = r["s2"], r["fused"], r["acting"]
        L.append(f"| {r['rev']} | {r['n']} | {s['macro_f1']} / {s['stand_recall']} | "
                 f"{f['macro_f1']} / {f['stand_recall']} | "
                 f"{a['macro_f1']} / {a['accuracy']} / {a['stand_recall']} / {a['coverage']} |")
    return "\n".join(L)


# render the human-readable fusion report
def render(fused: dict, s2: dict, cal: dict, act: dict, n: int, unmatched: int,
          rev_rows: list[dict]) -> str:
    L = [
        "# S4 fusion report", "",
        f"Joined **{n:,}** windows (S2 out-of-fold intersect S3 verdicts); {unmatched} S2 windows "
        "unmatched to S3. Non-lockbox (train+val); the lockbox opens once, at the end.", "",
        "## Fused vs S2 alone (full coverage)", "",
        "| model | macro-F1 | accuracy | stand-recall | walk-recall |",
        "|---|---|---|---|---|",
        f"| S2 alone | {s2['macro_f1']} | {s2['accuracy']} | {s2['stand_recall']} | {s2['walk_recall']} |",
        f"| **fused** | **{fused['macro_f1']}** | {fused['accuracy']} | {fused['stand_recall']} | {fused['walk_recall']} |",
        "", "## Confidence tiers (is the signal calibrated?)", "",
        "| tier | windows | share | accuracy |", "|---|---|---|---|",
    ]
    for tier in (HIGH, MED, LOW):
        if tier in cal:
            c = cal[tier]
            L.append(f"| {tier} | {c['n']:,} | {c['share']} | {c['accuracy']} |")
    L += [
        "", "## Acting on confidence (abstain on LOW = disagreement)", "",
        f"A controller that acts on HIGH+MED and holds on LOW covers **{act['coverage']}** of "
        f"windows at accuracy **{act['accuracy']}** (stand-recall {act['stand_recall']}, "
        f"walk-recall {act['walk_recall']}) - vs S2's {s2['accuracy']} at full coverage. The "
        "abstained windows are the disagreement cases the agent reviews.",
        "", render_per_rev(rev_rows),
    ]
    return "\n".join(L) + "\n"


# s3's verdict string mapped to a class label so the S4 finding check can score it against
# truth; AMBIGUOUS has no class (absent from the map -> None), it is not a stand/walk claim
_S3_TO_CLASS = {STANDING: STAND, WALKING: WALK}


# group the fused windows into the per-trial lookup findings are measured against: (rev, trial) -> row
# dicts. carries t_start in SECONDS to match the agent's coordinates, and pre-maps s3 to a class.
def fused_windows_lookup(df: pd.DataFrame) -> dict:
    out: dict = {}
    for r in df.itertuples():
        out.setdefault((r.rev, int(r.trial)), []).append({
            "t_start_s": float(r.t_start_ms) / 1000.0,
            "s2_pred": int(r.s2_pred),
            "true": int(r.true),
            "fused_label": int(r.fused_label),
            "s3_class": _S3_TO_CLASS.get(r.s3),
            "is_low": r.confidence == LOW,
        })
    return out


# run the whole deterministic core into out_dir; returns the metrics dict
def run(out_dir: Path = S4_OUT_DIR, s2_dir: Path = S2_OUT_DIR, s3_dir: Path = S3_OUT_DIR) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    aligned = load_aligned(s2_dir, s3_dir)
    n_oof = len(pd.read_csv(s2_dir / OOF_CSV)) if (s2_dir / OOF_CSV).exists() else len(aligned)
    # fused-label policy, read off THIS corpus's measured S2xS3 agreement (Section 12), train+val only.
    # on the current 2-class corpus it reproduces the frozen default exactly (locked by tests).
    policy = derive_policy(zip(aligned["s2_pred"].astype(int), aligned["s3"],
                               aligned["true"].astype(int)))
    df = apply_fusion(aligned, policy)

    true = df["true"].to_numpy(int)
    s2 = score(true, df["s2_pred"].to_numpy(int))
    fused = score(true, df["fused_label"].to_numpy(int))
    act = score(true, df["fused_label"].to_numpy(int), keep=~df["abstain"].to_numpy(bool))
    cal = calibration(df)
    cases = disagreement_cases(df)
    rev_rows = per_rev(df)

    df.to_csv(out_dir / FUSED_CSV, index=False)
    metrics = {"n_windows": int(len(df)), "unmatched": int(n_oof - len(df)),
               "s2_alone": s2, "fused": fused, "acting_on_confidence": act,
               "calibration": cal, "per_rev": rev_rows, "disagreement_cases": cases}
    (out_dir / METRICS_JSON).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out_dir / DISAGREEMENT_JSON).write_text(json.dumps(cases, indent=2), encoding="utf-8")
    (out_dir / REPORT_MD).write_text(
        render(fused, s2, cal, act, len(df), int(n_oof - len(df)), rev_rows), encoding="utf-8")
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="S4 fusion deterministic core.")
    ap.add_argument("--out", type=Path, default=S4_OUT_DIR)
    args = ap.parse_args()
    m = run(args.out)
    f, s, a = m["fused"], m["s2_alone"], m["acting_on_confidence"]
    print(f"[s4] {m['n_windows']:,} windows ({m['unmatched']} unmatched)")
    print(f"[s4] fused macro-F1 {f['macro_f1']} (S2 alone {s['macro_f1']}), "
          f"stand-recall {f['stand_recall']} (S2 {s['stand_recall']})")
    print(f"[s4] acting on HIGH+MED: coverage {a['coverage']} at accuracy {a['accuracy']} "
          f"(S2 {s['accuracy']} at full coverage)")
    for tier, c in m["calibration"].items():
        print(f"[s4]   {tier:6} tier: {c['n']:>5} windows ({c['share']:.2f}) acc {c['accuracy']}")
    print(f"[s4] artifacts -> {args.out}")


if __name__ == "__main__":
    main()
