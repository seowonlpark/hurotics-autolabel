"""S2 train: fit the champion, score it honestly, write the artifacts.

    python -m stages.s2_ml.train --out runs/s2_ml

Two guarantees enforced in code, not in comments:

  1. **The lockbox is never touched.** Whole revs are sealed (§7). Training asserts it, so
     a refactor that quietly widens the split fails loudly instead of producing an
     optimistic final number nobody can trust again.
  2. **Validation is leave-one-rev-out.** A rev is one subject on one day, so holding a
     whole rev out is the closest honest stand-in for "a new person on a new day", which
     is the deployment question. Random k-fold would split one subject's trials across
     train and test and report a flattering, meaningless score.

The model is ExtraTrees rather than RandomForest. Measured on identical features, folds and
params (42 features, leave-one-rev-out, the `MODEL_PARAMS` below) over 5 seeds: accuracy
0.9625 ±0.0003 against 0.9568 ±0.0008, macro-F1 0.9194 against 0.9138 — seed ranges
disjoint, so the gap is the model and not the draw. At threshold 0.85 the two commit at the
SAME accuracy (0.9887 vs 0.9889, ranges overlapping) while ExtraTrees commits to 87.6% of
windows against 81.9%: equal precision over more of the data, which is what the abstention
layer consumes.

Two honesties about that comparison. RandomForest is genuinely better on balanced accuracy
(0.9246 vs 0.8995, disjoint) — it recovers more of the smaller class, paid for out of
overall accuracy. And at a single seed it appeared to hold a worst-subject edge at 0.85
(0.9754 vs 0.9699), which is why per-seed numbers are quoted here at all: over 5 seeds that
edge dissolves into noise (0.9727 ±0.0037 vs 0.9699 ±0.0001, overlapping) and ExtraTrees is
the far steadier of the two on that metric. Reproduce from `runs/s2_ml_rf42` (single RF fit
at these features) and `runs/s2_ml_seedsweep.json` (the 5-seed comparison).

Gradient boosting scored marginally higher raw accuracy but was badly overconfident, which
is the worse failure here — an overconfident model does not abstain when it should. That
one is inherited from the 23-feature stage and has NOT been re-measured at 42 features.

Alongside the model this writes the reference statistics `label.py` needs to explain WHY a
row is ambiguous. They are measured here, on training windows only, so no magic numbers
live in the labelling path and every one is reproducible by re-running this.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import LeaveOneGroupOut

from stages.s2_ml.dataset import STAND, TRAIN_CLASSES, WALK, load_dataset
from stages.s2_ml.features import WindowSpec, build_windows, feature_columns
from stages.s2_ml.locoeval import evaluate, render, save, selective_curve

REPO_ROOT = Path(__file__).resolve().parents[2]

MODEL_PARAMS = dict(n_estimators=400, random_state=0, n_jobs=-1, class_weight="balanced")

# Inference slides the same window at a fraction of its length, so a row's probability is
# an average over several overlapping views. Training stays non-overlapping (§9).
DEFAULT_INFERENCE_STRIDE_S = 0.25

# Named operating points; OPERATING_POINTS.md carries the measured tradeoff behind each.
# `balanced` is the default because it is the lowest threshold at which every held-out
# subject independently clears 95% accuracy on the rows it commits to.
PRESETS = {"high_coverage": 0.70, "balanced": 0.85, "high_precision": 0.95}
DEFAULT_PRESET = "balanced"


def build_model() -> ExtraTreesClassifier:
    return ExtraTreesClassifier(**MODEL_PARAMS)


def trainable(df: pd.DataFrame, split: str = "train") -> pd.DataFrame:
    """Label-pure windows of one split. Transitions are excluded from targets (§5.2).

    Selected positively, on membership of the trained classes. Excluding TRANSITION and
    None by name instead lets any third label state through silently, which is how 84
    all-unknown windows reached `to_numpy(int)` and crashed it -- a loud failure that
    would have been a quiet contamination had the codes been numeric.
    """
    return df[(df["split"] == split)
              & df["label"].isin(TRAIN_CLASSES)].reset_index(drop=True)


def cross_validate(df: pd.DataFrame, feats: list[str]) -> np.ndarray:
    """Leave-one-rev-out out-of-fold P(walk); every window scored by a model blind to its rev."""
    X = df[feats].to_numpy(float)
    y = df["label"].to_numpy(int)
    oof = np.zeros(len(y))
    for tr, te in LeaveOneGroupOut().split(X, y, df["rev"].to_numpy()):
        model = build_model()
        model.fit(X[tr], y[tr])
        oof[te] = model.predict_proba(X[te])[:, list(model.classes_).index(WALK)]
    return oof


def reference_stats(df: pd.DataFrame, feats: list[str]) -> dict:
    """Distributional constants the ambiguity reasons are stated against.

    Measured on training windows only. `label.py` reads these instead of hard-coding
    thresholds, so the explanations move with the data rather than with an author's memory.
    """
    walk = df[df["label"] == WALK]
    # How far BOTH thighs sit above this recording's own standing posture. Upright standing
    # and level walking both keep it near zero; the standing class carries a long tail that
    # walking does not, and those windows concentrate in the trials whose `stand` runs cover
    # sitting and transfers. It is therefore a posture signature for a state outside this
    # two-class taxonomy, not a stand-vs-walk discriminator.
    posture = np.minimum(df["L_ang_med_rest"], df["R_ang_med_rest"])
    return {
        # A window predicted walk whose interleg excursion sits below where real walking
        # lives is either very slow gait or standing sway. Either way, worth flagging.
        "walk_minhalf_p05": float(np.percentile(walk["ileg_minhalf"], 5)),
        "posture_shift_p99": float(np.percentile(posture, 99)),
        # Per-feature training range, for the out-of-distribution check.
        "feat_p01": {c: float(np.percentile(df[c], 1)) for c in feats},
        "feat_p99": {c: float(np.percentile(df[c], 99)) for c in feats},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/s2_ml")
    ap.add_argument("--window-s", type=float, default=None)
    args = ap.parse_args()

    out_dir = (REPO_ROOT / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    spec = (WindowSpec(window_s=args.window_s, stride_s=args.window_s)
            if args.window_s else WindowSpec())
    trials = load_dataset()
    windows = build_windows(trials, spec)
    feats = feature_columns(windows)

    train_df = trainable(windows, "train")
    assert "lockbox" not in set(train_df["split"]), "lockbox leaked into training"
    revs = sorted(train_df["rev"].unique())
    print(f"[s2] window={spec.window_s}s  train windows={len(train_df):,}  "
          f"features={len(feats)}  revs={len(revs)} {revs}")

    oof = cross_validate(train_df, feats)
    y = train_df["label"].to_numpy(int)
    result = evaluate(y, np.where(oof >= 0.5, WALK, STAND),
                      groups=train_df["rev"].to_numpy(),
                      unknown_frac=train_df["unknown_frac"].to_numpy())
    curve = selective_curve(y, oof, groups=train_df["rev"].to_numpy())

    print(f"[s2] leave-one-rev-out macro-F1 = {result.macro_f1:.4f}  "
          f"(acc {result.accuracy:.4f}, balanced {result.balanced_accuracy:.4f})")
    for rev, f1 in sorted(result.per_rev_macro_f1.items()):
        print(f"[s2]   held-out {rev}: macro-F1 {f1:.4f}")
    print("[s2] window-level selective accuracy:")
    for row in curve:
        if row["threshold"] in tuple(PRESETS.values()):
            print(f"[s2]   thr {row['threshold']:.2f}: coverage {row['coverage']:.4f}  "
                  f"selective_acc {row['selective_accuracy']:.4f}  "
                  f"worst_rev {row['worst_rev_accuracy']:.4f}")

    # Champion: refit on every training rev. The lockbox stays sealed.
    model = build_model()
    model.fit(train_df[feats].to_numpy(float), y)
    importances = sorted(zip(feats, model.feature_importances_), key=lambda x: -x[1])

    save(result, out_dir / "locoeval.json", curve=curve)
    (out_dir / "locoeval.md").write_text(
        render(result, curve, title="S2 champion - leave-one-rev-out CV")
        + "\n\n## Feature importance (top 12)\n\n"
        + "\n".join(f"- `{n}`: {v:.4f}" for n, v in importances[:12]) + "\n",
        encoding="utf-8",
    )
    (out_dir / "model_meta.json").write_text(json.dumps({
        "model": "ExtraTreesClassifier",
        "params": dict(MODEL_PARAMS),
        "window_s": spec.window_s, "stride_s": spec.stride_s, "fs_hz": spec.fs_hz,
        "inference_stride_s": DEFAULT_INFERENCE_STRIDE_S,
        "features": feats,
        "train_revs": revs,
        "n_train_windows": int(len(train_df)),
        "cv": "LeaveOneGroupOut(rev)",
        "macro_f1_cv": result.macro_f1,
        "presets": PRESETS,
        "default_preset": DEFAULT_PRESET,
        "reference_stats": reference_stats(train_df, feats),
    }, indent=2), encoding="utf-8")

    try:
        import joblib
        joblib.dump(model, out_dir / "champion.joblib")
    except Exception as exc:  # metrics are the deliverable; the artifact is convenience
        print(f"[s2] WARNING: could not persist model ({exc})")

    print(f"[s2] artifacts -> {out_dir}")


if __name__ == "__main__":
    main()
