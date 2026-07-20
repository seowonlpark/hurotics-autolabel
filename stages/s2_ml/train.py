"""S2 train: fit a model, score it honestly, write the artifacts.

    python -m stages.s2_ml.train --out runs/s2_ml

Deterministic core of the champion/challenger loop (PLAN S2). The agent layer proposes
changes; this module runs them and reports numbers. It never decides what is "best".

Two guarantees it enforces in code, not in comments:

  1. **The lockbox is never touched.** Whole revs are sealed (§7). Training asserts it,
     so a future refactor that quietly widens the split fails loudly instead of
     producing an optimistic final number nobody can trust again.
  2. **Validation is leave-one-rev-out.** A rev is one subject on one day, so holding a
     whole rev out is the closest honest stand-in for "a new person on a new day" —
     the deployment question. Random k-fold would split one subject's trials across
     train and test and report a flattering, meaningless score.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut

from stages.s2_ml.dataset import load_dataset
from stages.s2_ml.features import TRANSITION, WindowSpec, build_windows, feature_columns
from stages.s2_ml.locoeval import evaluate, render, save, transition_report

REPO_ROOT = Path(__file__).resolve().parents[2]

# Random Forest is the documented starting model (§9). Depth is left unbounded; the
# windowed feature count is small and the champion/challenger loop is where tuning
# belongs, not a hand-picked constant here.
MODEL_PARAMS = dict(n_estimators=300, random_state=0, n_jobs=-1, class_weight="balanced")


def build_model() -> RandomForestClassifier:
    return RandomForestClassifier(**MODEL_PARAMS)


def trainable(df: pd.DataFrame, split: str = "train") -> pd.DataFrame:
    """Label-pure windows of one split. Transitions are excluded from targets (§5.2)."""
    return df[(df["split"] == split) & (df["label"] != TRANSITION)].reset_index(drop=True)


def cross_validate(df: pd.DataFrame, feats: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Leave-one-rev-out out-of-fold predictions. Returns (y_true, y_pred)."""
    X = df[feats].to_numpy(float)
    y = df["label"].to_numpy(int)
    groups = df["rev"].to_numpy()

    oof = np.empty_like(y)
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        model = build_model()
        model.fit(X[tr], y[tr])
        oof[te] = model.predict(X[te])
    return y, oof


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/s2_ml")
    ap.add_argument("--window-s", type=float, default=None,
                    help="window length in seconds (§9 open tradeoff)")
    args = ap.parse_args()

    out_dir = (REPO_ROOT / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    spec = WindowSpec(window_s=args.window_s, stride_s=args.window_s) if args.window_s else WindowSpec()
    windows = build_windows(load_dataset(), spec)
    feats = feature_columns(windows)

    train_df = trainable(windows, "train")
    assert "lockbox" not in set(train_df["split"]), "lockbox leaked into training"
    revs = sorted(train_df["rev"].unique())
    print(f"[s2] window={spec.window_s}s  train windows={len(train_df):,}  "
          f"features={len(feats)}  revs={len(revs)} {revs}")

    y, oof = cross_validate(train_df, feats)
    result = evaluate(y, oof, groups=train_df["rev"].to_numpy(),
                      unknown_frac=train_df["unknown_frac"].to_numpy())
    trans = transition_report(train_df, oof)

    print(f"[s2] leave-one-rev-out macro-F1 = {result.macro_f1:.4f}  "
          f"(acc {result.accuracy:.4f}, balanced {result.balanced_accuracy:.4f})")
    for rev, f1 in sorted(result.per_rev_macro_f1.items()):
        print(f"[s2]   held-out {rev}: macro-F1 {f1:.4f}")

    # Final model: refit on every training rev. The lockbox stays sealed.
    model = build_model()
    model.fit(train_df[feats].to_numpy(float), train_df["label"].to_numpy(int))

    importances = sorted(zip(feats, model.feature_importances_), key=lambda x: -x[1])

    save(result, out_dir / "locoeval.json", trans)
    (out_dir / "locoeval.md").write_text(
        render(result, trans, title="S2 champion — leave-one-rev-out CV") + "\n\n"
        + "## Feature importance (top 12)\n\n"
        + "\n".join(f"- `{n}`: {v:.4f}" for n, v in importances[:12]) + "\n",
        encoding="utf-8",
    )
    (out_dir / "model_meta.json").write_text(json.dumps({
        "model": "RandomForestClassifier",
        "params": MODEL_PARAMS,
        "window_s": spec.window_s, "stride_s": spec.stride_s, "fs_hz": spec.fs_hz,
        "features": feats,
        "train_revs": revs,
        "n_train_windows": int(len(train_df)),
        "cv": "LeaveOneGroupOut(rev)",
        "macro_f1_cv": result.macro_f1,
    }, indent=2), encoding="utf-8")

    try:
        import joblib
        joblib.dump(model, out_dir / "champion.joblib")
    except Exception as exc:  # model artifact is optional; metrics are not
        print(f"[s2] WARNING: could not persist model ({exc})")

    print(f"[s2] artifacts -> {out_dir}")


if __name__ == "__main__":
    main()
