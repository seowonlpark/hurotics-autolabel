# S2 out-of-fold predictions as a stage artifact, so S4 fusion joins a file rather than
# recomputing the model (filesystem is the interface, PLAN principle 2). reproduces the
# CHAMPION exactly -- its full spec (dropped features, window/stride, model params via
# champion_config), same leave-one-rev-out folds, so every window is scored only by a model
# that never saw its rev -- and adds the class probability the hard-label CV path (train.loro)
# discards. lockbox never enters: OOF is over training revs only.

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut

from stages.s2_ml.dataset import load_dataset
from stages.s2_ml.experiment import resolve_champion, select_features
from stages.s2_ml.features import WindowSpec, build_windows, feature_columns
from stages.s2_ml.train import build_model, trainable

REPO_ROOT = Path(__file__).resolve().parents[2]
S2_OUT_DIR = REPO_ROOT / "runs" / "s2_ml"
OOF_CSV = "oof_champion.csv"

# the keys S4 joins on, plus the ground-truth label carried through for scoring
META_KEYS = ["rev", "trial", "segment", "t_start_ms", "label"]


# leave-one-rev-out out-of-fold (prediction, max-class probability) for the FULL champion spec
# -- its dropped features, window/stride, AND model params (champion_config), so oof_champion.csv
# is the real champion's fusion input, not a default-window/default-param stand-in that would
# silently describe a different model the moment a non-default champion is promoted.
def champion_oof(out_dir: Path = S2_OUT_DIR, spec: WindowSpec | None = None) -> pd.DataFrame:
    wspec, params, drops = resolve_champion(out_dir, window_override=spec)
    windows = build_windows(load_dataset(), wspec)
    df = trainable(windows, "train").copy()
    feats = select_features(feature_columns(windows), drops)

    X = df[feats].to_numpy(float)
    y = df["label"].to_numpy(int)
    groups = df["rev"].to_numpy()

    pred = np.empty_like(y)
    proba = np.zeros(len(y))
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        model = build_model(params)
        model.fit(X[tr], y[tr])
        p = model.predict_proba(X[te])
        pred[te] = model.classes_[p.argmax(1)]
        proba[te] = p.max(1)

    df["s2_pred"] = pred
    df["s2_proba"] = proba
    return df[META_KEYS + ["s2_pred", "s2_proba"]].reset_index(drop=True)


# write the champion OOF artifact and print a one-line summary
def main() -> None:
    ap = argparse.ArgumentParser(description="Write the champion's out-of-fold predictions.")
    ap.add_argument("--out", type=Path, default=S2_OUT_DIR)
    args = ap.parse_args()

    oof = champion_oof(args.out)
    path = args.out / OOF_CSV
    oof.to_csv(path, index=False)
    acc = (oof["s2_pred"] == oof["label"]).mean()
    print(f"[s2] champion OOF: {len(oof):,} windows, acc {acc:.4f} -> {path}")


if __name__ == "__main__":
    main()
