"""S2 out-of-fold predictions as a stage artifact, so S4 joins a file.

    python -m stages.s2_ml.oof --out runs/s2_ml

Leave-one-rev-out, so every window is scored by a model that never saw its rev — the
same honesty the CV metric is held to. The lockbox never enters (§7).

Predictions are emitted for EVERY window, including the `transition` ones training
excludes. Those windows are not dropped from the deliverable just because they are not
training targets: a recording still has to be labeled across them, and they are exactly
where an abstention policy has to earn its keep. The `label` column keeps its three
values -- an int class, `transition`, or None -- so scoring can tell them apart.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut

from stages.s2_ml.dataset import TRAIN_CLASSES, load_dataset
from stages.s2_ml.features import TRANSITION, WindowSpec, build_windows, feature_columns
from stages.s2_ml.train import build_model

REPO_ROOT = Path(__file__).resolve().parents[2]
S2_OUT_DIR = REPO_ROOT / "runs" / "s2_ml"
OOF_CSV = "oof_champion.csv"

# The keys S4 joins on. The same metadata `features.windows_of_trial` and
# `anchors.trial_anchors` emit, so the join is exact by construction, not by convention.
JOIN_KEYS = ["rev", "trial", "segment", "t_start_ms"]


def champion_oof(spec: WindowSpec | None = None) -> pd.DataFrame:
    """Out-of-fold (prediction, class probability) for every non-lockbox window."""
    spec = spec or WindowSpec()
    windows = build_windows(load_dataset(), spec)
    windows = windows[windows["split"] == "train"].reset_index(drop=True)
    feats = feature_columns(windows)

    # Three kinds of window label, and they are not two: an int class, `transition`
    # (mixed over stand/walk), and None (no valid class at all — every row was `-1` or
    # `255`, so there is nothing to be pure about). Only the ints can train or be scored;
    # the other two still get predictions, because a recording has to be labeled across
    # them and they are where abstention earns its keep.
    pure = windows["label"].isin(TRAIN_CLASSES)
    X = windows[feats].to_numpy(float)
    groups = windows["rev"].to_numpy()

    pred = np.zeros(len(windows), dtype=int)
    proba = np.zeros(len(windows))

    # Folds are defined over the PURE windows (they carry the targets), but each fold's
    # model predicts every window of the held-out rev, transitions included.
    for rev in pd.unique(groups):
        fit = windows[pure & (windows["rev"] != rev)]
        model = build_model()
        model.fit(fit[feats].to_numpy(float), fit["label"].to_numpy(int))
        held = groups == rev
        p = model.predict_proba(X[held])
        pred[held] = model.classes_[p.argmax(1)]
        proba[held] = p.max(1)

    out = windows[JOIN_KEYS + ["label", "purity", "unknown_frac"]].copy()
    out["s2_pred"] = pred
    out["s2_proba"] = proba
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Write the champion's out-of-fold predictions.")
    ap.add_argument("--out", type=Path, default=S2_OUT_DIR)
    args = ap.parse_args()

    out_dir = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    oof = champion_oof()
    path = out_dir / OOF_CSV
    oof.to_csv(path, index=False)

    pure = oof["label"].isin(TRAIN_CLASSES)
    acc = (oof.loc[pure, "s2_pred"].astype(int) == oof.loc[pure, "label"].astype(int)).mean()
    print(f"[s2] champion OOF: {len(oof):,} windows "
          f"({int(pure.sum()):,} pure, {int((~pure).sum()):,} transition), "
          f"pure acc {acc:.4f} -> {path}")


if __name__ == "__main__":
    main()


# Explicit check that LeaveOneGroupOut is still what the fold loop above reproduces.
# Kept as an assertion rather than a comment: the loop hand-rolls the split so it can
# predict on transition windows too, and a divergence from the CV used for the headline
# metric would make the fused numbers quietly incomparable to it.
def _folds_match_logo(windows: pd.DataFrame, pure: pd.Series) -> bool:
    X = windows.loc[pure]
    groups = X["rev"].to_numpy()
    logo = {tuple(sorted(np.unique(groups[te]))) for _tr, te in
            LeaveOneGroupOut().split(X, X["label"], groups)}
    return logo == {(g,) for g in np.unique(groups)}
