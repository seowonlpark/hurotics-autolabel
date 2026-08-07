# 5 seeds through the shipped LORO row path; python -m stages.s2_ml.rowseedsweep
#
# `s2_ml_featseedsweep.json` measures the seed band at the WINDOW unit (5,984 label-pure
# windows). The headline this repo actually quotes -- coverage 84.66% at 0.9901 -- is at the
# ROW unit, over 1.24M scored rows, and it has only ever been measured at seed 0. The two are
# not interchangeable: the row path puts the ambiguity gate on top of the same fitted model,
# so the window band is a proxy for the row band and nobody had checked how good a proxy.
#
# This is a sweep, not a re-run of the claim. roweval.py stays single-seed on purpose (its
# threshold is not a flag for the same reason); the headline is what seed 0 measured, and this
# file is the band you read a CHANGE to that headline against.

from __future__ import annotations

import argparse
import json
import time

import pandas as pd

from runslayout import ABLATIONS
from stages.s2_ml.dataset import load_dataset
from stages.s2_ml.features import WindowSpec, build_windows, feature_columns
from stages.s2_ml.roweval import curve, per_rev, score_trials
from stages.s2_ml.train import (
    MODEL_PARAMS, PRESETS, build_model, load_spec, reference_stats, select_features, trainable,
)

OUT_PATH = ABLATIONS / "s2_ml_rowseedsweep.json"


# roweval.fit_on, with the seed lifted out of MODEL_PARAMS -- the ONLY thing that varies here
def fit_on(train_df: pd.DataFrame, feats: list[str], exclude_rev: str, seed: int):
    sub = train_df[train_df["rev"] != exclude_rev]
    model = build_model({**MODEL_PARAMS, "random_state": seed})
    model.fit(sub[feats].to_numpy(float), sub["label"].to_numpy(int))
    return model, reference_stats(sub, feats)


# one full leave-one-rev-out pass at `seed`, scored through the shipping path row by row
def one_seed(trials, train_df, feats, base_meta, threshold: float, seed: int) -> dict:
    parts = []
    for rev in sorted(train_df["rev"].unique()):
        model, ref = fit_on(train_df, feats, rev, seed)
        meta = {**base_meta, "reference_stats": ref}
        held = [t for t in trials if t.split == "train" and t.rev == rev]
        print(f"[rowsweep] seed {seed}: held-out {rev} ({len(held)} trials)")
        parts.append(score_trials(model, meta, held, threshold))
    df = pd.concat(parts, ignore_index=True)

    shipped = next(r for r in curve(df) if abs(r["threshold"] - threshold) < 1e-9)
    return {
        "seed": seed,
        "n_features": len(feats),
        "rows_scored": int(len(df)),
        "threshold": float(threshold),
        "coverage": shipped["coverage"],
        "selective_accuracy": shipped["selective_accuracy"],
        "worst_rev_accuracy": shipped["worst_rev_accuracy"],
        "worst_rev": shipped["worst_rev"],
        "errors_kept": shipped["errors_kept"],
        # the whole curve, so a later threshold question does not need another 7-fold refit
        "curve": curve(df),
        "per_rev": [{k: r[k] for k in ("rev", "coverage", "accuracy", "stand_recall")}
                    for r in per_rev(df, threshold)],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5, help="seeds 0..n-1")
    ap.add_argument("--out", default=str(OUT_PATH))
    args = ap.parse_args()

    threshold = PRESETS["balanced"]
    trials = load_dataset()
    spec = WindowSpec()
    windows = build_windows([t for t in trials if t.split == "train"], spec)
    feats = select_features(feature_columns(windows), load_spec()["drop_features"])
    train_df = trainable(windows, "train")
    base_meta = {"window_s": spec.window_s, "fs_hz": spec.fs_hz,
                 "inference_stride_s": 0.25, "features": feats}

    rows = []
    for seed in range(args.seeds):
        t0 = time.perf_counter()
        rows.append(one_seed(trials, train_df, feats, base_meta, threshold, seed))
        r = rows[-1]
        print(f"[rowsweep] seed {seed}: coverage {r['coverage']:.4f}  "
              f"sel_acc {r['selective_accuracy']:.4f}  "
              f"worst {r['worst_rev_accuracy']:.4f} ({r['worst_rev']})  "
              f"[{time.perf_counter() - t0:.0f}s]\n")

    out = __import__("pathlib").Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")

    for key in ("selective_accuracy", "coverage", "worst_rev_accuracy"):
        v = [r[key] for r in rows]
        print(f"[rowsweep] {key:<20} mean {sum(v) / len(v):.4f}  min {min(v):.4f}  "
              f"max {max(v):.4f}  spread {max(v) - min(v):.4f}")
    print(f"[rowsweep] wrote {out}")


if __name__ == "__main__":
    main()
