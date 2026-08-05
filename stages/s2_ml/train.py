# S2 train: fit the champion, score it leave-one-rev-out, write the artifacts; lockbox untouched
#
# WHY ExtraTrees and not RandomForest -- `champion_spec.json`'s rationale points here for this,
# so it lives here rather than in the spec, where every edit re-trips the input stamps of three
# stages that declare it. Both fitted on identical features, folds and params, 5 seeds each
# (`runs/keep/ablations/s2_ml_seedsweep.json`):
#
#   ExtraTrees     macro-F1 0.9194   coverage @0.85 0.8760   selective accuracy 0.9887
#   RandomForest   macro-F1 0.9138   coverage @0.85 0.8189   selective accuracy 0.9889
#
# The precision a caller gets is the same to within a thousandth. What differs is how much of
# the corpus is left to be precise ABOUT: RandomForest abstains on nearly one window in five,
# ExtraTrees on one in eight. The usual explanation -- randomized splits decorrelate the trees,
# so the vote spreads out instead of piling up against the threshold -- is not measured here;
# what is measured is the coverage gap, and it is what the choice rests on.
# `runs/keep/ablations/s2_ml_rf42` is the earlier single-seed RF fit at 42 features that the
# sweep supersedes; it is kept because it is the only copy of that measurement.

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import LeaveOneGroupOut

from freshness import stamp_inputs
from runslayout import REGEN
from stages.s2_ml.dataset import STAND, TRAIN_CLASSES, WALK, load_dataset
from stages.s2_ml.features import (
    WindowSpec,
    amplitude_band,
    build_windows,
    feature_columns,
)
from stages.s2_ml.locoeval import evaluate, render, save, selective_curve
from stages.report import add_report_flag

REPO_ROOT = Path(__file__).resolve().parents[2]

# the champion, DECLARED and git-tracked, so the claim travels with the code instead of with runs/
CHAMPION_SPEC_PATH = Path(__file__).resolve().parent / "champion_spec.json"


def load_spec(path: Path = CHAMPION_SPEC_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# READ from the spec, not restated: a promotion rewrites it, and a second copy would trip the assert
MODEL_PARAMS = load_spec()["params"]

# inference slides the window, so a row averages several overlapping views; training does not
DEFAULT_INFERENCE_STRIDE_S = 0.25

# named operating points; `balanced` is the lowest threshold where every held-out subject clears 95%
PRESETS = {"high_coverage": 0.70, "balanced": 0.85, "high_precision": 0.95}
DEFAULT_PRESET = "balanced"


# the champion estimator; the CLASS is not a parameter- a proposal may retune, not replace
def build_model(params: dict | None = None) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(**(params or MODEL_PARAMS))


# the feature set minus `drop`; an unknown name is fatal- a typo would record a no-op as an ablation
def select_features(all_feats: list[str], drop: list[str]) -> list[str]:
    if (unknown := [f for f in drop if f not in all_feats]):
        raise SystemExit(
            f"[s2] cannot drop {unknown}: not in the {len(all_feats)} features this corpus "
            f"builds. Check the spelling against runs/regen/s2_ml/model_meta.json."
        )
    return [f for f in all_feats if f not in drop]


# refuse to fit a champion the spec does not describe; a DECLARATION check, not a quality gate
def assert_matches_spec(spec: dict, model, window: WindowSpec, feats: list[str],
                        dropped: list[str]) -> None:
    got = {
        "model": type(model).__name__,
        "window_s": window.window_s,
        "stride_s": window.stride_s,
        "fs_hz": window.fs_hz,
        "n_features": len(feats),
    }
    bad = [f"  {k}: spec says {spec[k]!r}, code builds {v!r}"
           for k, v in got.items() if spec[k] != v]
    if sorted(spec["drop_features"]) != sorted(dropped):
        bad.append(f"  drop_features: spec drops {sorted(spec['drop_features'])}, "
                   f"this run dropped {sorted(dropped)}")
    if bad:
        # ASCII only: an em-dash on a cp949 console turns a clear refusal into a UnicodeEncodeError
        raise SystemExit(
            f"[s2] champion drift - {CHAMPION_SPEC_PATH.name} does not describe this code:\n"
            + "\n".join(bad)
            + "\n[s2] Fix ONE of them on purpose: revert the code, or update the spec and "
              "re-run to record what the new champion actually scores."
        )


# label-pure windows of one split, selected POSITIVELY: excluding by name lets a third state through
def trainable(df: pd.DataFrame, split: str = "train") -> pd.DataFrame:
    return df[(df["split"] == split)
              & df["label"].isin(TRAIN_CLASSES)].reset_index(drop=True)


# leave-one-rev-out OOF P(walk); challengers score through THIS function so the gate compares like
def cross_validate(df: pd.DataFrame, feats: list[str],
                   params: dict | None = None) -> np.ndarray:
    X = df[feats].to_numpy(float)
    y = df["label"].to_numpy(int)
    oof = np.zeros(len(y))
    for tr, te in LeaveOneGroupOut().split(X, y, df["rev"].to_numpy()):
        model = build_model(params)
        model.fit(X[tr], y[tr])
        oof[te] = model.predict_proba(X[te])[:, list(model.classes_).index(WALK)]
    return oof


# the constants the ambiguity reasons are stated against; label.py reads them instead of hard-coding
def reference_stats(df: pd.DataFrame, feats: list[str]) -> dict:
    walk = df[df["label"] == WALK]
    stand = df[df["label"] == STAND]
    # how far BOTH thighs sit above this recording's own standing posture- a signature for sitting
    posture = np.minimum(df["L_ang_med_rest"], df["R_ang_med_rest"])
    band_lo, band_hi = amplitude_band(df["ileg_minhalf"], df["label"])
    return {
        # a walk prediction below where real walking lives is slow gait or sway; either is worth a flag
        "walk_minhalf_p05": float(np.percentile(walk["ileg_minhalf"], 5)),
        "posture_shift_p99": float(np.percentile(posture, 99)),
        # the ambiguity band, so `label.py` reads it off the champion rather than recomputing at serve
        "band_lo": band_lo,
        "band_hi": band_hi,
        # Per-feature training range, for the out-of-distribution check
        "feat_p01": {c: float(np.percentile(df[c], 1)) for c in feats},
        "feat_p99": {c: float(np.percentile(df[c], 99)) for c in feats},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REGEN / "s2_ml"))
    add_report_flag(ap)
    # The ONE override here, because it is the one whose output is read: an ablation's `locoeval`
    # sits beside the champion's in `breakdown`. There is deliberately no `--window-s` twin -- the
    # window is a field of `ExperimentSpec`, so a window change belongs in the ledger, where it is
    # recorded with the rationale that motivated it and the decision it drew.
    ap.add_argument("--drop", nargs="+", metavar="FEATURE", default=None,
                    help="ablation: train without these features instead of the spec's "
                         "drop_features. Marks the run an experiment, not the champion.")
    args = ap.parse_args()

    out_dir = (REPO_ROOT / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    spec = WindowSpec()
    trials = load_dataset()
    windows = build_windows(trials, spec)

    # The champion's own drop list unless an ablation overrides it on the command line
    champion = load_spec()
    dropped = args.drop if args.drop is not None else champion["drop_features"]
    feats = select_features(feature_columns(windows), dropped)

    train_df = trainable(windows, "train")
    assert "lockbox" not in set(train_df["split"]), "lockbox leaked into training"
    revs = sorted(train_df["rev"].unique())
    print(f"[s2] window={spec.window_s}s  train windows={len(train_df):,}  "
          f"features={len(feats)}  revs={len(revs)} {revs}")

    # `--drop` is an experiment, not a refit, so say the declaration cannot hold
    is_experiment = args.drop is not None
    if is_experiment:
        print(f"[s2] NOTE: experiment, not the champion "
              f"(--drop {len(dropped)} feature(s): {' '.join(dropped)})")
    else:
        assert_matches_spec(champion, build_model(), spec, feats, dropped)
        print(f"[s2] champion spec '{champion['name']}' matches the code")

    oof = cross_validate(train_df, feats)
    y = train_df["label"].to_numpy(int)
    result = evaluate(y, np.where(oof >= 0.5, WALK, STAND),
                      groups=train_df["rev"].to_numpy(),
                      unknown_frac=train_df["unknown_frac"].to_numpy())
    curve = selective_curve(y, oof, groups=train_df["rev"].to_numpy())

    print(f"[s2] leave-one-rev-out macro-F1 = {result.macro_f1:.4f}  "
          f"(acc {result.accuracy:.4f}, balanced {result.balanced_accuracy:.4f})")
    for rev, f1 in sorted(result.per_rev_macro_f1.items()):
        acc = result.per_rev_accuracy[rev]
        print(f"[s2]   held-out {rev}: macro-F1 {f1:.4f}  acc {acc * 100:.2f}%")
    print("[s2] window-level selective accuracy:")
    for row in curve:
        if row["threshold"] in tuple(PRESETS.values()):
            print(f"[s2]   thr {row['threshold']:.2f}: coverage {row['coverage']:.4f}  "
                  f"selective_acc {row['selective_accuracy']:.4f}  "
                  f"worst_rev {row['worst_rev_accuracy']:.4f} "
                  f"({row.get('worst_rev') or '?'})")

    # Champion: refit on every training rev; the lockbox stays sealed
    model = build_model()
    model.fit(train_df[feats].to_numpy(float), y)
    importances = sorted(zip(feats, model.feature_importances_), key=lambda x: -x[1])

    save(result, out_dir / "locoeval.json", curve=curve)
    if args.report:
        (out_dir / "locoeval.md").write_text(
            render(result, curve, title="S2 champion - leave-one-rev-out CV")
            + "\n\n## Feature importance (top 12)\n\n"
            + "\n".join(f"- `{n}`: {v:.4f}" for n, v in importances[:12]) + "\n",
            encoding="utf-8",
        )
    (out_dir / "model_meta.json").write_text(json.dumps({
        "champion_spec": champion["name"],
        "dropped_features": sorted(dropped),
        "model": "ExtraTreesClassifier",
        "params": dict(MODEL_PARAMS),
        "window_s": spec.window_s, "stride_s": spec.stride_s, "fs_hz": spec.fs_hz,
        "inference_stride_s": DEFAULT_INFERENCE_STRIDE_S,
        "features": feats,
        # every feature, not locoeval.md's top 12: the spec's ablation argues from ranks that deep
        "feature_importance": [[n, float(v)] for n, v in importances],
        "train_revs": revs,
        "n_train_windows": int(len(train_df)),
        "cv": "LeaveOneGroupOut(rev)",
        "macro_f1_cv": result.macro_f1,
        "presets": PRESETS,
        "default_preset": DEFAULT_PRESET,
        "reference_stats": reference_stats(train_df, feats),
    }, indent=2), encoding="utf-8")

    # only a refit persists the estimator, because only a refit has a reader; rerun to get one back
    if is_experiment:
        print("[s2] experiment: writing metrics only, no champion.joblib (nothing loads it)")
    else:
        try:
            import joblib
            joblib.dump(model, out_dir / "champion.joblib")
        except Exception as exc:  # metrics are the deliverable; the artifact is convenience
            print(f"[s2] WARNING: could not persist model ({exc})")

    # the spec can change without anyone touching data, so between a promotion and a refit all this lies
    stamp_inputs(out_dir, {"champion_spec": CHAMPION_SPEC_PATH}, stage="train")

    # This stage used to write the bare `_inputs.json` here, before stamps were split per stage.
    # Left behind it becomes a stamp with no writer: nothing refreshes it, so the next spec change
    # makes it complain about a report that was in fact rebuilt. Superseded, so removed by the
    # stage that owned it -- the same reasoning that deleted `s3_physics/physics.md` (RUNBOOK §7).
    (out_dir / "_inputs.json").unlink(missing_ok=True)

    print(f"[s2] artifacts -> {out_dir}")


if __name__ == "__main__":
    main()
