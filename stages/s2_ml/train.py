"""S2 train: fit the champion, score it honestly, write the artifacts.

    python -m stages.s2_ml.train

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

from freshness import stamp_inputs
from stages.s2_ml.dataset import STAND, TRAIN_CLASSES, WALK, load_dataset
from stages.s2_ml.features import (
    WindowSpec,
    amplitude_band,
    build_windows,
    feature_columns,
)
from stages.s2_ml.locoeval import evaluate, render, save, selective_curve

REPO_ROOT = Path(__file__).resolve().parents[2]

# The champion, DECLARED. `runs/` is gitignored, so before this file existed the only
# statement of what the champion is was whatever this module happened to fit — readable by
# running it, not by reading anything. A stale `runs/s2_ml/champion.json` inherited from the
# sibling repo described an 18-feature RandomForest that this code has never built, and
# nothing could contradict it. The spec is git-tracked so the claim travels with the code,
# and `assert_matches_spec` makes the two disagree loudly instead of silently.
CHAMPION_SPEC_PATH = Path(__file__).resolve().parent / "champion_spec.json"


def load_spec(path: Path = CHAMPION_SPEC_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# The champion's hyperparameters, READ from the spec rather than restated here. They were a
# literal until `experiment.py` came back: a promoted challenger rewrites the spec, so a
# second copy in code would have made every promotion trip `assert_matches_spec` and stop
# the pipeline until a human retyped the same numbers. Two sources are only checkable while
# nothing is allowed to change either one.
#
# What is NOT weakened by that: the estimator class, window grid and feature count below are
# still built by this file and still checked against the spec, because those live in code
# that a promotion does not touch.
MODEL_PARAMS = load_spec()["params"]

# Inference slides the same window at a fraction of its length, so a row's probability is
# an average over several overlapping views. Training stays non-overlapping (§9).
DEFAULT_INFERENCE_STRIDE_S = 0.25

# Named operating points; OPERATING_POINTS.md carries the measured tradeoff behind each.
# `balanced` is the default because it is the lowest threshold at which every held-out
# subject independently clears 95% accuracy on the rows it commits to.
PRESETS = {"high_coverage": 0.70, "balanced": 0.85, "high_precision": 0.95}
DEFAULT_PRESET = "balanced"


def build_model(params: dict | None = None) -> ExtraTreesClassifier:
    """The champion estimator, or a challenger's variant of it.

    `params` exists for `experiment.py`, which fits the same class under proposed
    hyperparameters. The CLASS is not a parameter: a proposal may retune the champion, not
    replace it with a different model, and swapping the estimator is a decision to argue in
    `champion_spec.json` rather than to make from a JSON field.
    """
    return ExtraTreesClassifier(**(params or MODEL_PARAMS))


def select_features(all_feats: list[str], drop: list[str]) -> list[str]:
    """The feature set minus `drop`, refusing to drop a name that is not there.

    An unknown name is a hard error rather than a no-op, because the failure it prevents is
    silent: a typo'd or renamed feature would leave the full set trained, scored, and
    recorded as the ablation it is not — and the ablation would look like it changed
    nothing, which is exactly the answer a typo produces.
    """
    if (unknown := [f for f in drop if f not in all_feats]):
        raise SystemExit(
            f"[s2] cannot drop {unknown}: not in the {len(all_feats)} features this corpus "
            f"builds. Check the spelling against runs/s2_ml/model_meta.json."
        )
    return [f for f in all_feats if f not in drop]


def assert_matches_spec(spec: dict, model, window: WindowSpec, feats: list[str],
                        dropped: list[str]) -> None:
    """Refuse to fit a champion the spec does not describe.

    This is a DECLARATION check, not a quality gate: it says nothing about whether the
    model is good, only that the thing about to be written as `champion.joblib` is the
    thing `champion_spec.json` claims. Both halves are cheap to change — the point is that
    changing one alone stops the run, so a drifted champion cannot be produced quietly and
    then be quoted from a report that looks exactly like a correct one.

    The feature COUNT is checked rather than the names: the names are derived from the
    corpus, so pinning them would fail on any new session, while a silent change in how
    many features exist is precisely the drift worth catching.

    `params` is deliberately absent from the comparison — `MODEL_PARAMS` is now read from
    this same spec, so checking it would compare the file against itself and always pass.
    """
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
        # ASCII only: this message goes to a cp949 console as an uncaught SystemExit, and an
        # em-dash there raises UnicodeEncodeError — replacing a clear refusal with a
        # traceback about encoding. Same reason the .md files keep their typography and the
        # console does not.
        raise SystemExit(
            f"[s2] champion drift - {CHAMPION_SPEC_PATH.name} does not describe this code:\n"
            + "\n".join(bad)
            + "\n[s2] Fix ONE of them on purpose: revert the code, or update the spec and "
              "re-run to record what the new champion actually scores."
        )


def trainable(df: pd.DataFrame, split: str = "train") -> pd.DataFrame:
    """Label-pure windows of one split. Transitions are excluded from targets (§5.2).

    Selected positively, on membership of the trained classes. Excluding TRANSITION and
    None by name instead lets any third label state through silently, which is how 84
    all-unknown windows reached `to_numpy(int)` and crashed it -- a loud failure that
    would have been a quiet contamination had the codes been numeric.
    """
    return df[(df["split"] == split)
              & df["label"].isin(TRAIN_CLASSES)].reset_index(drop=True)


def cross_validate(df: pd.DataFrame, feats: list[str],
                   params: dict | None = None) -> np.ndarray:
    """Leave-one-rev-out out-of-fold P(walk); every window scored by a model blind to its rev.

    `experiment.py` scores challengers through this function rather than through a copy of
    it. That is the whole basis of the promotion gate: a challenger's macro-F1 is comparable
    to the champion's only if the two were produced by the same folds, the same threshold and
    the same code, and a second implementation is a second set of ways to differ.
    """
    X = df[feats].to_numpy(float)
    y = df["label"].to_numpy(int)
    oof = np.zeros(len(y))
    for tr, te in LeaveOneGroupOut().split(X, y, df["rev"].to_numpy()):
        model = build_model(params)
        model.fit(X[tr], y[tr])
        oof[te] = model.predict_proba(X[te])[:, list(model.classes_).index(WALK)]
    return oof


def reference_stats(df: pd.DataFrame, feats: list[str]) -> dict:
    """Distributional constants the ambiguity reasons are stated against.

    Measured on training windows only. `label.py` reads these instead of hard-coding
    thresholds, so the explanations move with the data rather than with an author's memory.
    """
    walk = df[df["label"] == WALK]
    stand = df[df["label"] == STAND]
    # How far BOTH thighs sit above this recording's own standing posture. Upright standing
    # and level walking both keep it near zero; the standing class carries a long tail that
    # walking does not, and those windows concentrate in the trials whose `stand` runs cover
    # sitting and transfers. It is therefore a posture signature for a state outside this
    # two-class taxonomy, not a stand-vs-walk discriminator.
    posture = np.minimum(df["L_ang_med_rest"], df["R_ang_med_rest"])
    band_lo, band_hi = amplitude_band(df["ileg_minhalf"], df["label"])
    return {
        # A window predicted walk whose interleg excursion sits below where real walking
        # lives is either very slow gait or standing sway. Either way, worth flagging.
        "walk_minhalf_p05": float(np.percentile(walk["ileg_minhalf"], 5)),
        "posture_shift_p99": float(np.percentile(posture, 99)),
        # The ambiguity band, recorded here so `label.py` reads it off the champion rather
        # than recomputing it at serve time. ONE definition, in `features.amplitude_band`
        # — a second copy would drift, and a band the serve path derived for itself is
        # train/serve skew in the one place the model is allowed to refuse an answer.
        "band_lo": band_lo,
        "band_hi": band_hi,
        # Per-feature training range, for the out-of-distribution check.
        "feat_p01": {c: float(np.percentile(df[c], 1)) for c in feats},
        "feat_p99": {c: float(np.percentile(df[c], 99)) for c in feats},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/s2_ml")
    ap.add_argument("--window-s", type=float, default=None)
    ap.add_argument("--drop", nargs="+", metavar="FEATURE", default=None,
                    help="ablation: train without these features instead of the spec's "
                         "drop_features. Marks the run an experiment, not the champion.")
    args = ap.parse_args()

    out_dir = (REPO_ROOT / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    spec = (WindowSpec(window_s=args.window_s, stride_s=args.window_s)
            if args.window_s else WindowSpec())
    trials = load_dataset()
    windows = build_windows(trials, spec)

    # The champion's own drop list unless an ablation overrides it on the command line.
    champion = load_spec()
    dropped = args.drop if args.drop is not None else champion["drop_features"]
    feats = select_features(feature_columns(windows), dropped)

    train_df = trainable(windows, "train")
    assert "lockbox" not in set(train_df["split"]), "lockbox leaked into training"
    revs = sorted(train_df["rev"].unique())
    print(f"[s2] window={spec.window_s}s  train windows={len(train_df):,}  "
          f"features={len(feats)}  revs={len(revs)} {revs}")

    # `--window-s` and `--drop` are experiments, not champion refits, so the declaration
    # cannot hold — say so rather than either failing a legitimate sweep or letting it pass
    # unremarked. The artifacts still land, which is the point: an ablation is only worth
    # anything if its locoeval can be read next to the champion's.
    if args.window_s or args.drop is not None:
        why = []
        if args.window_s:
            why.append(f"--window-s {args.window_s} over spec {champion['window_s']}")
        if args.drop is not None:
            why.append(f"--drop {len(dropped)} feature(s): {' '.join(dropped)}")
        print(f"[s2] NOTE: experiment, not the champion ({'; '.join(why)})")
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
        "champion_spec": champion["name"],
        "dropped_features": sorted(dropped),
        "model": "ExtraTreesClassifier",
        "params": dict(MODEL_PARAMS),
        "window_s": spec.window_s, "stride_s": spec.stride_s, "fs_hz": spec.fs_hz,
        "inference_stride_s": DEFAULT_INFERENCE_STRIDE_S,
        "features": feats,
        # Also here, not only in locoeval.md's top-12 list: the ablation rationale in
        # `champion_spec.json` argues from importance RANKS, and a reader who wants to
        # re-check that argument had to re-fit the model to see rank 39. Every feature,
        # machine-readable, next to the feature list it orders.
        "feature_importance": [[n, float(v)] for n, v in importances],
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

    # What everything in this directory describes. The corpus is this stage's other input, but
    # `champion_spec.json` is the one that can change WITHOUT anyone touching data — a promotion
    # rewrites it, and until this refits, `locoeval`, `model_meta` and `champion.joblib` all
    # describe the model that just lost. The window between those two events is exactly the
    # silence `freshness.py` exists for, so it is stamped rather than assumed to be brief.
    stamp_inputs(out_dir, {"champion_spec": CHAMPION_SPEC_PATH})

    print(f"[s2] artifacts -> {out_dir}")


if __name__ == "__main__":
    main()
