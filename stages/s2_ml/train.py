# S2 train: fit a model, score it honestly, write the artifacts
# deterministic core of the champion/challenger loop (PLAN S2); reports numbers, never
# decides "best". enforces in code: the lockbox is never touched, and validation is
# leave-one-rev-out (one subject/day held out = the deployment question). see README.

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
from stages.s2_ml.predict import DEFAULT_INFERENCE_STRIDE_S, dense_predict_trial
from stages.s2_ml.taxonomy import (
    ERROR_BUCKETS,
    FLICKER_MAX_MS,
    LAG_MAX_MS,
    aggregate,
    bucket_errors,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

# Random Forest is the documented starting model (Section 9); depth unbounded, tuning belongs in
# the champion/challenger loop, not a hand-picked constant
MODEL_PARAMS = dict(n_estimators=300, random_state=0, n_jobs=-1, class_weight="balanced")


# fresh model at the fixed params
def build_model() -> RandomForestClassifier:
    return RandomForestClassifier(**MODEL_PARAMS)


# label-pure windows of one split; transitions excluded from targets (Section 5.2)
def trainable(df: pd.DataFrame, split: str = "train") -> pd.DataFrame:
    return df[(df["split"] == split) & (df["label"] != TRANSITION)].reset_index(drop=True)


# leave-one-rev-out error taxonomy, scored at row level via dense inference; a rev's rows
# are only ever scored by a model that never saw that rev
def taxonomy_loro(trials, train_df: pd.DataFrame, feats: list[str], spec: WindowSpec,
                  stride_s: float) -> dict:
    per_run = []
    for rev in sorted(train_df["rev"].unique()):
        fit = train_df[train_df["rev"] != rev]
        model = build_model()
        model.fit(fit[feats].to_numpy(float), fit["label"].to_numpy(int))
        for tr in trials:
            if tr.split != "train" or tr.rev != rev:
                continue
            for gt, pred, t in dense_predict_trial(model, tr.frame, feats, spec, stride_s):
                per_run.append(bucket_errors(gt, pred, t))
    return aggregate(per_run)


# render the row-level taxonomy section as markdown
def render_taxonomy(agg: dict, stride_s: float) -> str:
    lines = [
        "## Error taxonomy (row-level, leave-one-rev-out)", "",
        f"Ported from `hurotics-locotool/locoeval/diagnose.py`; thresholds unchanged "
        f"(flicker < {FLICKER_MAX_MS:g} ms, lag < {LAG_MAX_MS:g} ms). Scored on dense "
        f"inference at {stride_s * 1000:.0f} ms so the thresholds are resolvable.", "",
        f"- row accuracy: **{agg['row_accuracy']:.4f}**  "
        f"({agg['correct_rows']:,} correct / {agg['total_error_rows']:,} error rows)",
        f"- dominant error bucket: **{agg['dominant']}**", "",
        "| bucket | rows | share of errors |", "|---|---|---|",
    ]
    for b in ERROR_BUCKETS:
        lines.append(f"| `{b}` | {agg['counts'][b]:,} | {agg['fractions'][b]:.3f} |")
    return "\n".join(lines)


# leave-one-rev-out out-of-fold predictions; returns (y_true, y_pred)
def cross_validate(df: pd.DataFrame, feats: list[str]) -> tuple[np.ndarray, np.ndarray]:
    X = df[feats].to_numpy(float)
    y = df["label"].to_numpy(int)
    groups = df["rev"].to_numpy()

    oof = np.empty_like(y)
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        model = build_model()
        model.fit(X[tr], y[tr])
        oof[te] = model.predict(X[te])
    return y, oof


# train + LORO-score at the given window, optionally the taxonomy, write all artifacts
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/s2_ml")
    ap.add_argument("--window-s", type=float, default=None,
                    help="window length in seconds (Section 9 open tradeoff)")
    ap.add_argument("--taxonomy", action="store_true",
                    help="also run the row-level error taxonomy via dense inference (slow)")
    ap.add_argument("--stride-s", type=float, default=DEFAULT_INFERENCE_STRIDE_S,
                    help="dense inference stride in seconds")
    args = ap.parse_args()

    out_dir = (REPO_ROOT / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    spec = WindowSpec(window_s=args.window_s, stride_s=args.window_s) if args.window_s else WindowSpec()
    trials = load_dataset()
    windows = build_windows(trials, spec)
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

    # final model: refit on every training rev; the lockbox stays sealed
    model = build_model()
    model.fit(train_df[feats].to_numpy(float), train_df["label"].to_numpy(int))

    importances = sorted(zip(feats, model.feature_importances_), key=lambda x: -x[1])

    tax = None
    if args.taxonomy:
        print(f"[s2] dense inference @ {args.stride_s * 1000:.0f} ms for the row-level taxonomy...")
        tax = taxonomy_loro(trials, train_df, feats, spec, args.stride_s)
        print(f"[s2] row accuracy {tax['row_accuracy']:.4f}, dominant error: {tax['dominant']}")
        for b in ERROR_BUCKETS:
            if tax["counts"][b]:
                print(f"[s2]   {b:<16} {tax['counts'][b]:>8,}  ({tax['fractions'][b]:.3f})")

    save(result, out_dir / "locoeval.json", trans)
    body = render(result, trans, title="S2 champion - leave-one-rev-out CV")
    if tax:
        body += "\n\n" + render_taxonomy(tax, args.stride_s)
    (out_dir / "locoeval.md").write_text(
        body + "\n\n## Feature importance (top 12)\n\n"
        + "\n".join(f"- `{n}`: {v:.4f}" for n, v in importances[:12]) + "\n",
        encoding="utf-8",
    )
    if tax:
        (out_dir / "taxonomy.json").write_text(json.dumps(tax, indent=2), encoding="utf-8")
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
    except Exception as exc: # model artifact is optional; metrics are not
        print(f"[s2] WARNING: could not persist model ({exc})")

    print(f"[s2] artifacts -> {out_dir}")


if __name__ == "__main__":
    main()
