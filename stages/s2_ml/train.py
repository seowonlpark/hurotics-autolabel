# S2 train: fit a model, score it honestly, write the artifacts
# deterministic core of the champion/challenger loop (PLAN S2); reports numbers, never
# decides "best". enforces in code: the lockbox is never touched, and validation is
# leave-one-rev-out (one subject/day held out = the deployment question). see README.
#
# the persisted artifacts (champion.joblib, model_meta.json) honor the CHAMPION SPEC by
# default: it loads stages/s2_ml/champion_spec.json (the git-tracked seed every promotion
# rewrites) and applies its drop_features / window_s / model_params before fitting, so the
# saved model is the real champion, not the full-feature baseline. --full ignores the spec
# and trains on every feature (the baseline, for comparison); --spec PATH points elsewhere.

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from stages.s2_ml.dataset import load_dataset
from stages.s2_ml.experiment import (
    CHAMPION_SPEC_PATH,
    ExperimentSpec,
    champion_config,
    load_champion_spec,
    select_features,
)
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


# fresh model at the fixed params, or a spec's resolved overrides. default keeps the baseline
# params, so oof.py / lockbox.py (which call build_model()) are unaffected.
def build_model(params: dict | None = None) -> RandomForestClassifier:
    return RandomForestClassifier(**(params or MODEL_PARAMS))


# label-pure windows of one split; transitions excluded from targets (Section 5.2)
def trainable(df: pd.DataFrame, split: str = "train") -> pd.DataFrame:
    return df[(df["split"] == split) & (df["label"] != TRANSITION)].reset_index(drop=True)


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


# one leave-one-rev-out pass -> (y_true, y_pred, taxonomy|None). the fold that holds a rev
# out scores that rev's windows (OOF) and, when taxonomy is asked, its rows via dense
# inference -- one model per rev, used for both, so the identical LORO forests are never
# refit a second time. a rev's rows are only ever scored by a model that never saw that rev.
def loro(trials, train_df: pd.DataFrame, feats: list[str], spec: WindowSpec,
         stride_s: float, taxonomy: bool,
         params: dict | None = None) -> tuple[np.ndarray, np.ndarray, dict | None]:
    X = train_df[feats].to_numpy(float)
    y = train_df["label"].to_numpy(int)
    groups = train_df["rev"].to_numpy()

    oof = np.empty_like(y)
    per_run: list = []
    for rev in sorted(train_df["rev"].unique()):
        te = groups == rev
        model = build_model(params)
        model.fit(X[~te], y[~te])
        oof[te] = model.predict(X[te])
        if not taxonomy:
            continue
        for tr in trials:
            if tr.split != "train" or tr.rev != rev:
                continue
            for gt, pred, t in dense_predict_trial(model, tr.frame, feats, spec, stride_s):
                per_run.append(bucket_errors(gt, pred, t))
    return y, oof, (aggregate(per_run) if taxonomy else None)


# train + LORO-score at the given window, optionally the taxonomy, write all artifacts
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/s2_ml")
    ap.add_argument("--window-s", type=float, default=None,
                    help="window length in seconds; overrides the spec window only, stride "
                         "unchanged (Section 9 open tradeoff)")
    ap.add_argument("--taxonomy", action="store_true",
                    help="also run the row-level error taxonomy via dense inference (slow)")
    ap.add_argument("--stride-s", type=float, default=DEFAULT_INFERENCE_STRIDE_S,
                    help="dense inference stride in seconds")
    ap.add_argument("--spec", default=str(CHAMPION_SPEC_PATH),
                    help="champion spec JSON to honor (drop_features/window_s/model_params); "
                         "defaults to the git-tracked champion_spec.json")
    ap.add_argument("--full", action="store_true",
                    help="ignore the spec and train on every feature (the baseline)")
    ap.add_argument("--skip-if-current", action="store_true",
                    help="skip the (re)fit when champion.joblib/model_meta already reflect this "
                         "exact spec/window/stride -- the post-cycle refit is then a no-op when "
                         "nothing was promoted (avoids a redundant LORO CV + dense taxonomy)")
    args = ap.parse_args()

    out_dir = (REPO_ROOT / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # the champion spec drives which features/window/params the persisted model uses (the SAME
    # champion_config resolution the serve path uses, so the joblib matches the OOF/lockbox refit);
    # --full opts out to the all-feature baseline. an explicit --window-s overrides the spec window.
    champ_spec: ExperimentSpec | None = None if args.full else load_champion_spec(Path(args.spec))
    if champ_spec:
        spec, params, drops = champion_config(champ_spec)
    else:
        spec, params, drops = WindowSpec(), dict(MODEL_PARAMS), []
    if args.window_s:  # override the window ONLY; stride is independent (its own Section 9
        spec = replace(spec, window_s=args.window_s)  # tradeoff), so it never rides on window_s

    # a refit after a cycle that promoted nothing is redundant work: the persisted model_meta +
    # joblib already describe this exact champion. skip the whole LORO CV + dense taxonomy then.
    if args.skip_if_current and champ_spec is not None:
        meta = out_dir / "model_meta.json"
        if (out_dir / "champion.joblib").exists() and meta.exists():
            prev = json.loads(meta.read_text(encoding="utf-8"))
            if (prev.get("spec"), prev.get("window_s"), prev.get("stride_s")) == \
                    (champ_spec.name, spec.window_s, spec.stride_s):
                print(f"[s2] champion '{champ_spec.name}' artifacts already current; skipping refit")
                return

    trials = load_dataset()
    windows = build_windows(trials, spec)
    feats = select_features(feature_columns(windows), drops)

    train_df = trainable(windows, "train")
    assert "lockbox" not in set(train_df["split"]), "lockbox leaked into training"
    revs = sorted(train_df["rev"].unique())
    tag = f"champion '{champ_spec.name}'" if champ_spec else "baseline (all features)"
    print(f"[s2] {tag}: window={spec.window_s}s  train windows={len(train_df):,}  "
          f"features={len(feats)} ({len(drops)} dropped)  revs={len(revs)} {revs}")

    if args.taxonomy:
        print(f"[s2] dense inference @ {args.stride_s * 1000:.0f} ms for the row-level taxonomy...")
    y, oof, tax = loro(trials, train_df, feats, spec, args.stride_s, args.taxonomy, params)
    result = evaluate(y, oof, groups=train_df["rev"].to_numpy(),
                      unknown_frac=train_df["unknown_frac"].to_numpy())
    trans = transition_report(train_df, oof)

    print(f"[s2] leave-one-rev-out macro-F1 = {result.macro_f1:.4f}  "
          f"(acc {result.accuracy:.4f}, balanced {result.balanced_accuracy:.4f})")
    for rev, f1 in sorted(result.per_rev_macro_f1.items()):
        print(f"[s2]   held-out {rev}: macro-F1 {f1:.4f}")

    # final model: refit on every training rev; the lockbox stays sealed
    model = build_model(params)
    model.fit(train_df[feats].to_numpy(float), train_df["label"].to_numpy(int))

    importances = sorted(zip(feats, model.feature_importances_), key=lambda x: -x[1])

    if tax:
        print(f"[s2] row accuracy {tax['row_accuracy']:.4f}, dominant error: {tax['dominant']}")
        for b in ERROR_BUCKETS:
            if tax["counts"][b]:
                print(f"[s2]   {b:<16} {tax['counts'][b]:>8,}  ({tax['fractions'][b]:.3f})")

    save(result, out_dir / "locoeval.json", trans)
    title = (f"S2 champion ({champ_spec.name}) - leave-one-rev-out CV" if champ_spec
             else "S2 baseline (all features) - leave-one-rev-out CV")
    body = render(result, trans, title=title)
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
        "spec": champ_spec.name if champ_spec else "baseline_all_features",
        "drop_features": drops,
        "params": params,
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
