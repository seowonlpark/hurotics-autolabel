# reconcile our champion against the incumbent `loco` on a COMMON footing. most loco pairs are the
# sealed lockbox, so the full comparison is deferred to lockbox-open. what's scorable now: both
# predictors on the one shared non-lockbox recording, same rows and denominator. directional, not a finding.

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut  # noqa: F401  (kept for parity/reference)

from stages.s2_ml.dataset import (
    DEFAULT_LOCKBOX_REVS, LABEL_COL, TIME_COL, load_dataset, load_trial, rev_of,
)
from stages.s2_ml.experiment import BASE_MODEL_PARAMS, load_champion, select_features
from stages.s2_ml.features import TRANSITION, WindowSpec, build_windows, feature_columns
from stages.s2_ml.predict import dense_predict_segment, labeled_runs
from stages.s2_ml.profile_incumbent import LOCO_COL, best_mapping, find_pairs
from stages.s2_ml.taxonomy import ERROR_BUCKETS, aggregate, bucket_errors

REPO_ROOT = Path(__file__).resolve().parents[2]
S2_RUN_DIR = REPO_ROOT / "runs" / "s2_ml"


# nearest-in-time categorical resample (the clean layer's label rule); puts `loco` on the
# same canonical grid our model predicts on, so both are scored on identical rows
def nearest_sample(src_t: np.ndarray, src_v: np.ndarray, grid_t: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(src_t, grid_t).clip(1, src_t.size - 1)
    left = np.abs(grid_t - src_t[idx - 1]) <= np.abs(src_t[idx] - grid_t)
    return src_v[np.where(left, idx - 1, idx)]


# champion model with held_out_rev excluded -- the honest 'unseen rev' predictor for that
# recording (same discipline as leave-one-rev-out)
def fit_heldout_model(trials, feats, wspec, held_out_rev: str, params: dict):
    windows = build_windows(trials, wspec)
    train_df = windows[(windows["split"] == "train") &
                       (windows["rev"] != held_out_rev) &
                       (windows["label"] != TRANSITION)].reset_index(drop=True)
    model = RandomForestClassifier(**params)
    model.fit(train_df[feats].to_numpy(float), train_df["label"].to_numpy(int))
    return model


# bucket one trial's rows given a full per-row prediction aligned to frame; both predictors
# go through this on the same frame, so the common denominator is guaranteed
def score_frame(frame: pd.DataFrame, pred_full: np.ndarray) -> list[dict]:
    per_run = []
    base = 0
    for _seg_id, seg in frame.groupby("segment", sort=True):
        n = len(seg)
        gt = seg[LABEL_COL].to_numpy()
        t = seg[TIME_COL].to_numpy(float)
        pred = pred_full[base:base + n]
        base += n
        for g, pr, tt in labeled_runs(gt, pred, t):
            per_run.append(bucket_errors(g, pr, tt))
    return per_run


# dense per-row predictions aligned to frame, segment by segment
def our_predictions(model, frame: pd.DataFrame, feats, wspec) -> np.ndarray:
    out = np.empty(len(frame), dtype=int)
    base = 0
    for _seg_id, seg in frame.groupby("segment", sort=True):
        seg = seg.reset_index(drop=True)
        out[base:base + len(seg)] = dense_predict_segment(model, seg, feats, wspec)
        base += len(seg)
    return out


# each bucket as a fraction of ALL scored rows (the common denominator), unlike
# agg['fractions'] which is a share of that model's own errors and hides how many there are
def _rates(agg: dict) -> dict:
    total = agg["correct_rows"] + agg["total_error_rows"]
    return {b: (agg["counts"][b] / total if total else 0.0) for b in ERROR_BUCKETS}


# score both predictors on the shared non-lockbox recordings, report common-denominator rates
def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    champion = load_champion(S2_RUN_DIR)
    if champion is None:
        raise SystemExit("no champion.json to reconcile against")
    spec = champion["spec"]
    params = {**BASE_MODEL_PARAMS, **(spec.get("model_params") or {})}
    drop = spec.get("drop_features") or []

    pairs = find_pairs()
    train_pairs = [(p, raw) for p, raw in pairs if rev_of(p) not in DEFAULT_LOCKBOX_REVS]
    lock_pairs = [(p, raw) for p, raw in pairs if rev_of(p) in DEFAULT_LOCKBOX_REVS]

    print("=" * 72)
    print("INCUMBENT RECONCILIATION")
    print("=" * 72)
    print(f"loco-paired recordings: {len(pairs)} total - "
          f"{len(train_pairs)} train-rev, {len(lock_pairs)} LOCKBOX-rev "
          f"({sorted({rev_of(p) for p, _ in lock_pairs})}).")
    print("The published loco profile is dominated by LOCKBOX recordings; our champion is\n"
          "scored on train revs. They are NOT the same recordings - the naive\n"
          "0.888-vs-0.519 comparison is cross-recording. A like-for-like number on rev13\n"
          "requires scoring our model on the lockbox, which opens exactly once at the end.\n"
          f"Shared, scorable-now recordings: {[p.name for p, _ in train_pairs]}\n")

    if not train_pairs:
        raise SystemExit("no shared non-lockbox recording - full reconciliation is "
                         "deferred to lockbox-open. Nothing to score now.")

    # generous loco decode, fit on the shared recordings only (best case for the incumbent)
    mapping = best_mapping(train_pairs)

    trials = load_dataset()
    wspec = WindowSpec()
    feats = select_features(feature_columns(build_windows(trials, wspec)), drop)

    ours_runs, loco_runs = [], []
    per_recording = []
    for p, raw in train_pairs:
        rev = rev_of(p)
        trial = load_trial(p, "train")
        frame = trial.frame

        model = fit_heldout_model(trials, feats, wspec, rev, params)
        our_pred = our_predictions(model, frame, feats, wspec)

        raw_t = raw[TIME_COL].to_numpy(float)
        raw_loco_class = np.array([mapping[int(v)] for v in raw[LOCO_COL].to_numpy()])
        loco_pred = nearest_sample(raw_t, raw_loco_class, frame[TIME_COL].to_numpy(float))

        r_ours = score_frame(frame, our_pred)
        r_loco = score_frame(frame, loco_pred)
        ours_runs += r_ours
        loco_runs += r_loco
        per_recording.append({
            "recording": p.name, "rev": rev,
            "ours": aggregate(r_ours), "loco": aggregate(r_loco),
        })

    ours, loco = aggregate(ours_runs), aggregate(loco_runs)
    ours_rate, loco_rate = _rates(ours), _rates(loco)

    # print helper: one predictor's buckets, both denominators side by side
    def show(title, agg, rate):
        print(f"\n{title}")
        print(f"  row_accuracy      {agg['row_accuracy']:.4f}   "
              f"({agg['correct_rows']:,} correct / {agg['total_error_rows']:,} errors)")
        print(f"  {'bucket':<18}{'rows':>8}{'/all_rows':>12}{'/errors':>10}")
        for b in ERROR_BUCKETS:
            print(f"  {b:<18}{agg['counts'][b]:>8,}{rate[b]:>12.4f}"
                  f"{agg['fractions'][b]:>10.3f}")

    print("-" * 72)
    print(f"SAME-ROWS COMPARISON on {len(train_pairs)} shared recording(s), "
          f"identical canonical 100 Hz grid:")
    show("OUR CHAMPION (rev held out):", ours, ours_rate)
    show("INCUMBENT loco (generous decode):", loco, loco_rate)

    sc_ours, sc_loco = ours_rate["steady_confusion"], loco_rate["steady_confusion"]
    print("\n" + "-" * 72)
    print("steady_confusion as a fraction of ALL scored rows (the hazardous bucket, Section 7):")
    print(f"  ours {sc_ours:.4f}   loco {sc_loco:.4f}   "
          f"(ours {'higher' if sc_ours > sc_loco else 'lower'} by "
          f"{abs(sc_ours - sc_loco):.4f})")
    print("\nDIRECTIONAL ONLY - one shared non-lockbox recording. NOT a finding. The\n"
          "share-of-errors framing (0.888 vs 0.519) compared different recordings on\n"
          "different denominators; on the same rows the number to watch is the rate\n"
          "above. Full comparison waits for lockbox-open (rev13, where 7/8 loco pairs live).")

    out = S2_RUN_DIR / "incumbent_reconciliation.json"
    out.write_text(json.dumps({
        "n_pairs": len(pairs),
        "n_train_pairs": len(train_pairs),
        "lockbox_pairs": [p.name for p, _ in lock_pairs],
        "shared_recordings": [p.name for p, _ in train_pairs],
        "mapping": {str(k): v for k, v in mapping.items()},
        "ours": {**ours, "rate_over_all_rows": ours_rate},
        "loco": {**loco, "rate_over_all_rows": loco_rate},
        "per_recording": [
            {"recording": r["recording"], "rev": r["rev"],
             "ours_row_accuracy": r["ours"]["row_accuracy"],
             "loco_row_accuracy": r["loco"]["row_accuracy"],
             "ours_steady_confusion_rows": r["ours"]["counts"]["steady_confusion"],
             "loco_steady_confusion_rows": r["loco"]["counts"]["steady_confusion"]}
            for r in per_recording
        ],
        "caveat": "directional only; one shared non-lockbox recording; full comparison "
                  "deferred to lockbox-open (rev13).",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
