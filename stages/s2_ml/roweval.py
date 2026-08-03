"""Row-level evaluation of the labelling path, including the abstention reasons.

    python -m stages.s2_ml.roweval --out runs/s2_ml          # leave-one-rev-out
    python -m stages.s2_ml.roweval --lockbox --out runs/s2_ml  # SINGLE USE

Why a second evaluator when `train.py` already reports CV numbers: `train.py` scores
WINDOWS, which is the unit the model learns on. A caller labels a CSV and gets ROWS, and
the two differ — rows are scored by averaging every window that covers them, which changes
both the accuracy and, more importantly, the confidence ordering the abstention threshold
is set from. Publishing window numbers and shipping row behaviour would be measuring one
thing and selling another.

This runs the real `label.py` path, not a reimplementation of it. Anything that drifts
between evaluation and deployment is train/serve skew, and the whole point of the module
is to be the number you can believe.

**The lockbox is single use.** rev8 and rev13 were sealed through feature selection, model
selection and threshold selection. `--lockbox` fits on every training rev and scores them
once. Running it repeatedly and picking the best result would convert the only honest
measurement in the repo into another validation set (§7).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from stages.s2_ml.dataset import HUMAN_UNKNOWN, LABEL_COL, STAND, TRAIN_CLASSES, WALK, load_dataset
from stages.s2_ml.features import WindowSpec, build_windows, feature_columns
from stages.s2_ml.label import explain, score_frame
from stages.s2_ml.locoeval import DEFAULT_THRESHOLDS
from stages.s2_ml.train import PRESETS, build_model, reference_stats, trainable

REPO_ROOT = Path(__file__).resolve().parents[2]


def fit_on(train_df: pd.DataFrame, feats: list[str], exclude_rev: str | None):
    """Champion fitted on every training rev except `exclude_rev`, with its own reference
    stats. The stats must come from the SAME subset as the model: computing them once over
    all revs would leak the held-out subject into the ambiguity explanations."""
    sub = train_df if exclude_rev is None else train_df[train_df["rev"] != exclude_rev]
    model = build_model()
    model.fit(sub[feats].to_numpy(float), sub["label"].to_numpy(int))
    return model, reference_stats(sub, feats)


def score_trials(model, meta: dict, trials, threshold: float) -> pd.DataFrame:
    """Run the shipping path over trials and join ground truth back on."""
    out = []
    for tr in trials:
        frame = tr.frame.reset_index(drop=True)
        if frame.empty:
            continue
        spec = WindowSpec(window_s=meta["window_s"],
                          stride_s=meta["inference_stride_s"], fs_hz=meta["fs_hz"])
        scored = explain(score_frame(model, meta, frame), meta, threshold, spec)
        scored = scored.sort_values(["segment", "Time"]).reset_index(drop=True)
        truth = (frame.sort_values(["segment", "Time"])
                      .reset_index(drop=True)[LABEL_COL].to_numpy())
        scored["truth"] = truth
        scored["rev"] = tr.rev
        scored["trial"] = tr.trial
        out.append(scored)
    return pd.concat(out, ignore_index=True)


def curve(df: pd.DataFrame) -> list[dict]:
    """Row-level coverage vs accuracy. Only rows with a trainable ground truth count (§7)."""
    v = df[df["truth"].isin(TRAIN_CLASSES) & df["p_walk"].notna()]
    p = v["p_walk"].to_numpy(float)
    conf = np.maximum(p, 1 - p)
    correct = (np.where(p >= 0.5, WALK, STAND) == v["truth"].to_numpy())
    revs = v["rev"].to_numpy()
    rows = []
    for thr in DEFAULT_THRESHOLDS:
        k = conf >= thr
        per = [correct[k & (revs == r)].mean() for r in pd.unique(revs) if (k & (revs == r)).any()]
        rows.append({
            "threshold": float(thr),
            "coverage": float(k.mean()),
            "n_labeled": int(k.sum()),
            "selective_accuracy": float(correct[k].mean()) if k.any() else float("nan"),
            "worst_rev_accuracy": float(min(per)) if per else float("nan"),
            "errors_kept": int((~correct[k]).sum()),
        })
    return rows


def reason_report(df: pd.DataFrame) -> pd.DataFrame:
    """Is each ambiguity reason earning its place?

    A reason is useful when the rows carrying it are measurably harder than the rows the
    model kept: lower accuracy for the guess it would have made, and a higher share of
    human `-1`. A reason that fires on rows the model would have got right is noise
    dressed as an explanation.
    """
    v = df[df["truth"].isin(TRAIN_CLASSES) & df["p_walk"].notna()].copy()
    v["would_be_correct"] = v["label"] == v["truth"]
    v["human_unknown_near"] = False
    rows = []
    conf_rows = v[~v["ambiguous"]]
    rows.append({"reason": "(confident, kept)", "rows": len(conf_rows),
                 "accuracy_of_guess": conf_rows["would_be_correct"].mean()})
    for r, g in v[v["ambiguous"]].groupby("reason"):
        rows.append({"reason": r, "rows": len(g),
                     "accuracy_of_guess": g["would_be_correct"].mean()})
    return pd.DataFrame(rows).sort_values("rows", ascending=False)


def unknown_agreement(df: pd.DataFrame) -> dict:
    """Does the model abstain where a trained human also could not call it (§5.2)?

    The model never sees `-1`, so agreement here is independent evidence that the
    confidence signal tracks genuine ambiguity rather than its own miscalibration.
    """
    v = df[df["p_walk"].notna()]
    unk = v["truth"] == HUMAN_UNKNOWN
    if not unk.any():
        return {}
    return {
        "rows_human_unknown": int(unk.sum()),
        "abstain_rate_on_human_unknown": float(v.loc[unk, "ambiguous"].mean()),
        "abstain_rate_elsewhere": float(v.loc[~unk, "ambiguous"].mean()),
    }


def render(tag: str, c: list[dict], reasons: pd.DataFrame, unk: dict, revs: list[str]) -> str:
    lines = [f"# Row-level evaluation - {tag}", "",
             f"subjects scored: {', '.join(revs)}", "",
             "| threshold | coverage | selective acc | worst subject | errors kept |",
             "|---|---|---|---|---|"]
    for r in c:
        lines.append(f"| {r['threshold']:.2f} | {r['coverage']:.4f} | "
                     f"{r['selective_accuracy']:.4f} | {r['worst_rev_accuracy']:.4f} | "
                     f"{r['errors_kept']:,} |")
    lines += ["", "## Ambiguity reasons", "",
              "`accuracy of guess` is how often the guess WOULD have been right had the row "
              "not been flagged. A reason earns its place by sitting well below the "
              "confident row.", "",
              "| reason | rows | accuracy of guess |", "|---|---|---|"]
    for _, r in reasons.iterrows():
        lines.append(f"| `{r['reason']}` | {r['rows']:,} | {r['accuracy_of_guess']:.4f} |")
    if unk:
        lines += ["", "## Agreement with human `-1` (§5.2)", "",
                  f"- rows a human marked unknown: {unk['rows_human_unknown']:,}",
                  f"- model abstains on those: **{unk['abstain_rate_on_human_unknown']:.1%}**",
                  f"- model abstains elsewhere: {unk['abstain_rate_elsewhere']:.1%}", "",
                  "The model never sees `-1` in training, so this is independent evidence."]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/s2_ml")
    ap.add_argument("--threshold", type=float, default=PRESETS["balanced"])
    ap.add_argument("--lockbox", action="store_true",
                    help="SINGLE USE: fit on all training revs, score the sealed revs")
    args = ap.parse_args()

    out_dir = (REPO_ROOT / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    trials = load_dataset()
    spec = WindowSpec()
    windows = build_windows([t for t in trials if t.split == "train"], spec)
    feats = feature_columns(windows)
    train_df = trainable(windows, "train")

    base_meta = {
        "window_s": spec.window_s, "fs_hz": spec.fs_hz, "inference_stride_s": 0.25,
        "features": feats,
    }

    parts = []
    if args.lockbox:
        model, ref = fit_on(train_df, feats, None)
        meta = {**base_meta, "reference_stats": ref}
        held = [t for t in trials if t.split == "lockbox"]
        parts.append(score_trials(model, meta, held, args.threshold))
        tag = "LOCKBOX (single use)"
    else:
        for rev in sorted(train_df["rev"].unique()):
            model, ref = fit_on(train_df, feats, rev)
            meta = {**base_meta, "reference_stats": ref}
            held = [t for t in trials if t.split == "train" and t.rev == rev]
            print(f"[rowe] scoring held-out {rev} ({len(held)} trials)")
            parts.append(score_trials(model, meta, held, args.threshold))
        tag = "leave-one-rev-out"

    df = pd.concat(parts, ignore_index=True)
    c = curve(df)
    reasons = reason_report(df)
    unk = unknown_agreement(df)
    revs = sorted(df["rev"].unique())

    print(f"\n[rowe] {tag}: {len(df):,} rows over {revs}")
    for r in c:
        if r["threshold"] in tuple(PRESETS.values()):
            print(f"[rowe]   thr {r['threshold']:.2f}: coverage {r['coverage']:.4f}  "
                  f"selective_acc {r['selective_accuracy']:.4f}  "
                  f"worst_subject {r['worst_rev_accuracy']:.4f}")
    print()
    print(reasons.to_string(index=False))
    if unk:
        print(f"\n[rowe] abstains on {unk['abstain_rate_on_human_unknown']:.1%} of human `-1` "
              f"rows vs {unk['abstain_rate_elsewhere']:.1%} elsewhere")

    stem = "roweval_lockbox" if args.lockbox else "roweval_loro"
    (out_dir / f"{stem}.md").write_text(render(tag, c, reasons, unk, revs), encoding="utf-8")
    (out_dir / f"{stem}.json").write_text(json.dumps(
        {"tag": tag, "threshold": args.threshold, "revs": revs, "curve": c,
         "reasons": reasons.to_dict("records"), "human_unknown": unk}, indent=2),
        encoding="utf-8")
    print(f"\n[rowe] -> {out_dir / (stem + '.md')}")


if __name__ == "__main__":
    main()
