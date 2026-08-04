# end-to-end accuracy of the RAW DEVICE path, scored against human labels
#   python -m stages.s2_ml.raweval
# the lockbox is refused in code, and no subject is scored by a model that saw it
# read this next to roweval's number, not next to the lockbox one

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from stages.console import use_replacement_encoding
from stages.s2_ml.dataset import (
    DEFAULT_LOCKBOX_REVS,
    EXCLUDED_TRIALS,
    LABEL_COL,
    STAND,
    TIME_COL,
    TRAIN_CLASSES,
    WALK,
    _read_raw,
    load_dataset,
    rev_of,
    trial_of,
)
from stages.s2_ml.features import WindowSpec, build_windows, feature_columns
from stages.s2_ml.label import DEFAULT_MODEL_DIR, label_csv
from stages.s2_ml.locoeval import DEFAULT_THRESHOLDS
from stages.s2_ml.roweval import fit_on
from stages.s2_ml.train import PRESETS, load_spec, select_features, trainable
from stages.s2_ml.verify_transform import find_pairs, index_raw_files

REPO_ROOT = Path(__file__).resolve().parents[2]

CLASS_NAME = {STAND: "stand", WALK: "walk"}


# paired recordings, minus every sealed rev
def _drop_lockbox(pairs: list, lockbox: tuple[str, ...] = DEFAULT_LOCKBOX_REVS) -> list:
    kept, refused = [], []
    for pair in pairs:
        (refused if rev_of(pair[0]) in lockbox else kept).append(pair)
    if refused:
        revs = sorted({rev_of(p[0]) for p in refused})
        print(f"[rawe] lockbox: refusing {len(refused)} pair(s) from {revs} (§7)")
    return kept


# minus the trials whose ANNOTATIONS are quarantined
def _drop_quarantined(pairs: list) -> list:
    kept = [p for p in pairs
            if (rev_of(p[0]), trial_of(p[0])) not in EXCLUDED_TRIALS]
    if (n := len(pairs) - len(kept)):
        print(f"[rawe] quarantine: refusing {n} pair(s) with excluded annotations")
    return kept


# A model directory `label_csv` can be pointed at, holding the champion refit WITHOUT `rev`
def fold_model_dir(train_df: pd.DataFrame, feats: list[str], rev: str,
                   base_meta: dict, root: Path):
    import joblib

    model, ref = fit_on(train_df, feats, rev)
    d = root / f"without_{rev}"
    d.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, d / "champion.joblib")
    (d / "model_meta.json").write_text(
        json.dumps({**base_meta, "features": feats, "reference_stats": ref,
                    "held_out_rev": rev}, indent=2),
        encoding="utf-8")
    return d


# the human label for every row of a raw-path output, or -2 where none aligns
def truth_for(out: pd.DataFrame, ann_path: Path, fs_hz: float) -> np.ndarray:
    ann = _read_raw(ann_path)
    t_ann = ann[TIME_COL].to_numpy(float)
    y_ann = ann[LABEL_COL].to_numpy(float)
    order = np.argsort(t_ann)
    ta, ya = t_ann[order], y_ann[order]

    t_out = out[TIME_COL].to_numpy(float)
    j = np.searchsorted(ta, t_out).clip(1, ta.size - 1)
    near = np.where(np.abs(t_out - ta[j - 1]) <= np.abs(ta[j] - t_out), j - 1, j)
    ok = np.abs(t_out - ta[near]) <= (1000.0 / fs_hz) / 2.0
    return np.where(ok, ya[near], -2.0)


# label one raw device CSV through the shipping entry point, truth joined back on
def score_pair(ann_path: Path, raw_path: Path, model_dir: Path, variant: str,
               threshold: float, fs_hz: float) -> pd.DataFrame:
    out, prov = label_csv(raw_path, model_dir, threshold)
    truth = truth_for(out, ann_path, fs_hz)
    guess = pd.to_numeric(out["label"], errors="coerce").to_numpy(float)
    return pd.DataFrame({
        "rev": rev_of(ann_path),
        "trial": trial_of(ann_path),
        "variant": variant,
        "axis": prov["sagittal_deg_axis"],
        "truth": truth,
        "guess": guess,
        "confidence": pd.to_numeric(out["confidence"], errors="coerce").to_numpy(float),
        "ambiguous": out["ambiguous"].to_numpy(bool),
        "reason": out["reason"].where(out["reason"].notna(), "").to_numpy(object),
    })


# rows a claim about accuracy may be made from: a human label in the trained vocabulary
def _scorable(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["truth"].isin(TRAIN_CLASSES)]


# coverage and selective accuracy over the whole paired set, and per subject
def summarize(df: pd.DataFrame) -> dict:
    v = _scorable(df)
    committed = ~v["ambiguous"].to_numpy(bool) & np.isfinite(v["guess"].to_numpy(float))
    correct = v["guess"].to_numpy(float) == v["truth"].to_numpy(float)

    def point(sub_c: np.ndarray, sub_ok: np.ndarray, n: int) -> dict:
        return {"rows_scored": int(n),
                "committed": int(sub_c.sum()),
                "coverage": float(sub_c.mean()) if n else float("nan"),
                "selective_accuracy": (float(sub_ok[sub_c].mean()) if sub_c.any()
                                       else float("nan")),
                "errors_kept": int((~sub_ok[sub_c]).sum())}

    per_rev = {}
    for rev in sorted(v["rev"].unique()):
        k = (v["rev"] == rev).to_numpy()
        sub = point(committed[k], correct[k], int(k.sum()))
        for cls, name in CLASS_NAME.items():
            m = k & (v["truth"].to_numpy(float) == cls) & committed
            n_cls = int((k & (v["truth"].to_numpy(float) == cls)).sum())
            sub[f"{name}_recall"] = float(correct[m].mean()) if m.any() else float("nan")
            sub[f"{name}_support"] = int(m.sum())
            sub[f"{name}_rows"] = n_cls
        per_rev[rev] = sub

    per_variant = {}
    for var in sorted(v["variant"].unique()):
        k = (v["variant"] == var).to_numpy()
        per_variant[var] = {"revs": sorted(v.loc[k, "rev"].unique().tolist()),
                            **point(committed[k], correct[k], int(k.sum()))}

    conf = {}
    for t_cls, t_name in CLASS_NAME.items():
        for g_cls, g_name in CLASS_NAME.items():
            k = committed & (v["truth"].to_numpy(float) == t_cls) \
                & (v["guess"].to_numpy(float) == g_cls)
            conf[f"{t_name}->{g_name}"] = int(k.sum())

    worst = min((r["selective_accuracy"] for r in per_rev.values()), default=float("nan"))
    return {
        "overall": {**point(committed, correct, len(v)), "worst_subject": float(worst)},
        "per_rev": per_rev,
        "per_variant": per_variant,
        "confusion": conf,
        "excluded": {
            "human_unknown": int((df["truth"] == -1).sum()),
            "unmatched_time": int((df["truth"] == -2).sum()),
            "total_rows_returned": int(len(df)),
        },
    }


def render(res: dict, threshold: float, pairs: list, transitive: dict | None) -> str:
    o = res["overall"]
    lines = [
        "# Raw device path - end-to-end accuracy",
        "",
        f"Raw device CSVs labelled through `label_csv` at threshold **{threshold:.2f}** and "
        f"scored against the human annotation of the same recording.",
        "",
        f"- paired recordings: **{len(pairs)}** across **{len(res['per_rev'])}** subjects "
        f"({', '.join(sorted(res['per_rev']))})",
        f"- rows returned: {res['excluded']['total_rows_returned']:,}  →  "
        f"scorable: **{o['rows_scored']:,}** "
        f"(dropped {res['excluded']['human_unknown']:,} human `-1`, "
        f"{res['excluded']['unmatched_time']:,} with no annotated sample in range)",
        "",
        "**The lockbox is not in this table.** `rev8`'s four pairs are refused in code (§7), "
        "so every subject here is a development subject. Read this against `roweval_loro`'s "
        "0.9901, not against rev8's 0.9308.",
        "",
        "| | |",
        "|---|---|",
        f"| coverage | **{o['coverage']:.4f}** |",
        f"| selective accuracy | **{o['selective_accuracy']:.4f}** |",
        f"| worst subject | {o['worst_subject']:.4f} |",
        f"| committed rows | {o['committed']:,} |",
        f"| errors kept | {o['errors_kept']:,} |",
        "",
    ]

    if transitive:
        lines += [
            "## Against the transitive claim",
            "",
            "The number this replaces: raw equals `lpf_view` (`verify_serve`), `lpf_view` "
            "scores X (`roweval_loro`), therefore raw scores X. Same three subjects, same "
            "held-out discipline, so the two rows below are comparable.",
            "",
            "| route | rows | coverage | selective acc |",
            "|---|---|---|---|",
            f"| `lpf_view` (roweval, these 3 revs) | {transitive['rows_scored']:,} "
            f"| {transitive['coverage']:.4f} | {transitive['selective_accuracy']:.4f} |",
            f"| **raw device (this file)** | {o['rows_scored']:,} "
            f"| **{o['coverage']:.4f}** | **{o['selective_accuracy']:.4f}** |",
            "",
            "The two row sets are not identical — `roweval` scores the annotated export's "
            "grid and this scores the raw file's own rows — so exact equality is not the "
            "bar. A gap that changed the operating point would be.",
            "",
        ]

    lines += ["## Per subject", "",
              "Each labelled by a champion refit without it.", "",
              "| rev | trials | rows | coverage | accuracy | errors | stand recall | walk recall |",
              "|---|---|---|---|---|---|---|---|"]
    for rev, r in sorted(res["per_rev"].items()):
        n_tr = sum(1 for p in pairs if rev_of(p[0]) == rev)
        lines.append(
            f"| `{rev}` | {n_tr} | {r['rows_scored']:,} | {r['coverage']:.1%} "
            f"| {r['selective_accuracy']:.4f} | {r['errors_kept']:,} "
            f"| {r['stand_recall']:.4f} (n={r['stand_support']:,}) "
            f"| {r['walk_recall']:.4f} (n={r['walk_support']:,}) |")

    lines += ["", "## Per hardware variant", "",
              "The axis map is what the raw route adds over the `lpf_view` route, so it is "
              "the thing this measurement is really testing. A variant whose accuracy sits "
              "apart from the others is a mis-mapped sagittal axis, which is exactly the "
              "failure that shipped undetected until 2026-08-03 (`caveats.md` §5).", "",
              "| variant | revs | rows | coverage | accuracy | errors |",
              "|---|---|---|---|---|---|"]
    for var, r in sorted(res["per_variant"].items()):
        lines.append(f"| `{var}` | {', '.join(r['revs'])} | {r['rows_scored']:,} "
                     f"| {r['coverage']:.1%} | {r['selective_accuracy']:.4f} "
                     f"| {r['errors_kept']:,} |")

    c = res["confusion"]
    lines += ["", "## Direction of error", "",
              "| truth → guess | rows |", "|---|---|"]
    for k in ("walk->walk", "stand->stand", "stand->walk", "walk->stand"):
        mark = " ⚠" if k in ("stand->walk", "walk->stand") else ""
        lines.append(f"| {k.replace('->', ' → ')}{mark} | {c[k]:,} |")
    return "\n".join(lines) + "\n"


# `roweval`'s own measurement, restricted to the revs this module could pair
def transitive_baseline(trials, train_df, feats, base_meta, revs, threshold) -> dict:
    from stages.s2_ml.roweval import score_trials

    parts = []
    for rev in revs:
        model, ref = fit_on(train_df, feats, rev)
        held = [t for t in trials if t.split == "train" and t.rev == rev]
        parts.append(score_trials(model, {**base_meta, "features": feats,
                                          "reference_stats": ref}, held, threshold))
    df = pd.concat(parts, ignore_index=True)
    v = df[df["truth"].isin(TRAIN_CLASSES)]
    committed = ~v["ambiguous"].to_numpy(bool) & v["p_walk"].notna().to_numpy()
    correct = (np.where(v["p_walk"].to_numpy(float) >= 0.5, WALK, STAND)
               == v["truth"].to_numpy())
    return {"rows_scored": int(len(v)), "committed": int(committed.sum()),
            "coverage": float(committed.mean()),
            "selective_accuracy": float(correct[committed].mean())}


def main() -> None:
    use_replacement_encoding()   # the lockbox notice prints a section sign
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/s2_ml")
    ap.add_argument("--threshold", type=float, default=PRESETS["balanced"])
    ap.add_argument("--no-baseline", action="store_true",
                    help="skip the matched-subject roweval comparison")
    args = ap.parse_args()

    out_dir = (REPO_ROOT / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = _drop_quarantined(_drop_lockbox(find_pairs(index_raw_files())))
    if not pairs:
        raise SystemExit("no scorable pairs — nothing to measure")
    paired_revs = sorted({rev_of(p[0]) for p in pairs})
    print(f"[rawe] {len(pairs)} pair(s) over {paired_revs}")

    trials = load_dataset()
    spec = WindowSpec()
    windows = build_windows([t for t in trials if t.split == "train"], spec)
    feats = select_features(feature_columns(windows), load_spec()["drop_features"])
    train_df = trainable(windows, "train")
    assert "lockbox" not in set(train_df["split"]), "lockbox leaked into training"
    print(f"[rawe] fitting on {len(train_df):,} windows, {len(feats)} features")

    base_meta = {"window_s": spec.window_s, "stride_s": spec.stride_s, "fs_hz": spec.fs_hz,
                 "inference_stride_s": 0.25}

    tmp = Path(tempfile.mkdtemp(prefix="raweval_"))
    try:
        parts = []
        for rev in paired_revs:
            d = fold_model_dir(train_df, feats, rev, base_meta, tmp)
            mine = [p for p in pairs if rev_of(p[0]) == rev]
            print(f"[rawe] held-out {rev}: labelling {len(mine)} raw file(s)")
            for ann_path, raw_path, _df, vid in mine:
                parts.append(score_pair(ann_path, Path(raw_path), d, vid,
                                        args.threshold, spec.fs_hz))
        df = pd.concat(parts, ignore_index=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    res = summarize(df)
    transitive = (None if args.no_baseline else
                  transitive_baseline(trials, train_df, feats, base_meta,
                                      paired_revs, args.threshold))

    o = res["overall"]
    print(f"\n[rawe] RAW DEVICE PATH, thr {args.threshold:.2f}: "
          f"coverage {o['coverage']:.4f}  selective_acc {o['selective_accuracy']:.4f}  "
          f"worst_subject {o['worst_subject']:.4f}  errors {o['errors_kept']:,}")
    if transitive:
        print(f"[rawe] lpf_view route, same subjects: "
              f"coverage {transitive['coverage']:.4f}  "
              f"selective_acc {transitive['selective_accuracy']:.4f}")

    payload = {"threshold": args.threshold, "n_pairs": len(pairs),
               "revs": paired_revs, "n_features": len(feats),
               "lockbox_refused": list(DEFAULT_LOCKBOX_REVS),
               "pairs": [{"annotated": Path(a).name, "raw": Path(r).name, "variant": v}
                         for a, r, _d, v in pairs],
               "transitive_lpf_view": transitive, **res}
    (out_dir / "raweval.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out_dir / "raweval.md").write_text(
        render(res, args.threshold, pairs, transitive), encoding="utf-8")
    print(f"[rawe] artifacts -> {out_dir}")


if __name__ == "__main__":
    main()
