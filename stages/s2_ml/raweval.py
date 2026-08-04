"""End-to-end accuracy of the RAW DEVICE path, scored against human labels.

    python -m stages.s2_ml.raweval

`roweval.py` is the accuracy number this repo quotes, and it is measured on the `lpf_view`
family — the annotated export MATLAB produced. `verify_serve.py` then shows the raw device
route agrees with that export row for row, but deliberately drops `Label` before comparing,
so it proves *equivalence* and never touches *correctness*. The accuracy of the route a
caller actually uses has therefore only ever been available transitively: raw equals
lpf_view (verify_serve), lpf_view is 0.9904 (roweval), so raw is 0.9904. That chain is
sound and it is still a chain. This module measures the endpoint directly — a raw device
CSV in, `label_csv` out, joined against the human annotation of the same recording.

**What makes it honest, and what it costs.**

  1. *The lockbox stays sealed.* `rev8` has four paired recordings and they are refused
     here, in code (`_drop_lockbox`), not by remembering to pass a flag. §7 spends the
     lockbox once and it is spent; a "direct" number that quietly re-read it would be worth
     less than the transitive one it replaced.
  2. *No subject is scored by a model that saw it.* Pairs are grouped by rev and each rev
     is labelled by a champion refit without it, exactly as `roweval` does — reference
     statistics included, since those enter the abstention reasons. The shipped
     `champion.joblib` is fit on every training rev, so pointing it at rev7's raw file
     would report memorization.

The cost of (1) and (2) together is scope: 14 pairs across **three** subjects, all of them
development subjects. This number belongs next to roweval's 0.9904, measured on the same
population under the same discipline, and NOT next to rev8's 0.9308. It answers "does the
raw route deliver what the lpf_view route was measured to deliver", not "how does the
pipeline do on a new person" — the lockbox already answered that one, less flatteringly.

Unlike `roweval`, this fits at the champion's declared feature count (38, per
`champion_spec.json`) rather than at every column `build_windows` emits (42). The two are
inside the noise band by measurement, but the number this file reports is meant to be the
shipped artifact's, so it uses the shipped artifact's spec.
"""

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


def _drop_lockbox(pairs: list, lockbox: tuple[str, ...] = DEFAULT_LOCKBOX_REVS) -> list:
    """Paired recordings, minus every sealed rev.

    A filter rather than a flag. `verify_serve` may include rev8 because it structurally
    cannot read a label; this module exists to read labels, so the same pair set is not
    safe here and the difference has to live in code. Returns the kept pairs and prints
    what it refused, because a silent exclusion looks identical to a corpus that never had
    those files.
    """
    kept, refused = [], []
    for pair in pairs:
        (refused if rev_of(pair[0]) in lockbox else kept).append(pair)
    if refused:
        revs = sorted({rev_of(p[0]) for p in refused})
        print(f"[rawe] lockbox: refusing {len(refused)} pair(s) from {revs} (§7)")
    return kept


def _drop_quarantined(pairs: list) -> list:
    """Minus the trials whose ANNOTATIONS are quarantined.

    `find_pairs` passes `excluded=set()` on purpose — it asks whether four columns can be
    rebuilt from raw, which no label quarantine bears on. Here the label IS the measuring
    stick, so a trial excluded for having wrong labels would be scoring the model against
    an error this repo has already documented and rejected.
    """
    kept = [p for p in pairs
            if (rev_of(p[0]), trial_of(p[0])) not in EXCLUDED_TRIALS]
    if (n := len(pairs) - len(kept)):
        print(f"[rawe] quarantine: refusing {n} pair(s) with excluded annotations")
    return kept


def fold_model_dir(train_df: pd.DataFrame, feats: list[str], rev: str,
                   base_meta: dict, root: Path):
    """A model directory `label_csv` can be pointed at, holding the champion refit
    WITHOUT `rev`.

    Written to disk rather than passed in memory because the point of this module is to
    call the caller's entry point. `label_csv` loads a champion from a directory; handing
    it a directory is how it gets used, and a variant that accepts a live model would be a
    second serve path measured in place of the first.
    """
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


def truth_for(out: pd.DataFrame, ann_path: Path, fs_hz: float) -> np.ndarray:
    """The human label for every row of a raw-path output, or -2 where none aligns.

    Joined on TIME rather than by position. The pair criterion already requires equal row
    counts and a shared start, so position would almost always work — but "almost always"
    is how an off-by-one in one recording becomes a quiet 3% accuracy loss attributed to
    the model. A row further than half a sample from any annotated sample is reported as
    unmatched instead of being given its neighbour's label.
    """
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


def score_pair(ann_path: Path, raw_path: Path, model_dir: Path, variant: str,
               threshold: float, fs_hz: float) -> pd.DataFrame:
    """Label one raw device CSV through the shipping entry point, truth joined back on."""
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


def _scorable(df: pd.DataFrame) -> pd.DataFrame:
    """Rows a claim about accuracy may be made from: a human label in the trained
    vocabulary. Drops `-1` (annotator marked it unknown, §5.2) and `-2` (no annotated
    sample within half a grid step)."""
    return df[df["truth"].isin(TRAIN_CLASSES)]


def summarize(df: pd.DataFrame) -> dict:
    """Coverage and selective accuracy over the whole paired set, and per subject.

    Committed is read off `ambiguous`, the flag the caller receives — not off a confidence
    comparison. The band, the physics policy and the uncovered rule all move that flag
    without moving the confidence, so recomputing the threshold here would report a
    coverage `label.py` does not deliver.
    """
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
        "0.9904, not against rev8's 0.9308.",
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


def transitive_baseline(trials, train_df, feats, base_meta, revs, threshold) -> dict:
    """`roweval`'s own measurement, restricted to the revs this module could pair.

    Without it the headline has nothing to be compared against: the paired subset is three
    subjects and roweval's published 0.9904 is seven, so quoting them side by side would
    attribute a population difference to the route. Recomputed here rather than read from
    `roweval_loro.json`, which reports only the pooled seven-subject curve.
    """
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
