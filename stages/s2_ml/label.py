"""Label a CSV: per-row state, confidence, and — where confidence is low — WHY.

    python -m stages.s2_ml.label input.csv --out labelled.csv
    python -m stages.s2_ml.label input.csv --preset high_precision
    python -m stages.s2_ml.label input.csv --threshold 0.88

This is the deliverable. Everything else in S2 exists to make this row honest.

**It takes either shape of file**, dispatched on FAMILY resolved from the header by name
(§1.3): the derived rev2 view (`L/R_ang_LPF`, `L/R_angvel_LPF`) or a **raw device log**,
whose four channels are rebuilt by `transform.raw_csv_to_features` first. The raw path is
the one a device actually produces, and until 2026-08-03 it did not exist — the pipeline
could only label files that had already been through HUROTICS' MATLAB (caveats §3.1). It
refuses rather than guesses: an unmapped hardware revision, a file whose measured
permutation contradicts the axis about to be read, a gyro that is not natively deg/s, or a
final timestamp that cannot carry the filter all abstain with a stated reason and no
output. `verify_serve.py` runs both shapes of the same recording through this module and
asserts they agree row for row.

**Abstention is the point.** A confident wrong label is worse than an admitted unknown,
so every row carries `confidence` and, below the threshold, `ambiguous=True` with a
`reason` and an `alternative`. Coverage bought at each threshold is measured in
`OPERATING_POINTS.md`; the default commits to ~81% of rows and is right on ~98.8% of them,
with every held-out subject independently above 95%.

**Rows in, rows out.** The model works on a 100 Hz canonical grid, but the output is
indexed by the INPUT file's own rows: probabilities are computed on the grid and mapped
back by time. A caller gets their own CSV with columns appended, not a resampled one they
have to re-join.

**A row is only scored where it was measured.** Rows in a segment shorter than one window,
or outside any segment, come back `ambiguous` with reason `uncovered` and no guess at all.
Extrapolating a class into time that was never observed is exactly the invention §3.1
forbids.
"""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from stages.s1_clean.census import family_of, read_header, strip_prefix
from stages.s1_clean.config import CANONICAL_HZ, FAMILY_MARKERS
from stages.s1_clean.resample import resample_file
from stages.s1_clean.validate import health
from stages.s2_ml.dataset import FEATURES, STAND, TIME_COL, WALK
from stages.s2_ml.features import WindowSpec, feature_names, rest_reference, segment_features
from stages.s2_ml.transform import (
    AxisConflictError,
    DegenerateClockError,
    GyroUnitError,
    NotRawDeviceError,
    UnknownVariantError,
    raw_csv_to_features,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = REPO_ROOT / "runs" / "s2_ml"

RAW_DEVICE, REV2_VIEW = "raw_device", "rev2_view"

# Every way the raw path can decline a file. Caught as a group at the CLI boundary so a
# refusal prints its reason instead of a traceback — an operator holding an unservable
# recording needs to read WHY, and each of these messages says so and names the remedy.
SERVE_REFUSALS = (UnknownVariantError, AxisConflictError, GyroUnitError,
                  DegenerateClockError, NotRawDeviceError, FileNotFoundError)

CLASS_NAME = {STAND: "stand", WALK: "walk"}

# Reasons a row is ambiguous, most specific first. Ordered, so a row near a boundary is
# reported as a boundary rather than as whatever else also happens to be true there.
REASONS = {
    "uncovered": (
        "no full window covers this row (segment shorter than the window, or a gap)",
        "unknown - not enough measured signal to judge"),
    "out_of_distribution": (
        "feature values fall outside the range seen anywhere in training",
        "unmodelled state, or a channel mapping error - check the input columns"),
    "near_transition": (
        "a predicted state change falls within one window of this row",
        "the other class - the boundary is real, its exact timing is not resolvable"),
    "weight_shift_or_step": (
        "interleg signal crosses once in the window: the swap rule's own AMBIGUOUS verdict",
        "stand (a weight shift) or walk (a single step) - one leg passing the other is both"),
    "posture_shift": (
        "both thighs sit well above this recording's own standing posture: not upright",
        "a state outside {stand, walk}: sitting, sit-to-stand, or stand-to-sit"),
    "low_excursion_gait": (
        "called walking, but the legs swing less than walking normally does",
        "stand (postural sway) or very slow shuffling gait"),
    "model_split": (
        "the ensemble is genuinely divided with no single diagnostic cause",
        "the other class"),
}


@lru_cache(maxsize=4)
def load_champion(model_dir: Path):
    """(model, meta). Both are required: the meta carries the window spec the model was
    fitted at and the reference statistics the reasons are stated against, so loading one
    without the other is how train/serve skew starts.

    Cached because `verify_serve` labels 36 files in one process and unpickling 400 trees
    each time dominates its runtime. Keyed on the directory, so pointing at a different
    champion still loads a different model.
    """
    import joblib

    meta = json.loads((model_dir / "model_meta.json").read_text(encoding="utf-8"))
    model = joblib.load(model_dir / "champion.joblib")
    return model, meta


def read_input(path: Path) -> pd.DataFrame:
    """Read a rev2-view CSV and resolve the required columns BY NAME (§1.3 - never by
    position)."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in (TIME_COL, *FEATURES) if c not in df.columns]
    if missing:
        raise SystemExit(
            f"{path.name}: reads as the derived rev2 view but is missing {missing}.\n"
            f"That path needs {TIME_COL} plus all of {list(FEATURES)}."
        )
    return df


def read_source(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """(the caller's own rows, the rev2 view to score, provenance).

    Dispatch is on FAMILY, resolved from the header by name (§1.3) through the same census
    S1 uses — never on "does this file happen to have the columns I want". The two families
    share only `Time`, so the marker column is the honest question and duck-typing the
    columns would misroute a half-written file instead of rejecting it.

    A raw device log goes through the bridge first. That call is where every provenance
    guard fires — unmapped revision, contradicted axis, non-deg/s gyro, unusable final
    timestamp — and each one raises rather than returning a frame, so a refusal can never
    be mistaken for a labelled file.
    """
    family = family_of([strip_prefix(c) for c in read_header(path)])

    if family == REV2_VIEW:
        df = read_input(path)
        return df, df[[TIME_COL, *FEATURES]], {
            "source": str(path), "family": family, "derived": False,
            "n_rows": int(len(df)),
        }

    if family == RAW_DEVICE:
        rows, view, provenance = raw_csv_to_features(path)
        return rows, view, {**provenance, "derived": True}

    raise SystemExit(
        f"{path.name}: unrecognized file family {family!r}.\n"
        f"This module labels a raw device log (marker column "
        f"{FAMILY_MARKERS[RAW_DEVICE]!r}) or the derived rev2 view (marker "
        f"{FAMILY_MARKERS[REV2_VIEW]!r}). Neither marker is present, so nothing here can "
        f"say what the columns mean, and resolving them by position is the §1.3 hazard."
    )


def _accumulate(values: np.ndarray, starts: np.ndarray, n: int, n_rows: int
                ) -> tuple[np.ndarray, np.ndarray]:
    """(sum, count) per row over every window covering it, by difference array.

    O(windows + rows) rather than O(windows * window length), which is what makes a
    0.25 s stride over million-row recordings affordable.
    """
    acc = np.zeros(n_rows + 1)
    cnt = np.zeros(n_rows + 1)
    np.add.at(acc, starts, values)
    np.add.at(acc, starts + n, -values)
    np.add.at(cnt, starts, 1.0)
    np.add.at(cnt, starts + n, -1.0)
    return np.cumsum(acc)[:n_rows], np.cumsum(cnt)[:n_rows]


def score_frame(model, meta: dict, frame: pd.DataFrame) -> pd.DataFrame:
    """Per-row P(walk) and diagnostics on the canonical grid.

    A row's probability is the MEAN over every window containing it. That is what makes a
    boundary abstain on its own: rows near a state change are covered by windows that
    straddle it, so their probabilities pull apart and the mean lands mid-scale. No
    separate boundary rule, and no post-hoc smoothing constant to tune.
    """
    spec = WindowSpec(window_s=meta["window_s"],
                      stride_s=meta.get("inference_stride_s", 0.25),
                      fs_hz=meta.get("fs_hz", CANONICAL_HZ))
    feats = meta["features"]
    idx = [feature_names().index(f) for f in feats]
    zeros, ileg_zero, rest_trusted = rest_reference(frame, spec.fs_hz)
    walk_col = list(model.classes_).index(WALK)

    out = []
    for seg_id, seg in frame.groupby("segment", sort=True):
        seg = seg.reset_index(drop=True)
        n_rows = len(seg)
        base = pd.DataFrame({
            TIME_COL: seg[TIME_COL].to_numpy(float),
            "segment": int(seg_id),
            "p_walk": np.nan, "n_windows": 0,
            "ileg_minhalf": np.nan, "ileg_swaps": np.nan, "n_oob": np.nan,
            "posture": np.nan,
        })
        if n_rows >= spec.n:
            chan = {c: seg[c].to_numpy(float) for c in FEATURES}
            X, starts = segment_features(chan, spec, zeros, ileg_zero)
            if X.shape[0]:
                Xf = X[:, idx]
                pw = model.predict_proba(Xf)[:, walk_col]
                lo = np.array([meta["reference_stats"]["feat_p01"][f] for f in feats])
                hi = np.array([meta["reference_stats"]["feat_p99"][f] for f in feats])
                oob = ((Xf < lo) | (Xf > hi)).sum(1).astype(float)
                names = feature_names()
                mh = X[:, names.index("ileg_minhalf")]
                sw = X[:, names.index("ileg_swaps")]
                post = np.minimum(X[:, names.index("L_ang_med_rest")],
                                  X[:, names.index("R_ang_med_rest")])

                s_p, cnt = _accumulate(pw, starts, spec.n, n_rows)
                covered = cnt > 0
                with np.errstate(invalid="ignore", divide="ignore"):
                    base.loc[covered, "p_walk"] = s_p[covered] / cnt[covered]
                    for col, vals in (("ileg_minhalf", mh), ("ileg_swaps", sw),
                                      ("n_oob", oob), ("posture", post)):
                        s, _ = _accumulate(vals, starts, spec.n, n_rows)
                        base.loc[covered, col] = s[covered] / cnt[covered]
                base["n_windows"] = cnt.astype(int)
        out.append(base)

    scored = pd.concat(out, ignore_index=True)
    scored["rest_trusted"] = rest_trusted
    return scored


def explain(scored: pd.DataFrame, meta: dict, threshold: float,
            spec: WindowSpec) -> pd.DataFrame:
    """Attach guess, confidence, ambiguity and reason. Reasons are ordered by specificity."""
    ref = meta["reference_stats"]
    p = scored["p_walk"].to_numpy(float)
    covered = np.isfinite(p)

    guess = np.where(p >= 0.5, WALK, STAND).astype(float)
    guess[~covered] = np.nan
    conf = np.where(covered, np.maximum(p, 1.0 - p), np.nan)
    ambiguous = ~covered | (conf < threshold)

    # A predicted state change within one window of this row.
    g = pd.Series(guess).ffill().bfill().to_numpy()
    change = np.zeros(len(g), dtype=bool)
    if len(g) > 1:
        at = np.flatnonzero(g[1:] != g[:-1]) + 1
        half = max(1, spec.n // 2)
        for i in at:
            change[max(0, i - half):min(len(g), i + half)] = True
    change &= scored["segment"].to_numpy() == scored["segment"].to_numpy()

    swaps = scored["ileg_swaps"].to_numpy(float)
    mh = scored["ileg_minhalf"].to_numpy(float)
    oob = scored["n_oob"].to_numpy(float)

    reason = np.full(len(scored), "", dtype=object)

    def mark(mask, key):
        m = mask & ambiguous & (reason == "")
        reason[m] = key

    posture = scored["posture"].to_numpy(float)

    mark(~covered, "uncovered")
    # Posture first: "not upright at all" outranks any stand-vs-walk story told about it.
    mark(posture > ref["posture_shift_p99"], "posture_shift")
    mark(np.nan_to_num(oob, nan=0.0) >= 3, "out_of_distribution")
    mark(change, "near_transition")
    mark((swaps >= 0.5) & (swaps < 1.5), "weight_shift_or_step")
    mark((guess == WALK) & (mh < ref["walk_minhalf_p05"]), "low_excursion_gait")
    mark(np.ones(len(scored), dtype=bool), "model_split")

    scored = scored.copy()
    scored["state"] = [CLASS_NAME.get(v) if np.isfinite(v) else None
                       for v in np.where(covered, guess, np.nan)]
    scored["label"] = np.where(covered, guess, np.nan)
    scored["confidence"] = np.round(conf, 4)
    scored["ambiguous"] = ambiguous
    scored["reason"] = reason
    scored["reason_detail"] = [REASONS[r][0] if r else "" for r in reason]
    scored["alternative"] = [REASONS[r][1] if r else "" for r in reason]
    # A recording that never rests has a fallback interleg zero, so every interleg feature
    # on it is weaker evidence. Said once, on every row, rather than silently.
    if not bool(scored["rest_trusted"].iloc[0]):
        scored["reason_detail"] = scored["reason_detail"].astype(str) + (
            " | recording never rests: interleg zero is a fallback median, not a measured "
            "standing posture")
    return scored


def label_csv(path: Path, model_dir: Path, threshold: float) -> tuple[pd.DataFrame, dict]:
    """Label one CSV; returns (the INPUT rows with the label columns appended, provenance).

    Provenance travels with the frame rather than being printed and forgotten: on the raw
    path it records which hardware revision was resolved, which two channels were actually
    read, and which trust record was consulted. A labelled file whose axis resolution
    cannot be reconstructed later is exactly the "recorded but never read" hole that let
    the wrong sagittal axis ship for the majority variant.
    """
    model, meta = load_champion(model_dir)
    spec = WindowSpec(window_s=meta["window_s"],
                      stride_s=meta.get("inference_stride_s", 0.25),
                      fs_hz=meta.get("fs_hz", CANONICAL_HZ))
    raw, view, provenance = read_source(path)

    # Same normalization the model was trained under: segment at gaps, then the canonical
    # grid. Anything else is train/serve skew.
    frame, _segments = resample_file(view, TIME_COL)
    if frame.empty:
        raise SystemExit(f"{path.name}: no usable segment survived resampling")

    # S1's label-free gate, in front of the model rather than beside it. A dead channel or
    # a file with nothing scorable would otherwise produce confident numbers built on
    # nothing, which is worse than refusing.
    h = health(frame, window_s=spec.window_s, fs_hz=spec.fs_hz)
    for w in h.warnings:
        print(f"[label] warning: {w}")
    if not h.usable:
        raise SystemExit(f"{path.name}: not usable\n  "
                         + "\n  ".join(h.errors))

    scored = explain(score_frame(model, meta, frame), meta, threshold, spec)

    # Map the grid back onto the caller's own rows by nearest time.
    t_grid = scored[TIME_COL].to_numpy(float)
    t_in = raw[TIME_COL].to_numpy(float)
    order = np.argsort(t_grid)
    tg = t_grid[order]
    j = np.searchsorted(tg, t_in).clip(1, tg.size - 1)
    nearest = order[np.where(np.abs(t_in - tg[j - 1]) <= np.abs(tg[j] - t_in), j - 1, j)]
    # Beyond half a grid step from any scored sample, the row sits in a gap.
    gap = np.abs(t_in - t_grid[nearest]) > (1000.0 / spec.fs_hz)

    cols = ["state", "label", "confidence", "ambiguous", "reason", "reason_detail",
            "alternative", "n_windows"]
    picked = scored.iloc[nearest][cols].reset_index(drop=True)
    picked.loc[gap, ["state", "label", "confidence"]] = None
    picked.loc[gap, "ambiguous"] = True
    picked.loc[gap, "reason"] = "uncovered"
    picked.loc[gap, "reason_detail"] = REASONS["uncovered"][0]
    picked.loc[gap, "alternative"] = REASONS["uncovered"][1]

    # On the raw path the four derived channels are the model's actual input and appear
    # nowhere in the caller's file, so they ride along. Without them the output states a
    # verdict over columns nobody can see, and the sagittal-axis bug would have been
    # invisible in the deliverable as well as in the code.
    parts = [raw.reset_index(drop=True)]
    if provenance["derived"]:
        parts.append(view[list(FEATURES)].reset_index(drop=True))
    parts.append(picked)
    return pd.concat(parts, axis=1), provenance


def describe_source(provenance: dict) -> str:
    """The one-block answer to "what did the model actually read?".

    Printed for every run, not only the interesting ones. The axis a raw file resolves to
    is the single assumption this path cannot verify from inside the file, so it is stated
    every time rather than left to be reconstructed from the variant id later.
    """
    if not provenance["derived"]:
        return (f"source: {provenance['family']} (already the model's input, no bridge)\n"
                f"rows in: {provenance['n_rows']:,}")
    return "\n".join([
        f"source: raw device log, variant {provenance['variant_id']}",
        f"sagittal axis: Deg_{provenance['sagittal_deg_axis']} "
        f"-> Gyro_{provenance['gyro_axis']}  (per-revision lookup, never guessed)",
        f"channels read: {', '.join(provenance['columns_read'])}",
        f"filter dt: {provenance['dt_ms']:.5f} ms "
        f"({provenance['dt_last_over_median']:.4f}x the median interval)",
        f"trust record: {provenance['trust_record']}",
        f"axis detection: L={provenance['axis_check']['L']} R={provenance['axis_check']['R']}"
        f"   gyro unit: L={provenance['gyro_unit']['L']} R={provenance['gyro_unit']['R']}",
        f"rows in: {provenance['n_rows']:,}",
    ])


def summarize(df: pd.DataFrame, threshold: float) -> str:
    n = len(df)
    amb = int(df["ambiguous"].sum())
    lines = [f"rows: {n:,}   threshold: {threshold:.2f}",
             f"labelled: {n - amb:,} ({(n - amb) / n:.1%})   "
             f"ambiguous: {amb:,} ({amb / n:.1%})"]
    conf = df.loc[~df["ambiguous"], "state"].value_counts()
    if len(conf):
        lines.append("confident states: " + ", ".join(f"{k} {v:,}" for k, v in conf.items()))
    if amb:
        lines.append("ambiguity reasons:")
        for k, v in df.loc[df["ambiguous"], "reason"].value_counts().items():
            lines.append(f"  {k:<22} {v:>8,}  ({v / amb:.1%} of ambiguous)")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Label a CSV with stand/walk + confidence.")
    ap.add_argument("csv", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--preset", default=None, help="high_coverage | balanced | high_precision")
    ap.add_argument("--threshold", type=float, default=None)
    args = ap.parse_args()

    meta = json.loads((args.model_dir / "model_meta.json").read_text(encoding="utf-8"))
    presets = meta["presets"]
    if args.threshold is not None:
        threshold = args.threshold
    else:
        name = args.preset or meta["default_preset"]
        if name not in presets:
            raise SystemExit(f"unknown preset {name!r}; known: {sorted(presets)}")
        threshold = presets[name]

    try:
        out, provenance = label_csv(args.csv, args.model_dir, threshold)
    except SERVE_REFUSALS as exc:
        # An abstention, not a crash. The file is unservable for a stated reason and the
        # right outcome is no output at all — a labelled file the pipeline cannot defend
        # is worse than none, because a controller would act on it.
        raise SystemExit(f"{args.csv.name}: ABSTAINED — {type(exc).__name__}\n  {exc}")

    print(describe_source(provenance))
    print()
    print(summarize(out, threshold))
    dest = args.out or args.csv.with_name(args.csv.stem + "_labelled.csv")
    out.to_csv(dest, index=False)
    print(f"-> {dest}")


if __name__ == "__main__":
    main()
