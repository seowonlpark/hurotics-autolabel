# label a CSV: per-row state, confidence, and WHY it is low; Label commits, guess is the opinion

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from stages.console import use_replacement_encoding
from stages.s1_clean.census import family_of, read_header, strip_prefix
from stages.s1_clean.config import CANONICAL_HZ, FAMILY_MARKERS, LABEL_UNKNOWN_MACHINE
from stages.s1_clean.resample import resample_file
from stages.s1_clean.validate import health
from stages.s2_ml.dataset import (
    CLASS_NAME,
    FEATURES,
    HUMAN_UNKNOWN,
    LABEL_COL,
    STAND,
    TIME_COL,
    WALK,
)
from stages.s2_ml.features import WindowSpec, feature_names, rest_reference, segment_features
# S3 imports S2, never the reverse- except this leaf import of the swap rule, which closes no cycle
from stages.s3_physics.plausibility import check as plausibility_check
from stages.s3_physics.serve import STANDING, WALKING, row_verdict, segment_verdicts
from stages.s2_ml.transform import (
    AxisConflictError,
    DegenerateClockError,
    GyroUnitError,
    NotRawDeviceError,
    UnknownVariantError,
    raw_csv_to_features,
)

from runslayout import REGEN

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = REGEN / "s2_ml"

RAW_DEVICE, LPF_VIEW = "raw_device", "lpf_view"

# every way the raw path declines a file; caught as a group so a refusal prints WHY, not a traceback
SERVE_REFUSALS = (UnknownVariantError, AxisConflictError, GyroUnitError,
                  DegenerateClockError, NotRawDeviceError, FileNotFoundError)

# ---- the physics floor, DEFAULT OFF; every OPERATING_POINTS number assumes that ----
# the ceiling that used to sit beside it is gone: measured at -11 errors at the shipped point,
# noise, and an addressable set of exactly 0 at p>=0.95. `roweval` still sweeps it as a candidate
PHYSICS_FLOOR = False

# the reduced bar when the swap rule agrees; not tuned- it is the `balanced` preset's own threshold
PHYSICS_FLOOR_THRESHOLD = 0.70

# abstain on the whole ambiguity band. OFF, but NOT retracted like the ceiling was: the table that
# beats it scores the band against the very labels the band exists to distrust, so it is a judgement
# and not a measured loss. kept switchable for anyone holding other evidence about the annotation
BAND_ABSTAINS = False

# reasons a row is ambiguous, most specific first, so a boundary is reported as a boundary
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
    "amplitude_ambiguous": (
        "the legs swing in the range where human annotators call both stand and walk, so "
        "no amplitude rule separates it and neither does a model reading amplitude",
        "either class - the ground truth itself is not consistent in this range"),
    "model_split": (
        "the ensemble is genuinely divided with no single diagnostic cause",
        "the other class"),
}


# (model, meta); both required- meta carries the window spec and reference stats, so one alone skews
@lru_cache(maxsize=4)
def load_champion(model_dir: Path):
    import joblib

    meta = json.loads((model_dir / "model_meta.json").read_text(encoding="utf-8"))
    model = joblib.load(model_dir / "champion.joblib")
    return model, meta


# read an lpf_view CSV, resolving the required columns BY NAME, never by position
def read_input(path: Path) -> pd.DataFrame:
    # index_col=False: resolving BY NAME does not protect against a trailing comma shifting the data
    df = pd.read_csv(path, encoding="utf-8-sig", index_col=False)
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in (TIME_COL, *FEATURES) if c not in df.columns]
    if missing:
        raise SystemExit(
            f"{path.name}: reads as family 'lpf_view' but is missing {missing}.\n"
            f"That path needs {TIME_COL} plus all of {list(FEATURES)}."
        )
    return df


# (caller's rows, the frame to score, provenance); dispatch is on FAMILY, never on duck-typed columns
def read_source(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    family = family_of([strip_prefix(c) for c in read_header(path)])

    if family == LPF_VIEW:
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
        f"{FAMILY_MARKERS[RAW_DEVICE]!r}) or an lpf_view file (marker "
        f"{FAMILY_MARKERS[LPF_VIEW]!r}). Neither marker is present, so nothing here can "
        f"say what the columns mean, and resolving them by position is the §1.3 hazard."
    )


# (sum, count) per row over every covering window, by difference array: O(windows+rows), not product
def _accumulate(values: np.ndarray, starts: np.ndarray, n: int, n_rows: int
                ) -> tuple[np.ndarray, np.ndarray]:
    acc = np.zeros(n_rows + 1)
    cnt = np.zeros(n_rows + 1)
    np.add.at(acc, starts, values)
    np.add.at(acc, starts + n, -values)
    np.add.at(cnt, starts, 1.0)
    np.add.at(cnt, starts + n, -1.0)
    return np.cumsum(acc)[:n_rows], np.cumsum(cnt)[:n_rows]


# per-row P(walk) as the MEAN over covering windows, which makes a boundary abstain on its own
def score_frame(model, meta: dict, frame: pd.DataFrame) -> pd.DataFrame:
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
            "posture": np.nan, "physics_swaps": np.nan, "physics_span_s": np.nan,
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

                # S3's verdict on the same windows, ALWAYS; `starts` is asserted equal, not assumed
                sw_ad, span_ad, starts_p = segment_verdicts(chan, spec, ileg_zero)
                if not np.array_equal(starts_p, starts):
                    raise AssertionError(
                        f"physics grid != feature grid on segment {seg_id} "
                        f"({len(starts_p)} vs {len(starts)} windows). "
                        f"features.segment_features and s3_physics.serve.segment_verdicts "
                        f"must walk the same window starts.")

                s_p, cnt = _accumulate(pw, starts, spec.n, n_rows)
                covered = cnt > 0
                with np.errstate(invalid="ignore", divide="ignore"):
                    base.loc[covered, "p_walk"] = s_p[covered] / cnt[covered]
                    for col, vals in (("ileg_minhalf", mh), ("ileg_swaps", sw),
                                      ("n_oob", oob), ("posture", post),
                                      ("physics_swaps", sw_ad), ("physics_span_s", span_ad)):
                        s, _ = _accumulate(vals, starts, spec.n, n_rows)
                        base.loc[covered, col] = s[covered] / cnt[covered]
                base["n_windows"] = cnt.astype(int)
        out.append(base)

    scored = pd.concat(out, ignore_index=True)
    scored["rest_trusted"] = rest_trusted
    # the categorical form, derived once so every consumer reads the same bands off the same count
    scored["physics"] = row_verdict(scored["physics_swaps"].to_numpy(float))
    return scored


# attach guess, confidence, ambiguity and reason; reasons are ordered by specificity
def explain(scored: pd.DataFrame, meta: dict, threshold: float,
            spec: WindowSpec) -> pd.DataFrame:
    ref = meta["reference_stats"]
    p = scored["p_walk"].to_numpy(float)
    covered = np.isfinite(p)

    guess = np.where(p >= 0.5, WALK, STAND).astype(float)
    guess[~covered] = np.nan
    conf = np.where(covered, np.maximum(p, 1.0 - p), np.nan)

    # the floor as a per-row EFFECTIVE THRESHOLD, so one comparison decides every row
    phys = scored["physics"].to_numpy(object)
    phys_class = np.where(phys == WALKING, float(WALK),
                          np.where(phys == STANDING, float(STAND), np.nan))
    decisive = covered & np.isfinite(phys_class)
    agrees = decisive & (phys_class == guess)

    eff = np.full(len(scored), float(threshold))
    if PHYSICS_FLOOR:
        eff[agrees] = PHYSICS_FLOOR_THRESHOLD
    ambiguous = ~covered | (conf < eff)

    # the ambiguity band, averaged as the probability is; the probability itself is NOT consulted
    band = (float(ref["band_lo"]), float(ref["band_hi"])) if "band_lo" in ref else None
    in_band = np.zeros(len(scored), dtype=bool)
    if band is not None and BAND_ABSTAINS:
        mh_all = scored["ileg_minhalf"].to_numpy(float)
        with np.errstate(invalid="ignore"):
            in_band = covered & (mh_all >= band[0]) & (mh_all <= band[1])
        ambiguous = ambiguous | in_band

    # a predicted state change within one window; ACROSS a segment boundary counts, deliberately
    g = pd.Series(guess).ffill().bfill().to_numpy()
    change = np.zeros(len(g), dtype=bool)
    if len(g) > 1:
        at = np.flatnonzero(g[1:] != g[:-1]) + 1
        half = max(1, spec.n // 2)
        for i in at:
            change[max(0, i - half):min(len(g), i + half)] = True

    swaps = scored["ileg_swaps"].to_numpy(float)
    mh = scored["ileg_minhalf"].to_numpy(float)
    oob = scored["n_oob"].to_numpy(float)

    reason = np.full(len(scored), "", dtype=object)

    def mark(mask, key):
        m = mask & ambiguous & (reason == "")
        reason[m] = key

    posture = scored["posture"].to_numpy(float)

    mark(~covered, "uncovered")
    # Posture first: "not upright at all" outranks any stand-vs-walk story told about it
    mark(posture > ref["posture_shift_p99"], "posture_shift")
    mark(np.nan_to_num(oob, nan=0.0) >= 3, "out_of_distribution")
    mark(change, "near_transition")
    mark((swaps >= 0.5) & (swaps < 1.5), "weight_shift_or_step")
    mark((guess == WALK) & (mh < ref["walk_minhalf_p05"]), "low_excursion_gait")
    # after the diagnostic reasons: the band takes the rows nothing sharper explained
    mark(in_band, "amplitude_ambiguous")
    mark(np.ones(len(scored), dtype=bool), "model_split")

    scored = scored.copy()
    scored["state"] = [CLASS_NAME.get(v) if np.isfinite(v) else None
                       for v in np.where(covered, guess, np.nan)]
    scored["label"] = np.where(covered, guess, np.nan)
    scored["confidence"] = np.round(conf, 4)
    scored["ambiguous"] = ambiguous
    # its own column, not implicit in `reason`: a banded row can still be attributed to a sharper one
    scored["in_band"] = in_band
    scored["reason"] = reason
    scored["reason_detail"] = [REASONS[r][0] if r else "" for r in reason]
    scored["alternative"] = [REASONS[r][1] if r else "" for r in reason]
    # a recording that never rests has a fallback zero, so every interleg feature is weaker evidence
    if not bool(scored["rest_trusted"].iloc[0]):
        scored["reason_detail"] = scored["reason_detail"].astype(str) + (
            " | recording never rests: interleg zero is a fallback median, not a measured "
            "standing posture")
    return scored


# (INPUT rows + label columns, provenance); provenance travels with the frame, never just printed
def label_csv(path: Path, model_dir: Path, threshold: float) -> tuple[pd.DataFrame, dict]:
    model, meta = load_champion(model_dir)
    spec = WindowSpec(window_s=meta["window_s"],
                      stride_s=meta.get("inference_stride_s", 0.25),
                      fs_hz=meta.get("fs_hz", CANONICAL_HZ))
    raw, view, provenance = read_source(path)

    # the same normalization the model was trained under; anything else is train/serve skew
    frame, _segments = resample_file(view, TIME_COL)
    if frame.empty:
        raise SystemExit(f"{path.name}: no usable segment survived resampling")

    # S1's label-free gate, in FRONT of the model: a dead channel would otherwise score confidently
    h = health(frame, window_s=spec.window_s, fs_hz=spec.fs_hz)
    for w in h.warnings:
        print(f"[label] warning: {w}")
    if not h.usable:
        raise SystemExit(f"{path.name}: not usable\n  "
                         + "\n  ".join(h.errors))

    scored = explain(score_frame(model, meta, frame), meta, threshold, spec)

    # file-level bounds over `scored`, the rows actually scored; it REPORTS, refusing nothing
    provenance["plausibility"] = plausibility_check(scored)

    # Map the grid back onto the caller's own rows by nearest time
    t_grid = scored[TIME_COL].to_numpy(float)
    t_in = raw[TIME_COL].to_numpy(float)
    order = np.argsort(t_grid)
    tg = t_grid[order]
    j = np.searchsorted(tg, t_in).clip(1, tg.size - 1)
    nearest = order[np.where(np.abs(t_in - tg[j - 1]) <= np.abs(tg[j] - t_in), j - 1, j)]
    # Beyond half a grid step from any scored sample, the row sits in a gap
    gap = np.abs(t_in - t_grid[nearest]) > (1000.0 / spec.fs_hz)

    cols = ["state", "label", "confidence", "ambiguous", "reason", "reason_detail",
            "alternative", "n_windows"]
    picked = scored.iloc[nearest][cols].reset_index(drop=True)
    picked.loc[gap, ["state", "label", "confidence"]] = None
    picked.loc[gap, "ambiguous"] = True
    picked.loc[gap, "reason"] = "uncovered"
    picked.loc[gap, "reason_detail"] = REASONS["uncovered"][0]
    picked.loc[gap, "alternative"] = REASONS["uncovered"][1]

    # the integer-coded guess, derived here so every output shape inherits one definition of it
    picked.insert(picked.columns.get_loc("label") + 1, GUESS_COL,
                  np.where(pd.to_numeric(picked["label"], errors="coerce").isna(),
                           LABEL_UNKNOWN_MACHINE, picked["label"]).astype(int))

    # on the raw path the derived channels ride along, or the verdict is over columns nobody can see
    parts = [raw.reset_index(drop=True)]
    if provenance["derived"]:
        parts.append(view[list(FEATURES)].reset_index(drop=True))
    parts.append(picked)
    return pd.concat(parts, axis=1), provenance


# the golden corpus's shape- these six, in order, so a labelled file reads like an annotated one
GOLDEN_COLUMNS = (TIME_COL, *FEATURES, LABEL_COL)

# the model's call on every scored row; Label overwrites the ambiguous ones with -1
GUESS_COL = "guess"

# appended AFTER the six, so the golden prefix stays positionally intact for a reader expecting it
VERDICT_COLUMNS = ("confidence", "ambiguous", "reason")


# the six, then the verdict; -1 is scored-but-unsure, 255 is uncovered- different facts
def verdict_frame(out: pd.DataFrame, *, strict: bool = False) -> pd.DataFrame:
    lab = pd.to_numeric(out["label"], errors="coerce").to_numpy(float)
    amb = out["ambiguous"].fillna(True).astype(bool).to_numpy()
    # `lab` is already the guess wherever the row was scored; `Label` is that plus the one extra step
    code = np.where(np.isnan(lab), LABEL_UNKNOWN_MACHINE, np.where(amb, HUMAN_UNKNOWN, lab))
    frame = pd.DataFrame({
        TIME_COL: out[TIME_COL].to_numpy(float),
        **{c: out[c].to_numpy(float) for c in FEATURES},
        LABEL_COL: code.astype(int),
    })
    if not strict:
        frame[GUESS_COL] = out[GUESS_COL].to_numpy(int)
        frame["confidence"] = pd.to_numeric(out["confidence"], errors="coerce").to_numpy(float)
        frame["ambiguous"] = amb
        # "" on a confident row, so the column reads as "the doubt, where there was any"
        frame["reason"] = out["reason"].fillna("").astype(str).to_numpy()
    return frame


# the one-block answer to "what did the model actually read?"
def describe_source(provenance: dict) -> str:
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
    use_replacement_encoding()   # --help itself has an em-dash in it; see stages/console.py
    ap = argparse.ArgumentParser(description="Label a CSV with stand/walk + confidence.")
    ap.add_argument("csv", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--preset", default=None, help="high_coverage | balanced | high_precision")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--full", action="store_true",
                    help="write the complete labelled frame — every input column, the "
                         "derived features, and state/label/confidence/ambiguous/reason/"
                         "reason_detail/alternative/n_windows")
    ap.add_argument("--no-verdict", action="store_true",
                    help=f"drop {GUESS_COL} and {', '.join(VERDICT_COLUMNS)}, leaving only "
                         f"the {len(GOLDEN_COLUMNS)} columns the annotated corpus uses; "
                         f"abstention then survives in Label alone "
                         f"({HUMAN_UNKNOWN} unsure, {LABEL_UNKNOWN_MACHINE} uncovered)")
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
        # an abstention, not a crash: a labelled file the pipeline cannot defend is worse than none
        raise SystemExit(f"{args.csv.name}: ABSTAINED — {type(exc).__name__}\n  {exc}")

    print(describe_source(provenance))
    print()
    print(summarize(out, threshold))
    dest = args.out or args.csv.with_name(args.csv.stem + "_labelled.csv")
    emitted = out if args.full else verdict_frame(out, strict=args.no_verdict)
    emitted.to_csv(dest, index=False)
    print(f"-> {dest}")


if __name__ == "__main__":
    main()
