# score raw device logs with the champion: run the trained model on unlabeled data/raw recordings and
# write each back with a per-row `labeled` column. reuses the VERIFIED raw->features bridge (transform.py)
# and dense inference (predict.py), so serve matches train. read-only over data/raw; writes results/raw/.

# path per file: load raw by name -> raw_to_features (dt = last interval, matching the reference) ->
# resample to 100 Hz -> dense_predict per segment -> map labels back onto the native rows. rows the model
# cannot cover (too short, off-grid, outside a segment) carry MACHINE_UNKNOWN, never a guess.

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dataset_profile import MACHINE_UNKNOWN, STAND, WALK
from stages.s1_clean.config import CANONICAL_DT_MS
from stages.s1_clean.resample import resample_file
from stages.s2_ml.features import WindowSpec
from stages.s2_ml.predict import DEFAULT_INFERENCE_STRIDE_S, dense_predict_segment
from stages.s2_ml.transform import (
    FEATURE_COLUMNS,
    TRUST_UNCHECKED,
    matlab_dt,
    raw_to_features,
    safe_dt,
)
from stages.s2_ml.verify_transform import load_raw_by_name

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
S2_OUT_DIR = REPO_ROOT / "runs" / "s2_ml"
RESULTS_DIR = REPO_ROOT / "results" / "raw"

LABEL_OUT_COL = "labeled"
# a native row keeps a 100 Hz prediction only within this much of a grid point; further is no coverage
MATCH_TOL_MS = CANONICAL_DT_MS


# the champion model + the feature list and window it was trained with
def load_champion(s2_dir: Path = S2_OUT_DIR) -> tuple[object, list[str], WindowSpec]:
    import joblib

    meta = json.loads((s2_dir / "model_meta.json").read_text(encoding="utf-8"))
    model = joblib.load(s2_dir / "champion.joblib")
    spec = WindowSpec(window_s=meta["window_s"], stride_s=meta["window_s"], fs_hz=meta["fs_hz"])
    return model, list(meta["features"]), spec


# every raw csv under data/raw (sorted, session folders included)
def find_raw(raw_dir: Path = RAW_DIR) -> list[Path]:
    return sorted(raw_dir.rglob("*.csv"))


# 100 Hz predictions for one recording as (grid_time_ms, grid_label), stacked across usable segments.
# None when the file has no usable segment (all native rows then read as no-coverage)
def predict_grid(feats_native: pd.DataFrame, model, feat_names: list[str],
                 spec: WindowSpec, stride_s: float) -> tuple[np.ndarray, np.ndarray] | None:
    grid, _segs = resample_file(feats_native, "Time")
    if grid.empty:
        return None
    times, labels = [], []
    for _seg_id, seg in grid.groupby("segment", sort=True):
        seg = seg.reset_index(drop=True)
        pred = dense_predict_segment(model, seg, feat_names, spec, stride_s)
        keep = pred != -1  # -1 = segment too short to score
        if keep.any():
            times.append(seg["Time"].to_numpy(float)[keep])
            labels.append(pred[keep])
    if not times:
        return None
    t = np.concatenate(times)
    order = np.argsort(t)
    return t[order], np.concatenate(labels)[order]


# nearest grid label for each native row; rows past MATCH_TOL_MS from any grid point are no-coverage
def map_to_native(native_t: np.ndarray, grid: tuple[np.ndarray, np.ndarray] | None) -> np.ndarray:
    out = np.full(native_t.shape, MACHINE_UNKNOWN, dtype=int)
    if grid is None:
        return out
    gt, gp = grid
    idx = np.clip(np.searchsorted(gt, native_t), 1, gt.size - 1)
    left_closer = np.abs(native_t - gt[idx - 1]) <= np.abs(gt[idx] - native_t)
    nn = np.where(left_closer, idx - 1, idx)
    covered = np.abs(native_t - gt[nn]) <= MATCH_TOL_MS
    out[covered] = gp[nn][covered]
    return out


# one recording -> per-native-row labels; skip_reason set (labels all no-coverage) when it cannot bridge
def score_file(path: Path, model, feat_names: list[str], spec: WindowSpec,
               stride_s: float) -> tuple[np.ndarray, str | None]:
    raw = load_raw_by_name(str(path))
    needed = ["Time", *[f"{s}_Deg_Y" for s in "LR"], *[f"{s}_Gyro_Z" for s in "LR"]]
    missing = [c for c in needed if c not in raw.columns]
    native_t = raw["Time"].to_numpy(float) if "Time" in raw.columns else np.empty(0)
    if missing:
        return np.full(len(raw), MACHINE_UNKNOWN, dtype=int), f"missing channels: {missing}"

    dt = matlab_dt(native_t)  # last interval, reproducing the training features exactly
    if not np.isfinite(dt) or dt <= 0:  # broken clock (2026-05 duplicate/backward timestamps)
        dt = safe_dt(native_t)
    if not np.isfinite(dt) or dt <= 0:
        return np.full(len(raw), MACHINE_UNKNOWN, dtype=int), "degenerate time base"

    feats = raw_to_features(raw, trust=TRUST_UNCHECKED, dt_s=dt)
    grid = predict_grid(feats, model, feat_names, spec, stride_s)
    return map_to_native(native_t, grid), None


# session folder a raw file sits under (data/raw/<session>/<file>.csv); flat files go to _loose
def session_of(path: Path, raw_dir: Path) -> str:
    rel = path.relative_to(raw_dir)
    return rel.parts[0] if len(rel.parts) > 1 else "_loose"


# score every raw csv, mirror it to results/raw/<session>/ with the `labeled` column appended
def run(raw_dir: Path = RAW_DIR, out_dir: Path = RESULTS_DIR,
        s2_dir: Path = S2_OUT_DIR, stride_s: float = DEFAULT_INFERENCE_STRIDE_S) -> dict:
    model, feat_names, spec = load_champion(s2_dir)
    paths = find_raw(raw_dir)
    print(f"[score] {len(paths)} raw files -> labeling with the champion "
          f"(window {spec.window_s}s, dense stride {stride_s * 1000:.0f} ms)")

    written, skipped = 0, []
    totals = {STAND: 0, WALK: 0, MACHINE_UNKNOWN: 0}
    for i, p in enumerate(paths, 1):
        try:
            labels, skip = score_file(p, model, feat_names, spec, stride_s)
        except Exception as exc:  # a bad file must not abort the batch; mark it no-coverage
            orig = pd.read_csv(p, encoding="utf-8-sig")
            labels, skip = np.full(len(orig), MACHINE_UNKNOWN, dtype=int), \
                f"{type(exc).__name__}: {exc}"
        orig = pd.read_csv(p, encoding="utf-8-sig")
        orig = orig.loc[:, [c for c in orig.columns if not c.startswith("Unnamed")]]
        if len(labels) != len(orig):  # resolver dropped/added rows; fall back to no-coverage
            labels = np.full(len(orig), MACHINE_UNKNOWN, dtype=int)
            skip = skip or "row count mismatch after resolve"
        orig[LABEL_OUT_COL] = labels

        dest = out_dir / session_of(p, raw_dir) / p.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        orig.to_csv(dest, index=False, encoding="utf-8")
        written += 1
        for code in totals:
            totals[code] += int((labels == code).sum())
        if skip:
            skipped.append((p.name, skip))
        n_lab = int(((labels == STAND) | (labels == WALK)).sum())
        print(f"[score] {i}/{len(paths)} {p.name} -> {n_lab}/{len(labels)} rows labeled"
              f"{' !! ' + skip if skip else ''}")

    print(f"[score] {written} files -> {out_dir}")
    print(f"[score] rows: stand {totals[STAND]:,}, walk {totals[WALK]:,}, "
          f"no-coverage {totals[MACHINE_UNKNOWN]:,}")
    if skipped:
        print(f"[score] {len(skipped)} file(s) with no/partial coverage:")
        for name, why in skipped:
            print(f"[score]   {name}: {why}")
    return {"written": written, "totals": totals, "skipped": skipped}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Label raw device logs (data/raw) with the champion; write results/raw/ with a "
                    "per-row `labeled` column.")
    ap.add_argument("--raw", type=Path, default=RAW_DIR)
    ap.add_argument("--out", type=Path, default=RESULTS_DIR)
    ap.add_argument("--stride-s", type=float, default=DEFAULT_INFERENCE_STRIDE_S,
                    help="dense inference stride in seconds")
    args = ap.parse_args()
    run(args.raw, args.out, stride_s=args.stride_s)


if __name__ == "__main__":
    main()
