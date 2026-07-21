# regression guard for the raw -> rev2 bridge
# the model trains on labeled rev2 features but runs on raw CSVs; if raw_to_features stops
# reproducing the labeled columns, that's silent train/serve skew nothing else would catch.
# pairs annotated trials with their raw source and asserts reproduction to float roundoff.
# run after touching transform.py. see README.

from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from stages.s1_clean.census import read_header, strip_prefix
from stages.s1_clean.config import DOCUMENTED_GYRO_PERMUTATION
from stages.s2_ml.dataset import TIME_COL, _read_raw, find_trials
from stages.s2_ml.transform import (
    SAGITTAL_DEG_AXIS,
    FC_ANG_HZ,
    FC_ANGVEL_HZ,
    FEATURE_COLUMNS,
    TRUST_UNCHECKED,
    lpf,
    matlab_dt,
    raw_to_features,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

# float roundoff on ~1e5-sample IIR recursions; larger means the transform drifted
# from the MATLAB it mirrors
TOLERANCE = 1e-9


# raw device CSV with columns resolved BY NAME (never by position, §1.3)
def load_raw_by_name(path: str) -> pd.DataFrame:
    names = [strip_prefix(c) for c in read_header(Path(path))]
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df.loc[:, [c for c in df.columns if not c.startswith("Unnamed")]]
    df.columns = names[: len(df.columns)]
    return df


# index raw files by path -> (t0, t1, nrows, df)
def index_raw_files(raw_glob: str = "data/raw/**/*.csv") -> dict[str, tuple]:
    index = {}
    for f in sorted(glob.glob(str(REPO_ROOT / raw_glob), recursive=True)):
        try:
            df = load_raw_by_name(f)
            t = df["Time"].to_numpy(float)
            index[f] = (t[0], t[-1], len(t), df)
        except Exception:
            continue # unreadable/ragged files are S1's problem, not the bridge's
    return index


# annotated trials whose raw source is present, matched on the time vector
def find_pairs(index: dict[str, tuple]) -> list[tuple[Path, str, pd.DataFrame]]:
    pairs = []
    for p in find_trials():
        t = _read_raw(p)[TIME_COL].to_numpy(float)
        for f, (r0, r1, nr, df) in index.items():
            if abs(t[0] - r0) < 50 and abs(t[-1] - r1) < 5000 and nr == len(t):
                pairs.append((p, f, df))
                break
    return pairs


# max abs error reproducing the labeled features from ONE Deg axis (+ its permuted gyro);
# diagnostic only -- names the axis when Y doesn't reproduce, so an override is
# distinguishable from a real transform bug
def _axis_error(raw: pd.DataFrame, ann: pd.DataFrame, dt: float, deg_axis: str) -> float:
    gy = DOCUMENTED_GYRO_PERMUTATION[deg_axis]
    errs = []
    for side in ("L", "R"):
        errs.append(np.max(np.abs(lpf(raw[f"{side}_Deg_{deg_axis}"].to_numpy(float), dt, FC_ANG_HZ)
                                   - ann[f"{side}_ang_LPF"].to_numpy(float))))
        errs.append(np.max(np.abs(lpf(raw[f"{side}_Gyro_{gy}"].to_numpy(float), dt, FC_ANGVEL_HZ)
                                   - ann[f"{side}_angvel_LPF"].to_numpy(float))))
    return float(max(errs))


# pair recordings, reproduce the labeled features, classify each as verified/override/broken
def main() -> None:
    pairs = find_pairs(index_raw_files())
    if not pairs:
        raise SystemExit("no paired recordings found — cannot verify the bridge")

    worst, verified, overrides, broken = 0.0, [], [], []
    for ann_path, _raw_path, raw in pairs:
        ann = pd.read_csv(ann_path)
        ann.columns = [c.strip() for c in ann.columns]
        dt = matlab_dt(raw["Time"].to_numpy(float))
        # TRUST_UNCHECKED on purpose: this verifies the transform MATH on MATLAB fixtures;
        # the per-file axis check is a provenance guard that would refuse its own test data
        feat = raw_to_features(raw, trust=TRUST_UNCHECKED, dt_s=dt)
        err_y = max(float(np.max(np.abs(feat[c].to_numpy() - ann[c].to_numpy(float))))
                    for c in FEATURE_COLUMNS)
        if err_y <= TOLERANCE:
            verified.append((ann_path.name, err_y))
            worst = max(worst, err_y)
            continue
        # Y doesn't reproduce: if some other single axis does, it's an accepted OVERRIDE
        # (labeled on that axis, we read Y by policy §6.3); matching no axis is a real break
        others = {ax: _axis_error(raw, ann, dt, ax)
                  for ax in "XYZ" if ax != SAGITTAL_DEG_AXIS}
        best = min(others, key=others.get)
        bucket = overrides if others[best] <= TOLERANCE else broken
        bucket.append((ann_path.name, best, err_y, others[best]))

    for name, err in sorted(verified, key=lambda r: -r[1]):
        print(f"  {name:<34}  Y  max_err={err:.3g}")
    for name, ax, err_y, err_ax in overrides:
        print(f"  {name:<34}  OVERRIDE->Y  (labeled on Deg_{ax}={err_ax:.2g}; "
              f"read as Y by policy)")
    for name, ax, err_y, err_ax in broken:
        print(f"  {name:<34}  BROKEN  (matches no axis; Y={err_y:.3g}, "
              f"best Deg_{ax}={err_ax:.3g})")

    print(f"\n{len(verified)} Y-plane pairs verified, {len(overrides)} override-to-Y "
          f"(labeled on another axis, expected), {len(broken)} broken")
    print(f"worst Y-plane error: {worst:.3g}  (tolerance {TOLERANCE:g})")
    if broken or worst > TOLERANCE:
        print("FAIL: the bridge no longer reproduces the labeled features on the Y plane.")
        sys.exit(1)
    print("PASS: raw -> Y-plane reproduction is exact to float roundoff; overrides accounted for.")


if __name__ == "__main__":
    main()
