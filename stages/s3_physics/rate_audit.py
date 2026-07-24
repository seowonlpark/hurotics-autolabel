# S3 rate-invariance audit (Section 11.3): decimate every window to half rate through S1's anti-aliased
# filter, recompute each anchor, and ask whether the number moved. a body-defined anchor must not; one
# that is really a claim about the sampling grid does. output is the PLAN S3 per-anchor verdict.

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import decimate

from stages.s1_clean.config import CANONICAL_HZ, DECIMATE_FILTER
from stages.s2_ml.dataset import FEATURES, Trial
from stages.s2_ml.features import WindowSpec, iter_windows
from stages.s3_physics.anchors import ANCHOR_NAMES, WALKING, rest_offset, window_anchors

# halve the rate: genuine bandwidth cut, not timestamp quantization at the same rate (Section 2.2)
AUDIT_FACTOR = 2

# change above this => anchor tracks the grid, not the body; scale per ANCHOR_METRIC
AUDIT_TOL = 0.10

# how each anchor's change is measured: absolute delta for bounded scores/correlations, relative delta
# for ratio-scale magnitudes. mixing the two measures the wrong thing (Section 11.1).
ANCHOR_METRIC = {
    "periodicity": "abs", # normalized autocorr, [0, 1]
    "antiphase": "abs", # -pearson r, [-1, 1]
    "grav_stab": "abs", # stability score, (0, 1]
    "gyro_energy": "rel", # sum of deg^2/s^2, ratio scale
}

# denominator floor for ratio-scale anchors so a near-zero native value can't blow the delta up
ANCHOR_FLOOR = {"gyro_energy": 1.0}


# decimate one window's four channels to fs/factor with S1's anti-aliasing FIR. zero_phase keeps
# peaks in place, so the filter itself moves no frequency or swap.
def decimate_window(win: pd.DataFrame, factor: int = AUDIT_FACTOR) -> tuple[pd.DataFrame, float]:
    cols = {c: decimate(win[c].to_numpy(float), factor, ftype=DECIMATE_FILTER, zero_phase=True)
            for c in FEATURES}
    return pd.DataFrame(cols), CANONICAL_HZ / factor


# change between native and decimated value of one anchor, on its own scale:
# absolute for bounded score/correlation, floored-relative for ratio-scale magnitude
def _delta(anchor: str, native: float, decimated: float) -> float:
    native, decimated = float(native), float(decimated)
    if ANCHOR_METRIC[anchor] == "abs":
        return abs(native - decimated)
    return abs(native - decimated) / (abs(native) + ANCHOR_FLOOR[anchor])


# audit every anchor: recompute native vs half-rate, return one verdict per anchor; never raises on a
# moved anchor. gated to WALKING windows (via the swap rule) since standing has no cadence to compare,
# and that gate keeps it label-free (Section 11.3).
def audit_anchors(trials: list[Trial], spec: WindowSpec | None = None,
                  factor: int = AUDIT_FACTOR, tol: float = AUDIT_TOL) -> dict[str, dict]:
    spec = spec or WindowSpec()
    deltas: dict[str, list[float]] = {a: [] for a in ANCHOR_NAMES}
    n_total = 0

    for trial in trials:
        center = rest_offset(trial, spec.fs_hz) # same per-file zero the swap rule uses (10.4)
        for _meta, win in iter_windows(trial, spec):
            native = window_anchors(win, spec.fs_hz, center)
            n_total += 1
            if native["swap_verdict"] != WALKING: # audit anchors only where they describe
                continue
            dec_win, dec_fs = decimate_window(win, factor)
            if len(dec_win) < 8: # too short to recompute anything meaningful
                continue
            decd = window_anchors(dec_win, dec_fs, center)
            for a in ANCHOR_NAMES:
                deltas[a].append(_delta(a, native[a], decd[a]))

    report: dict[str, dict] = {}
    for a in ANCHOR_NAMES:
        med = float(np.median(deltas[a])) if deltas[a] else float("nan")
        report[a] = {
            "metric": ANCHOR_METRIC[a],
            "median_delta": round(med, 4),
            "p90_delta": round(float(np.percentile(deltas[a], 90)), 4) if deltas[a] else None,
            "tol": tol,
            "verdict": "invariant" if med <= tol else "rate_dependent",
            "n_windows_total": n_total,
            "n_windows_walking": len(deltas[a]), # motion windows the verdict rests on
        }
    return report


# run the audit on the labeled corpus and print the verdict table
def main() -> None:
    from stages.s2_ml.dataset import load_dataset

    trials = [t for t in load_dataset() if t.split != "lockbox"] # lockbox sealed (Section 7)
    report = audit_anchors(trials)
    r0 = report[ANCHOR_NAMES[0]]
    print(f"[s3] rate-invariance audit @ {CANONICAL_HZ:.0f} Hz vs "
          f"{CANONICAL_HZ / AUDIT_FACTOR:.0f} Hz (tol {AUDIT_TOL:g}), "
          f"{r0['n_windows_walking']:,} walking / {r0['n_windows_total']:,} windows\n")
    print(f"  {'anchor':14} {'metric':>6} {'median delta':>9} {'p90 delta':>8}  verdict")
    for a in ANCHOR_NAMES:
        r = report[a]
        print(f"  {a:14} {r['metric']:>6} {r['median_delta']:>9.4f} {r['p90_delta']:>8.4f}  "
              f"{r['verdict']}")


if __name__ == "__main__":
    main()
