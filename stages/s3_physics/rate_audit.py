# S3 rate-invariance audit: the test that disposes of anchors (Section 11.3)
# an anchor is supposed to be a claim about the BODY. this decimates every window to half
# the rate through S1's anti-aliased filter, recomputes each anchor, and asks whether the
# number moved. a body-defined anchor (a cadence in Hz, a correlation, a posture) must not;
# an anchor that is really a claim about the sampling grid does. this is the machinery that
# failed gyro_energy at id=69 -- run here, it re-derives that verdict from scratch, and the
# PLAN S3 gate ("no anchor without a rate-invariance verdict") is this file's output.

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import decimate

from stages.s1_clean.config import CANONICAL_HZ, DECIMATE_FILTER
from stages.s2_ml.dataset import FEATURES, Trial
from stages.s2_ml.features import WindowSpec, iter_windows
from stages.s3_physics.anchors import ANCHOR_NAMES, WALKING, rest_offset, window_anchors

# halve the rate: a genuine bandwidth cut, not the 99.4-vs-100 timestamp quantization that
# is really the same rate (Section 2.2). if an anchor survives a real 2x decimation it is not a
# grid artifact.
AUDIT_FACTOR = 2

# a change above this => the anchor tracks the grid, not the body. read against absolute
# delta for the bounded anchors, relative delta for the ratio-scale ones (see ANCHOR_METRIC).
AUDIT_TOL = 0.10

# how each anchor's change is measured. a correlation or a [0,1] score lives on a BOUNDED
# scale where absolute delta is honest -- a rhythm strength of 0.60 vs 0.55 is "the same to a
# twentieth", and framing that as a relative change distorts it near zero. an energy lives on
# a RATIO scale where relative delta is honest -- 1000 vs 1100 deg^2/s^2 is a real 10% shift,
# and its absolute size depends on the amplitude of the bout. mixing the two is the Section 11.1 trap
# of a metric measuring the wrong thing.
ANCHOR_METRIC = {
    "periodicity": "abs", # normalized autocorr, [0, 1]
    "antiphase": "abs", # -pearson r, [-1, 1]
    "grav_stab": "abs", # stability score, (0, 1]
    "gyro_energy": "rel", # sum of deg^2/s^2, ratio scale
}

# denominator floor for the ratio-scale anchors only, so a near-zero native value cannot
# blow the relative delta up. chosen from each anchor's scale, not fitted.
ANCHOR_FLOOR = {"gyro_energy": 1.0}


# decimate one window's four channels to fs/factor with S1's anti-aliasing FIR, returning a
# window frame at the new rate. zero_phase keeps peaks where they are, so a frequency or a
# swap is not moved by the filter itself.
def decimate_window(win: pd.DataFrame, factor: int = AUDIT_FACTOR) -> tuple[pd.DataFrame, float]:
    cols = {c: decimate(win[c].to_numpy(float), factor, ftype=DECIMATE_FILTER, zero_phase=True)
            for c in FEATURES}
    return pd.DataFrame(cols), CANONICAL_HZ / factor


# change between the native and decimated value of one anchor, on the anchor's own scale:
# absolute for a bounded score/correlation, floored-relative for a ratio-scale magnitude
def _delta(anchor: str, native: float, decimated: float) -> float:
    native, decimated = float(native), float(decimated)
    if ANCHOR_METRIC[anchor] == "abs":
        return abs(native - decimated)
    return abs(native - decimated) / (abs(native) + ANCHOR_FLOOR[anchor])


# audit every anchor: recompute native vs half-rate on the WALKING windows the anchors are
# meant to describe, collect the per-window change, return one verdict per anchor. never
# raises on a moved anchor -- a rate-dependent verdict is the finding, as a replay DIFF is
# (Section 11.3).
#
# the audit is gated to the MOTION regime, identified by the validated swap rule itself
# (Section 10). these four anchors are descriptors of locomotion; their invariance claim is "a
# walking bout measured at 100 vs 50 Hz reads the same". a standing window has no cadence and
# no rhythm, so comparing that silence to its decimated self measures decimation noise, not
# rate-dependence, and would falsely condemn a good anchor -- the Section 11.1 "test that measures
# noise" trap. gating on the swap rule (independently validated, and not itself under audit)
# keeps this label-free.
def audit_anchors(trials: list[Trial], spec: WindowSpec | None = None,
                  factor: int = AUDIT_FACTOR, tol: float = AUDIT_TOL) -> dict[str, dict]:
    spec = spec or WindowSpec()
    deltas: dict[str, list[float]] = {a: [] for a in ANCHOR_NAMES}
    n_total = 0

    for trial in trials:
        center = rest_offset(trial, spec.fs_hz) # same per-file zero the swap rule uses (Section 10.4)
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
