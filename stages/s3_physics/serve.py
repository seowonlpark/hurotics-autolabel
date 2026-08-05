# the swap rule on the SERVE grid: label_audit's rule, pointed at a raw recording with no labels

from __future__ import annotations

import numpy as np

from stages.s2_ml.dataset import FEATURES
from stages.s2_ml.features import WindowSpec
from stages.s3_physics.anchors import AMBIGUOUS, STANDING, WALKING, adaptive_verdict

# row-level bands on the swap count AVERAGED over covering windows, the arithmetic label.py uses
ROW_STANDING_MAX = 0.5
ROW_WALKING_MIN = 1.5


# STANDING / AMBIGUOUS / WALKING per row, from the mean adaptive swap count
def row_verdict(mean_swaps: np.ndarray) -> np.ndarray:
    out = np.full(mean_swaps.shape, AMBIGUOUS, dtype=object)
    with np.errstate(invalid="ignore"):
        out[mean_swaps < ROW_STANDING_MAX] = STANDING
        out[mean_swaps >= ROW_WALKING_MIN] = WALKING
    out[~np.isfinite(mean_swaps)] = AMBIGUOUS
    return out


# (swaps, span_s, starts) for every window of one gap-free segment
def segment_verdicts(chan: dict[str, np.ndarray], spec: WindowSpec,
                     ileg_zero: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    l = chan[FEATURES[0]]
    r = chan[FEATURES[1]]
    d = (l - r) - ileg_zero
    n_windows = max(0, (len(d) - spec.n) // spec.step + 1)
    starts = np.arange(n_windows) * spec.step

    swaps = np.empty(n_windows, dtype=float)
    span_s = np.empty(n_windows, dtype=float)
    for i, start in enumerate(starts):
        # `start + spec.n // 2` is the cell centre, the same one `trial_anchors` passes
        _verdict, sp, sw, _a, _b = adaptive_verdict(d, l, r, int(start) + spec.n // 2, spec)
        swaps[i] = sw
        span_s[i] = sp
    return swaps, span_s, starts


__all__ = ["AMBIGUOUS", "STANDING", "WALKING", "row_verdict", "segment_verdicts",
           "ROW_STANDING_MAX", "ROW_WALKING_MIN"]
