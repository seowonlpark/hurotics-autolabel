# per-recording REST CALIBRATION: the subject's own standing posture, measured
# per-file, NEVER corpus-wide; more data buys a better global constant, and global
# is the disease
# the signal PRIMITIVE only- the swap RULE is S3's, in anchors.py
# nothing here reads a label, which is what keeps the lockbox sealed

from __future__ import annotations

import numpy as np
import pandas as pd

from stages.s2_ml.dataset import FEATURES

# sensor noise gate: 5x measured standing noise came out 0.76-0.88 deg on three files
# independently, rounded to 1.0; NOT fitted and not tunable- the zero-fitted-parameter
# claim rests on this being a noise floor, so moving it to improve a score is fitting
SWAP_DELTA_DEG = 1.0

# opening-rest window for the per-file zero; a per-subject DC offset can hold L-R above
# -delta through real gait and silently suppress every swap, so recenter on the median
REST_ANCHOR_S = 3.0

# posture lives on the ANGLE channels; the rate channels have none
ANGLE_CHANNELS = (FEATURES[0], FEATURES[1])  # L_ang_LPF, R_ang_LPF


# interleg alternations: commits past +delta then past -delta, with hysteresis
# standing measures 0, a weight shift 1, a stride >=2, on every file across a 4x
# amplitude range- which is why 1 is the only integer between the classes, not a threshold
def swap_count(d: np.ndarray, delta: float = SWAP_DELTA_DEG) -> int:
    commits: list[int] = []
    state = 0  # last committed side: +1, -1, or 0 (uncommitted)
    for x in d:
        if x > delta and state != 1:
            commits.append(1)
            state = 1
        elif x < -delta and state != -1:
            commits.append(-1)
            state = -1
    return max(0, len(commits) - 1)  # commits alternate by construction => swaps = len-1


# L_ang- R_ang, the signal the swap rule reads; walking is the legs SWAPPING
def interleg(frame: pd.DataFrame) -> np.ndarray:
    return (frame[ANGLE_CHANNELS[0]].to_numpy(float)
            - frame[ANGLE_CHANNELS[1]].to_numpy(float))


# the swap rule's own STANDING verdict, locally centered; parameter-free on purpose,
# since a different criterion would let a span be rest for calibration and motion for
# scoring; min_n is required, not defaulted- a short span passes far too easily
def is_rest(d: np.ndarray, min_n: int) -> bool:
    return d.size >= min_n > 0 and swap_count(d - float(np.median(d)), SWAP_DELTA_DEG) == 0


# stillest genuine-rest slice in the recording, or None if it never rests; searched
# WITHIN a segment only, since a span straddling a gap averages across time that was
# never recorded; quarter-span hops so rest on a block boundary is still found
def rest_span_frame(frame: pd.DataFrame, span: int) -> pd.DataFrame | None:
    step = max(1, span // 4)
    best_ptp, best = np.inf, None
    for _seg_id, seg in frame.groupby("segment", sort=True):
        d = interleg(seg)
        for a in range(0, d.size - span + 1, step):
            s = d[a:a + span]
            if is_rest(s, span) and (p := float(np.ptp(s))) < best_ptp:
                best_ptp, best = p, seg.iloc[a:a + span]
    return best


# the rest ZERO is NOT derived here; features.rest_reference is the single implementation,
# because the per-side postures and the interleg offset must come from the SAME span- a
# second copy of this search once lived here and had drifted to a different preference order
