# per-recording REST CALIBRATION: the subject's own posture, per-file and NEVER corpus-wide

from __future__ import annotations

import numpy as np
import pandas as pd

from stages.s2_ml.dataset import FEATURES

# sensor noise gate: 5x measured standing noise on three files, rounded to 1.0; NOT fitted
SWAP_DELTA_DEG = 1.0

# opening-rest window for the per-file zero; a DC offset would otherwise suppress every swap
REST_ANCHOR_S = 3.0

# posture lives on the ANGLE channels; the rate channels have none
ANGLE_CHANNELS = (FEATURES[0], FEATURES[1])  # L_ang_LPF, R_ang_LPF


# interleg alternations past +/-delta with hysteresis; stand 0, weight shift 1, stride >=2
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


# the swap rule's own STANDING verdict, locally centered; min_n required- short spans pass easily
def is_rest(d: np.ndarray, min_n: int) -> bool:
    return d.size >= min_n > 0 and swap_count(d - float(np.median(d)), SWAP_DELTA_DEG) == 0


# stillest genuine-rest slice, or None; WITHIN a segment only, since a gap averages unrecorded time
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


# the rest ZERO is NOT derived here; features.rest_reference is the single implementation
