"""Per-recording REST CALIBRATION: the subject's own standing posture, measured.

DOMAIN_NOTES §7 is blunt about this — *per-file calibration, never corpus-wide*. A fixed
global threshold scores `walk_rec = 0.000` on rev8 while per-file calibration scores
1.000 on the same file. More data yields a better global constant, and global is the
disease. §10.2 is what makes the cure possible: recordings begin at rest, so every file
carries a standing reference measured on the same person, sensor and mounting, minutes
earlier.

This module holds the signal PRIMITIVE — the swap count and the rest-span search. The
swap RULE (verdict bands, adaptive span, the grow fallback) is S3's, in
`stages/s3_physics/anchors.py`. The split is not cosmetic: S3 imports S2, so defining
the primitive in S3 and reading it here would close an import cycle the moment anything
in S2 needs a rest zero.

Nothing here reads a label, by construction. That is what lets the calibration run on
`data/raw` recordings that have no ground truth, and what keeps the lockbox sealed —
a rest zero measured from the signal cannot leak an answer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stages.s1_clean.config import CANONICAL_HZ
from stages.s2_ml.dataset import FEATURES

# Sensor noise gate (§10): 5x the measured standing noise came out 0.76-0.88 deg on
# three files independently, rounded to 1.0. NOT fitted and not tunable — a swap has to
# clear real motion rather than jitter. The rule's zero-fitted-parameter claim rests on
# this number being a noise floor, so moving it to improve a score would be fitting.
SWAP_DELTA_DEG = 1.0

# Opening-rest window for the per-file zero (§10.2). A per-subject DC offset can hold
# L-R above -delta through real gait, which would silently suppress every swap;
# subtracting the rest median recenters it.
REST_ANCHOR_S = 3.0

# Posture lives on the ANGLE channels; the rate channels have none.
ANGLE_CHANNELS = (FEATURES[0], FEATURES[1])  # L_ang_LPF, R_ang_LPF


def swap_count(d: np.ndarray, delta: float = SWAP_DELTA_DEG) -> int:
    """Interleg alternations: commits past +delta, then past -delta, with hysteresis.

    Standing measures 0, a single weight shift 1, a stride >=2 — on every file across a
    4x amplitude range (§10), which is why `1` is the only integer between the classes
    and not a chosen threshold.
    """
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


def interleg(frame: pd.DataFrame) -> np.ndarray:
    """L_ang - R_ang: the signal the swap rule reads. Walking is the legs *swapping*."""
    return (frame[ANGLE_CHANNELS[0]].to_numpy(float)
            - frame[ANGLE_CHANNELS[1]].to_numpy(float))


def is_rest(d: np.ndarray, min_n: int) -> bool:
    """Is this span actually at rest? The swap rule's own STANDING verdict, locally centered.

    Parameter-free on purpose: testing rest with a *different* criterion than the one the
    rule uses would let a span count as rest for calibration and as motion for scoring.

    `min_n` is required rather than defaulted — the verdict is length-dependent and a
    short span passes far too easily.
    """
    return d.size >= min_n > 0 and swap_count(d - float(np.median(d)), SWAP_DELTA_DEG) == 0


def rest_span_frame(frame: pd.DataFrame, span: int) -> pd.DataFrame | None:
    """The stillest genuine-rest slice anywhere in the recording, or None if it never rests.

    Searched *within a segment only* — a span straddling a gap would average across time
    that was never recorded (§3.1). Quarter-span hops, so rest that straddles a block
    boundary is still found.
    """
    step = max(1, span // 4)
    best_ptp, best = np.inf, None
    for _seg_id, seg in frame.groupby("segment", sort=True):
        d = interleg(seg)
        for a in range(0, d.size - span + 1, step):
            s = d[a:a + span]
            if is_rest(s, span) and (p := float(np.ptp(s))) < best_ptp:
                best_ptp, best = p, seg.iloc[a:a + span]
    return best


def stillest_rest_offset(frame: pd.DataFrame, span: int) -> float | None:
    """Median interleg of the stillest rest span — this subject's mounting bias (§10.2)."""
    best = rest_span_frame(frame, span)
    return None if best is None else float(np.median(interleg(best)))


def rest_anchor(frame: pd.DataFrame, fs: float = CANONICAL_HZ,
                fallback_reference: float | None = None) -> tuple[float, bool]:
    """(interleg zero, trusted) for one recording.

    Preference order, most to least direct evidence:
      1. the opening `REST_ANCHOR_S`, when the swap rule calls it standing (§10.2 — the
         common case, and the one measured on this subject minutes before the trial)
      2. the stillest rest span found anywhere else in the recording
      3. a session-level reference from a sibling trial, when this one never rests
      4. the whole-recording median, returned UNTRUSTED

    Step 4 returns rather than raises: a recording that never rests still has to yield
    features, and the flag says the zero is a fallback. Detect-don't-assert, the same
    stance S1's channel trust takes when a file is too static to answer (§4.1b).
    """
    if frame.empty:
        return 0.0, False
    span = int(round(REST_ANCHOR_S * fs))
    seg0 = frame[frame["segment"] == frame["segment"].min()]
    d_open = interleg(seg0.iloc[:span])
    if is_rest(d_open, span):
        return float(np.median(d_open)), True
    found = stillest_rest_offset(frame, span)
    if found is not None:
        return found, True
    if fallback_reference is not None:
        return fallback_reference, True
    return float(np.median(interleg(frame))), False
