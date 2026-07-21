# S3 anchor features: the physics view of a window, model-free by construction
# the centerpiece is the SWAP RULE (§10) -- zero fitted parameters, validated across a 4x
# amplitude range: walking is the legs alternating, counted as sign-committed crossings of
# L_ang - R_ang. around it sit the two validated descriptors (§10.1) and the five PLAN
# anchors that S3 must give a rate-invariance verdict for. an anchor is a *claim about the
# body*; the rate audit (rate_audit.py) disposes of the ones that are really claims about
# the sampling grid -- gyro_energy already failed once (id=69) and is defined here the way
# that fails, on purpose, so the audit re-derives it. see README / DOMAIN_NOTES §10.

from __future__ import annotations

import numpy as np
import pandas as pd

from stages.s1_clean.config import CANONICAL_HZ
from stages.s2_ml.features import GAIT_BAND_HZ, WindowSpec, iter_windows
from stages.s2_ml.dataset import FEATURES, Trial

# the sensor noise floor (§10): 5x the measured 0.76-0.88 deg standing noise. NOT fitted --
# a swap must clear real motion, not jitter. do not tune this; the rule's whole claim is
# that nothing here is tuned.
SWAP_DELTA_DEG = 1.0

# the five anchors PLAN.md S3 requires a rate-invariance verdict for. the swap rule and the
# §10.1 descriptors are reported alongside but are not in this set -- they are already
# validated body descriptors, not candidate anchors under audit.
ANCHOR_NAMES = ("periodicity", "antiphase", "grav_stab", "gait_hz", "gyro_energy")

# swap-rule verdict bands (§10): 0 -> STANDING, 1 -> AMBIGUOUS, >=2 -> WALKING
STANDING, AMBIGUOUS, WALKING = "STANDING", "AMBIGUOUS", "WALKING"


# number of times the interleg signal commits to one side past +delta and then to the other
# past -delta (§10). hysteresis at +/-delta: a commit only counts once the signal clears the
# floor, and the next commit must be to the OTHER side. the count is the number of such
# alternations -- standing measures 0, a single weight shift 1, a stride >=2.
def swap_count(d: np.ndarray, delta: float = SWAP_DELTA_DEG) -> int:
    commits: list[int] = []
    state = 0 # last committed side: +1, -1, or 0 (uncommitted)
    for x in d:
        if x > delta and state != 1:
            commits.append(1); state = 1
        elif x < -delta and state != -1:
            commits.append(-1); state = -1
    return max(0, len(commits) - 1) # commits alternate by construction => swaps = len-1


# the swap rule's discrete verdict (§10). '1' is genuinely ambiguous, not a fudge: one leg
# passing the other happens both when you step and when you shift your weight.
def swap_verdict(count: int) -> str:
    if count == 0:
        return STANDING
    if count == 1:
        return AMBIGUOUS
    return WALKING


# pearson r; 0 when either series is constant or too short (matches S2 _corr)
def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


# dominant frequency of x inside the gait band, in Hz; 0 when there is no band power.
# resolution is 1/duration, so at a 2 s window the low edge is unresolvable (§9 tradeoff).
def _dominant_hz(x: np.ndarray, fs: float) -> float:
    x = x - x.mean()
    if x.size < 4 or not np.any(x):
        return 0.0
    power = np.abs(np.fft.rfft(x * np.hanning(x.size))) ** 2
    freq = np.fft.rfftfreq(x.size, 1.0 / fs)
    band = (freq >= GAIT_BAND_HZ[0]) & (freq <= GAIT_BAND_HZ[1])
    if not band.any() or power[band].sum() <= 0:
        return 0.0
    return float(freq[band][np.argmax(power[band])])


# rhythm STRENGTH, amplitude-independent: the tallest normalized-autocorrelation peak at a
# non-zero lag whose period is a plausible stride (up to half the window, so at least two
# cycles are seen). 1.0 = perfectly periodic, ~0 = no rhythm. KNOWN to fail at cadence
# CHANGES, not at arrhythmia (§11.2) -- a within-window property, so the agent must read it
# as "steady rhythm here", never "the subject has a rhythm".
def _periodicity(x: np.ndarray, fs: float) -> float:
    x = x - x.mean()
    n = x.size
    if n < 8 or x.std() == 0:
        return 0.0
    ac = np.correlate(x, x, mode="full")[n - 1:]
    ac = ac / ac[0] # lag 0 == 1
    lo = max(1, int(round(fs / GAIT_BAND_HZ[1]))) # shortest plausible stride period
    hi = min(n - 1, n // 2) # need >=2 cycles in-window
    if hi <= lo:
        return 0.0
    return float(np.clip(ac[lo:hi + 1].max(), 0.0, 1.0))


# every physics anchor + descriptor for one window. d = L_ang - R_ang is the interleg signal
# the swap rule reads. returns a flat dict; the five ANCHOR_NAMES keys are what the rate
# audit re-checks, the rest are the validated §10 / §10.1 descriptors reported alongside.
def window_anchors(win: pd.DataFrame, fs: float = CANONICAL_HZ) -> dict[str, float | int | str]:
    l_ang = win[FEATURES[0]].to_numpy(float) # L_ang_LPF
    r_ang = win[FEATURES[1]].to_numpy(float) # R_ang_LPF
    l_vel = win[FEATURES[2]].to_numpy(float) # L_angvel_LPF
    r_vel = win[FEATURES[3]].to_numpy(float) # R_angvel_LPF
    d = l_ang - r_ang

    # --- the swap rule (§10): the validated, zero-parameter walk/stand call ---
    swaps = swap_count(d, SWAP_DELTA_DEG)

    # --- validated descriptors (§10.1) ---
    half = d.size // 2
    ileg_minhalf = (min(float(np.ptp(d[:half])), float(np.ptp(d[half:])))
                    if half >= 1 else 0.0) # both halves must move, not one big excursion
    interleg_offset = float(np.median(d)) # posture: which leg leads, feet-together vs split

    # --- the five PLAN anchors, each under rate-invariance audit ---
    # antiphase: legs swing in opposition (r<0) when walking. NECESSARY but not sufficient
    # (§6.2): every projection of a planar swing is antiphase. reported as -r so >0 == more
    # antiphase.
    antiphase = -_corr(l_ang, r_ang)
    # grav_stab: steadiness of the gravity-referenced tilt (Deg_Y is a fused estimate, §4.6).
    # ->1 when posture holds still (standing), ->0 when the tilt sweeps (walking).
    grav_stab = 1.0 / (1.0 + 0.5 * (float(l_ang.std()) + float(r_ang.std())))
    # gait_hz: cadence, from the swing angle's dominant gait-band frequency (mean of legs)
    gait_hz = 0.5 * (_dominant_hz(l_ang, fs) + _dominant_hz(r_ang, fs))
    # periodicity: rhythm strength (autocorrelation), read on the more complete leg signal
    periodicity = 0.5 * (_periodicity(l_ang, fs) + _periodicity(r_ang, fs))
    # gyro_energy: TOTAL rotational energy, summed (not averaged) over the window. this scales
    # with the sample count, so it is really a claim about the grid, not the body -- it failed
    # the rate audit at id=69 and is defined here the way that fails, on purpose (§2.3).
    gyro_energy = float(np.sum(l_vel ** 2) + np.sum(r_vel ** 2))

    return {
        "periodicity": periodicity,
        "antiphase": antiphase,
        "grav_stab": grav_stab,
        "gait_hz": gait_hz,
        "gyro_energy": gyro_energy,
        # descriptors reported alongside (not audited as anchors)
        "swap_count": int(swaps),
        "swap_verdict": swap_verdict(swaps),
        "ileg_minhalf": ileg_minhalf,
        "interleg_offset": interleg_offset,
    }


# per-window anchors for one trial as a table: window metadata (rev/trial/segment/t_start/
# human label) joined to every anchor + descriptor. reuses the S2 window iterator so the
# "never window across a gap" rule stays single-sourced.
def trial_anchors(trial: Trial, spec: WindowSpec | None = None) -> pd.DataFrame:
    spec = spec or WindowSpec()
    rows = [{**meta, **window_anchors(win, spec.fs_hz)}
            for meta, win in iter_windows(trial, spec)]
    return pd.DataFrame(rows)
