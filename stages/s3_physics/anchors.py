# S3 anchor features: a model-free physics view of one window, zero fitted parameters, no labels

from __future__ import annotations

import numpy as np
import pandas as pd

from stages.s1_clean.config import CANONICAL_HZ
from stages.s2_ml.dataset import FEATURES, TIME_COL, Trial
# _corr and rest_reference are shared with S2 deliberately: two copies would disagree about rest
from stages.s2_ml.features import GAIT_BAND_HZ, WindowSpec, _corr, rest_reference
from stages.s2_ml.rest import SWAP_DELTA_DEG, swap_count

# stride-adaptive span: standing keeps the base window, motion grows one, capped below a bout
MAX_SWAP_WINDOW_S = 6.0
ADAPTIVE_PERIODICITY_FLOOR = 0.35  # autocorr peak height to accept a cell as periodic

# span-grow fallback: re-count over the max span, accepting WALKING only if it GENUINELY alternates
GROW_MIN_PTP_DEG = 8.0     # both halves must swing this far: a real stride, not jitter
GROW_MIN_ANTIPHASE = 0.5   # -corr(L,R): alternation, not two isolated weight shifts

# the four anchors PLAN.md requires a rate verdict for; the swap rule is not a candidate anchor
ANCHOR_NAMES = ("periodicity", "antiphase", "grav_stab", "gyro_energy")

# swap-rule verdict bands
STANDING, AMBIGUOUS, WALKING = "STANDING", "AMBIGUOUS", "WALKING"


# 0 -> STANDING, 1 -> AMBIGUOUS, >=2 -> WALKING; no fitted parameter anywhere
def swap_verdict(count: int) -> str:
    if count == 0:
        return STANDING
    if count == 1:
        return AMBIGUOUS
    return WALKING


# pearson r for a single pair, through S2's vectorized `_corr`
def _corr1(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2:
        return 0.0
    return float(_corr(a[None, :], b[None, :])[0])


# stride period in samples: the first autocorrelation local max AFTER the acf dips below zero
def stride_period(x: np.ndarray, fs: float,
                  floor: float = ADAPTIVE_PERIODICITY_FLOOR) -> int | None:
    x = x - x.mean()
    n = x.size
    if n < 8 or x.std() == 0:
        return None
    ac = np.correlate(x, x, mode="full")[n - 1:]
    ac = ac / ac[0]
    lo = max(1, int(round(fs / GAIT_BAND_HZ[1])))   # shortest plausible stride period
    hi = min(n - 1, int(round(fs / GAIT_BAND_HZ[0])))  # longest
    if hi <= lo:
        return None
    neg = np.where(ac[lo:hi + 1] < 0)[0]
    if neg.size == 0:
        return None
    start = lo + int(neg[0])
    k = int(np.argmax(ac[start:hi + 1]))
    return start + k if ac[start + k] >= floor else None


# rhythm STRENGTH, amplitude-independent: tallest normalized-autocorrelation peak at a non-zero
def _periodicity(x: np.ndarray, fs: float) -> float:
    x = x - x.mean()
    n = x.size
    if n < 8 or x.std() == 0:
        return 0.0
    ac = np.correlate(x, x, mode="full")[n - 1:]
    ac = ac / ac[0]
    lo = max(1, int(round(fs / GAIT_BAND_HZ[1])))
    hi = min(n - 1, n // 2)  # need >=2 cycles in-window
    if hi <= lo:
        return 0.0
    return float(np.clip(ac[lo:hi + 1].max(), 0.0, 1.0))


# does the full max span GENUINELY alternate?
def _grows_to_walking(d: np.ndarray, l: np.ndarray, r: np.ndarray, a: int, b: int) -> bool:
    seg = d[a:b]
    if seg.size < 8:
        return False
    h = seg.size // 2
    if min(float(np.ptp(seg[:h])), float(np.ptp(seg[h:]))) < GROW_MIN_PTP_DEG:
        return False  # not a both-sides swing
    if -_corr1(l[a:b], r[a:b]) < GROW_MIN_ANTIPHASE:
        return False  # legs not antiphase
    return swap_count(seg, SWAP_DELTA_DEG) >= 2


# stride-sized span for one cell as (a, b, half)
def _adaptive_span(d: np.ndarray, c: int, spec: WindowSpec) -> tuple[int, int, int]:
    base_half = spec.n // 2
    max_half = int(round(MAX_SWAP_WINDOW_S * spec.fs_hz / 2))
    boot = d[max(0, c - max_half):min(d.size, c + max_half)]
    period = stride_period(boot, spec.fs_hz)
    half = base_half if period is None else int(np.clip(period, base_half, max_half))
    return max(0, c - half), min(d.size, c + half), half


# (verdict, span_seconds, swaps, a, b) for one cell- the swap RULE and nothing else
def adaptive_verdict(d: np.ndarray, l: np.ndarray, r: np.ndarray, c: int,
                     spec: WindowSpec) -> tuple[str, float, int, int, int]:
    a, b, half = _adaptive_span(d, c, spec)
    swaps = swap_count(d[a:b], SWAP_DELTA_DEG)
    verdict = swap_verdict(swaps)
    span_s = 2 * half / spec.fs_hz
    if verdict != WALKING:  # grow fallback
        max_half = int(round(MAX_SWAP_WINDOW_S * spec.fs_hz / 2))
        ga, gb = max(0, c - max_half), min(d.size, c + max_half)
        if _grows_to_walking(d, l, r, ga, gb):
            verdict, span_s = WALKING, (gb - ga) / spec.fs_hz
            swaps = swap_count(d[ga:gb], SWAP_DELTA_DEG)
    return verdict, span_s, int(swaps), a, b


# (verdict, span_seconds, periodicity, swaps) for one cell over its stride-sized span
def _adaptive_cell(d: np.ndarray, l: np.ndarray, r: np.ndarray, c: int,
                   spec: WindowSpec) -> tuple[str, float, float, int]:
    verdict, span_s, swaps, a, b = adaptive_verdict(d, l, r, c, spec)
    periodicity = 0.5 * (_periodicity(l[a:b], spec.fs_hz) + _periodicity(r[a:b], spec.fs_hz))
    return verdict, span_s, periodicity, swaps


# every physics anchor + descriptor for one fixed window
def window_anchors(win: pd.DataFrame, fs: float = CANONICAL_HZ,
                   interleg_center: float = 0.0) -> dict[str, float | int | str]:
    l_ang = win[FEATURES[0]].to_numpy(float)   # L_ang_LPF
    r_ang = win[FEATURES[1]].to_numpy(float)   # R_ang_LPF
    l_vel = win[FEATURES[2]].to_numpy(float)   # L_angvel_LPF
    r_vel = win[FEATURES[3]].to_numpy(float)   # R_angvel_LPF
    d = l_ang - r_ang

    swaps = swap_count(d - interleg_center, SWAP_DELTA_DEG)

    # validated descriptors, on raw d
    half = d.size // 2
    ileg_minhalf = (min(float(np.ptp(d[:half])), float(np.ptp(d[half:])))
                    if half >= 1 else 0.0)
    # full-window swing ALONGSIDE the min-half version: `ileg_minhalf` collapses on a start/stop ramp
    ileg_ptp = float(np.ptp(d))
    interleg_offset = float(np.median(d))

    # the four audited anchors; antiphase: legs oppose when walking, NECESSARY not sufficient
    antiphase = -_corr1(l_ang, r_ang)
    # grav_stab: steadiness of tilt; -> 1 standing, -> 0 walking
    grav_stab = 1.0 / (1.0 + 0.5 * (float(l_ang.std()) + float(r_ang.std())))
    periodicity = 0.5 * (_periodicity(l_ang, fs) + _periodicity(r_ang, fs))
    # gyro_energy: TOTAL rotational energy, defined the way that FAILS the audit- its negative control
    gyro_energy = float(np.sum(l_vel ** 2) + np.sum(r_vel ** 2))

    return {
        "periodicity": periodicity,
        "antiphase": antiphase,
        "grav_stab": grav_stab,
        "gyro_energy": gyro_energy,
        "swap_count": int(swaps),
        "swap_verdict": swap_verdict(int(swaps)),
        "ileg_minhalf": ileg_minhalf,
        "ileg_ptp": ileg_ptp,
        "interleg_offset": interleg_offset,
        "interleg_center": float(interleg_center),
    }


# per-window anchors for one trial, on the same window grid S2 features use
def trial_anchors(trial: Trial, spec: WindowSpec | None = None) -> pd.DataFrame:
    spec = spec or WindowSpec()
    frame = trial.frame.reset_index(drop=True)
    if frame.empty:
        return pd.DataFrame()

    # S2's rest zero, not a second opinion about where rest is (see the import note)
    _zeros, center, rest_trusted = rest_reference(frame, spec.fs_hz)

    rows = []
    for seg_id, seg in frame.groupby("segment", sort=True):
        seg = seg.reset_index(drop=True)
        if len(seg) < spec.n:
            continue
        l = seg[FEATURES[0]].to_numpy(float)
        r = seg[FEATURES[1]].to_numpy(float)
        d = (l - r) - center
        times = seg[TIME_COL].to_numpy(float)
        for start in range(0, len(seg) - spec.n + 1, spec.step):
            win = seg.iloc[start:start + spec.n]
            verdict, span_s, per_adaptive, swaps_adaptive = _adaptive_cell(
                d, l, r, start + spec.n // 2, spec)
            rows.append({
                "rev": trial.rev, "trial": trial.trial, "split": trial.split,
                "segment": int(seg_id), "t_start_ms": float(times[start]),
                **window_anchors(win, spec.fs_hz, center),
                "swap_verdict_adaptive": verdict, "swap_window_s": span_s,
                "periodicity_adaptive": per_adaptive,
                "swap_count_adaptive": swaps_adaptive,
                "rest_offset_trusted": rest_trusted,
            })
    return pd.DataFrame(rows)
