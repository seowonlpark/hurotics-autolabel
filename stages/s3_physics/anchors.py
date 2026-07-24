# S3 anchor features: model-free physics view of a window. centerpiece is the SWAP RULE (Section 10,
# zero fitted params): walk = sign-committed crossings of L_ang - R_ang. rate audit drops grid-only anchors.

from __future__ import annotations

import numpy as np
import pandas as pd

from stages.s1_clean.config import CANONICAL_HZ
from stages.s2_ml.features import GAIT_BAND_HZ, WindowSpec, iter_windows
from stages.s2_ml.dataset import FEATURES, TIME_COL, Trial
from stages.s3_physics import discriminators

# sensor noise gate (Section 10): 5x standing noise ~= 0.8 deg, rounded to 1.0. NOT fitted, do not
# tune -- a swap must clear real motion, not jitter.
SWAP_DELTA_DEG = 1.0

# opening-rest window for the per-file interleg zero (Section 10.2). a per-subject DC offset can hold
# L-R above -delta through real gait; subtracting the rest median recenters it (Pareto win, Section 10.4).
REST_ANCHOR_S = 3.0

# stride-adaptive analysis span (Section 10.6). a fixed 2 s window under-calls slow gait; size the swap
# window to ~2 detected strides, capped. standing keeps the base window. walk-recall 0.69->0.86.
MAX_SWAP_WINDOW_S = 6.0
ADAPTIVE_PERIODICITY_FLOOR = 0.35 # autocorr peak height to accept a cell as periodic (else standing)

# span-grow fallback (Section 10.7). when the base verdict is NOT walking, re-count over the full max
# span and accept WALKING only if it GENUINELY alternates: two crossings, both halves past
# GROW_MIN_PTP_DEG, legs anti-correlated past GROW_MIN_ANTIPHASE (blocks weight-shift false WALKs).
GROW_MIN_PTP_DEG = 8.0     # both window-halves must swing this far: a real both-sides stride, not jitter
GROW_MIN_ANTIPHASE = 0.5   # -corr(L,R) over the span: real alternation, not two isolated weight-shifts

# the four anchors PLAN.md S3 requires a rate-invariance verdict for (gait_hz dropped Section 10.8).
# swap rule and Section 10.1 descriptors ride alongside but are not candidate anchors.
ANCHOR_NAMES = ("periodicity", "antiphase", "grav_stab", "gyro_energy")

# swap-rule verdict bands (Section 10): 0 -> STANDING, 1 -> AMBIGUOUS, >=2 -> WALKING
STANDING, AMBIGUOUS, WALKING = "STANDING", "AMBIGUOUS", "WALKING"


# interleg alternations: commits past +delta then -delta, with hysteresis (Section 10). standing 0,
# single weight shift 1, stride >=2.
def swap_count(d: np.ndarray, delta: float = SWAP_DELTA_DEG) -> int:
    commits: list[int] = []
    state = 0 # last committed side: +1, -1, or 0 (uncommitted)
    for x in d:
        if x > delta and state != 1:
            commits.append(1); state = 1
        elif x < -delta and state != -1:
            commits.append(-1); state = -1
    return max(0, len(commits) - 1) # commits alternate by construction => swaps = len-1


# swap-rule discrete verdict (Section 10). '1' is genuinely ambiguous: one leg passing the other
# happens both stepping and weight-shifting.
def swap_verdict(count: int) -> str:
    if count == 0:
        return STANDING
    if count == 1:
        return AMBIGUOUS
    return WALKING


# anchor vocabulary a declarative discriminator may reference: the four audited anchors plus the
# Section 10 / 10.1 descriptors. single-sourced so validate_spec cannot drift from what window_anchors emits.
DISCRIMINATOR_ANCHORS: frozenset[str] = frozenset(ANCHOR_NAMES) | {
    "swap_count", "ileg_minhalf", "interleg_offset", "interleg_center"}

# swap rule registered as the first discriminator (Section 13). a BESPOKE callable, not a threshold:
# window_anchors reads its verdict back through the registry so a new class rides the same seam.
# `origin=code` marks it a trusted built-in. the ONLY import-time anchors<->registry coupling.
SWAP_DISCRIMINATOR = discriminators.register(discriminators.Discriminator(
    name="swap",
    emits=WALKING,
    verdict_of=lambda a: swap_verdict(int(a["swap_count"])),
    classes=(STANDING, AMBIGUOUS, WALKING),
    kind="callable",
    origin="code",
))


# stride period in samples: first autocorrelation local max AFTER the acf dips below zero. skipping
# the lag-0 shoulder is essential -- on slow gait argmax grabs the decay shoulder, not the real peak.
# None when no full cycle resolves in-span or the peak is too weak to be gait (standing).
def stride_period(x: np.ndarray, fs: float, floor: float = ADAPTIVE_PERIODICITY_FLOOR) -> int | None:
    x = x - x.mean()
    n = x.size
    if n < 8 or x.std() == 0:
        return None
    ac = np.correlate(x, x, mode="full")[n - 1:]
    ac = ac / ac[0]
    lo = max(1, int(round(fs / GAIT_BAND_HZ[1]))) # shortest plausible stride period
    hi = min(n - 1, int(round(fs / GAIT_BAND_HZ[0]))) # longest
    if hi <= lo:
        return None
    neg = np.where(ac[lo:hi + 1] < 0)[0] # first descent past zero, past the lag-0 shoulder
    if neg.size == 0:
        return None
    start = lo + int(neg[0])
    k = int(np.argmax(ac[start:hi + 1]))
    return start + k if ac[start + k] >= floor else None


# does the full max span genuinely alternate? (Section 10.7 grow gate): two crossings, both halves
# past GROW_MIN_PTP_DEG, legs anti-correlated past GROW_MIN_ANTIPHASE. l/r are RAW leg angles.
def _grows_to_walking(d: np.ndarray, l: np.ndarray, r: np.ndarray, a: int, b: int) -> bool:
    seg = d[a:b]
    if seg.size < 8:
        return False
    h = seg.size // 2
    if min(float(np.ptp(seg[:h])), float(np.ptp(seg[h:]))) < GROW_MIN_PTP_DEG:
        return False # not a both-sides swing: jitter or one-sided excursion
    if -_corr(l[a:b], r[a:b]) < GROW_MIN_ANTIPHASE:
        return False # legs not antiphase: two weight-shifts, not a stride
    return swap_count(seg, SWAP_DELTA_DEG) >= 2


# stride-adaptive span for a cell (Section 10.6) as (a, b) plus the pre-clip half. sized to ~2 strides
# from stride_period, clipped to [base window, max span]; standing keeps the base window. single-sourced
# so swap count and adaptive periodicity (Section 10.9) read the SAME span.
def _adaptive_span(d: np.ndarray, c: int, spec: WindowSpec) -> tuple[int, int, int]:
    base_half = spec.n // 2
    max_half = int(round(MAX_SWAP_WINDOW_S * spec.fs_hz / 2))
    boot = d[max(0, c - max_half):min(d.size, c + max_half)] # bootstrap span for period detection
    period = stride_period(boot, spec.fs_hz)
    half = base_half if period is None else int(np.clip(period, base_half, max_half))
    return max(0, c - half), min(d.size, c + half), half


# per-cell adaptive computation over ONE span (Section 10.6): swap verdict, span seconds, adaptive
# periodicity (Section 10.9). when the base call is NOT walking but the max span alternates, grow to it
# (Section 10.7). returns (verdict, span_seconds, periodicity).
def _adaptive_cell(d: np.ndarray, l: np.ndarray, r: np.ndarray, c: int,
                   spec: WindowSpec) -> tuple[str, float, float]:
    a, b, half = _adaptive_span(d, c, spec)
    verdict = swap_verdict(swap_count(d[a:b], SWAP_DELTA_DEG))
    span_s = 2 * half / spec.fs_hz
    if verdict != WALKING: # Section 10.7 grow-fallback
        max_half = int(round(MAX_SWAP_WINDOW_S * spec.fs_hz / 2))
        ga, gb = max(0, c - max_half), min(d.size, c + max_half)
        if _grows_to_walking(d, l, r, ga, gb):
            verdict, span_s = WALKING, (gb - ga) / spec.fs_hz # report grown span
    periodicity = 0.5 * (_periodicity(l[a:b], spec.fs_hz) + _periodicity(r[a:b], spec.fs_hz))
    return verdict, span_s, periodicity


# back-compat wrapper: swap verdict + span only (Section 10.6). with l/r it applies the Section 10.7 grow
# gate; without them the grow gate is skipped. returns (verdict, span_seconds).
def adaptive_swap_at(d: np.ndarray, c: int, spec: WindowSpec,
                     l: np.ndarray | None = None, r: np.ndarray | None = None) -> tuple[str, float]:
    if l is not None and r is not None:
        verdict, span_s, _ = _adaptive_cell(d, l, r, c, spec)
        return verdict, span_s
    a, b, half = _adaptive_span(d, c, spec)
    return swap_verdict(swap_count(d[a:b], SWAP_DELTA_DEG)), 2 * half / spec.fs_hz


# pearson r; 0 when either series is constant or too short (matches S2 _corr)
def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


# rhythm STRENGTH, amplitude-independent: tallest normalized-autocorrelation peak at a non-zero lag
# whose period is a plausible stride (>=2 cycles in-window). 1.0 = perfectly periodic, ~0 = none.
# within-window only: read as "steady rhythm here", never "the subject has a rhythm" (Section 11.2).
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


# every physics anchor + descriptor for one window. d = L_ang - R_ang is the interleg signal the swap
# rule reads; interleg_center (Section 10.4) is subtracted before the swap count only, since the offset
# IS the posture for the descriptors. the four ANCHOR_NAMES keys are what the rate audit re-checks.
def window_anchors(win: pd.DataFrame, fs: float = CANONICAL_HZ,
                   interleg_center: float = 0.0) -> dict[str, float | int | str]:
    l_ang = win[FEATURES[0]].to_numpy(float) # L_ang_LPF
    r_ang = win[FEATURES[1]].to_numpy(float) # R_ang_LPF
    l_vel = win[FEATURES[2]].to_numpy(float) # L_angvel_LPF
    r_vel = win[FEATURES[3]].to_numpy(float) # R_angvel_LPF
    d = l_ang - r_ang

    # swap rule (Section 10): zero-parameter walk/stand call, on the interleg RECENTERED on the
    # per-file rest zero (Section 10.4) so a DC offset can't hold it above -delta through real gait
    swaps = swap_count(d - interleg_center, SWAP_DELTA_DEG)

    # validated descriptors (Section 10.1), on raw d
    half = d.size // 2
    ileg_minhalf = (min(float(np.ptp(d[:half])), float(np.ptp(d[half:])))
                    if half >= 1 else 0.0) # both halves must move, not one big excursion (offset-free)
    interleg_offset = float(np.median(d)) # posture: which leg leads, feet-together vs split (raw)

    # the four PLAN anchors, each under rate-invariance audit (gait_hz dropped Section 10.8: degenerate
    # 0.5 Hz FFT floor at the 2 s window).
    # antiphase: legs swing in opposition (r<0) when walking. NECESSARY but not sufficient (Section 6.2).
    antiphase = -_corr(l_ang, r_ang)
    # grav_stab: steadiness of gravity-referenced tilt. ->1 standing, ->0 walking.
    grav_stab = 1.0 / (1.0 + 0.5 * (float(l_ang.std()) + float(r_ang.std())))
    # periodicity: rhythm strength (autocorrelation), averaged over both legs
    periodicity = 0.5 * (_periodicity(l_ang, fs) + _periodicity(r_ang, fs))
    # gyro_energy: TOTAL rotational energy, summed. scales with sample count, so it's a grid claim, not
    # a body claim -- defined the way that fails the rate audit, on purpose (id=69, Section 2.3).
    gyro_energy = float(np.sum(l_vel ** 2) + np.sum(r_vel ** 2))

    return {
        "periodicity": periodicity,
        "antiphase": antiphase,
        "grav_stab": grav_stab,
        "gyro_energy": gyro_energy,
        # descriptors reported alongside (not audited as anchors)
        "swap_count": int(swaps),
        # verdict read back through the registry, not swap_verdict() directly, so the production column
        # flows through the same seam a new class would (verdict_of delegates to swap_verdict)
        "swap_verdict": SWAP_DISCRIMINATOR.verdict_of({"swap_count": int(swaps)}),
        "ileg_minhalf": ileg_minhalf,
        "interleg_offset": interleg_offset,
        "interleg_center": float(interleg_center), # per-file rest zero subtracted before the swap count
    }


# interleg signal L_ang - R_ang for a frame (or slice), as an array
def _interleg(frame: pd.DataFrame) -> np.ndarray:
    return frame[FEATURES[0]].to_numpy(float) - frame[FEATURES[1]].to_numpy(float)


# is a candidate rest span ACTUALLY at rest? swap rule's STANDING verdict on the locally centered span
# (0 alternations, Section 10). reusing the validated rule keeps this parameter-free; a span it would call
# WALKING is early gait masquerading as rest, its median a poisoned offset.
def _is_rest(d: np.ndarray) -> bool:
    return d.size > 0 and swap_count(d - float(np.median(d)), SWAP_DELTA_DEG) == 0


# stillest true-rest span anywhere in the recording: scan every segment in REST_ANCHOR_S hops, keep the
# median of the 0-swap span with the smallest interleg range. within a segment only (never straddle a
# gap). None when it never rests. fallback zero for files that do NOT begin at rest (Section 10.2).
def _stillest_rest_span(frame: pd.DataFrame, span: int) -> float | None:
    best_ptp, best_med = np.inf, None
    for _seg_id, seg in frame.groupby("segment", sort=True):
        d = _interleg(seg)
        for a in range(0, d.size - span + 1, span):
            s = d[a:a + span]
            if _is_rest(s) and (p := float(np.ptp(s))) < best_ptp:
                best_ptp, best_med = p, float(np.median(s))
    return best_med


# per-file interleg zero + whether it was measured on a span actually at rest. median(L-R) over the
# opening rest (Section 10.2) is the per-subject mounting bias; TRUST it only if the swap rule calls it
# STANDING, else the stillest rest span, else the whole-file median returned UNTRUSTED. lockbox-safe.
def rest_anchor(trial: Trial, fs: float = CANONICAL_HZ) -> tuple[float, bool]:
    frame = trial.frame
    if frame.empty:
        return 0.0, False
    span = int(round(REST_ANCHOR_S * fs))
    seg0 = frame[frame["segment"] == frame["segment"].min()] # first segment
    d_open = _interleg(seg0.iloc[:span])
    if _is_rest(d_open): # recordings begin at rest (Section 10.2): common case
        return float(np.median(d_open)), True
    fallback = _stillest_rest_span(frame, span) # opening isn't rest: find rest elsewhere
    if fallback is not None:
        return fallback, True
    return float(np.median(_interleg(frame))), False # never rests: robust but untrusted


# backward-compatible scalar zero (rate audit needs only the number, Section 10.4)
def rest_offset(trial: Trial, fs: float = CANONICAL_HZ) -> float:
    return rest_anchor(trial, fs)[0]


# stride-adaptive verdict + span + periodicity for every cell, keyed by (segment, t_start_ms). a keyed
# join, not a positional loop, so it can't drift from iter_windows -- it needs the whole segment (2
# strides of context), which is why it lives here not in window_anchors.
def _adaptive_swaps(trial: Trial, spec: WindowSpec, center: float) -> dict:
    out: dict[tuple[int, float], tuple[str, float, float]] = {}
    for seg_id, seg in trial.frame.groupby("segment", sort=True):
        seg = seg.reset_index(drop=True)
        l = seg[FEATURES[0]].to_numpy(float)
        r = seg[FEATURES[1]].to_numpy(float)
        d = (l - r) - center
        times = seg[TIME_COL].to_numpy(float)
        for start in range(0, len(seg) - spec.n + 1, spec.step):
            out[(int(seg_id), float(times[start]))] = _adaptive_cell(
                d, l, r, start + spec.n // 2, spec) # l/r enable grow gate + periodicity
    return out


# per-window anchors for one trial as a table: window metadata joined to every anchor + descriptor,
# reusing the S2 window iterator. the rest zero is subtracted from every swap count (Section 10.4); the
# adaptive verdict/periodicity join alongside the fixed ones, and rest_offset_trusted rides along per row.
def trial_anchors(trial: Trial, spec: WindowSpec | None = None) -> pd.DataFrame:
    spec = spec or WindowSpec()
    center, rest_trusted = rest_anchor(trial, spec.fs_hz)
    adaptive = _adaptive_swaps(trial, spec, center)
    rows = []
    for meta, win in iter_windows(trial, spec):
        verdict, span_s, per_adaptive = adaptive[(meta["segment"], meta["t_start_ms"])]
        rows.append({**meta, **window_anchors(win, spec.fs_hz, center),
                     "swap_verdict_adaptive": verdict, "swap_window_s": span_s,
                     "periodicity_adaptive": per_adaptive,
                     "rest_offset_trusted": rest_trusted})
    return pd.DataFrame(rows)
