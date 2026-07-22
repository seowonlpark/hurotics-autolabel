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
from stages.s2_ml.dataset import FEATURES, TIME_COL, Trial

# the sensor noise floor (§10): 5x the measured 0.76-0.88 deg standing noise. NOT fitted --
# a swap must clear real motion, not jitter. do not tune this; the rule's whole claim is
# that nothing here is tuned.
SWAP_DELTA_DEG = 1.0

# the opening-rest window used for the per-file interleg zero (§10.2 "recordings begin at
# rest"). the swap rule counts crossings of L-R past ±delta, so a file whose interleg sits at
# a per-subject DC offset (the §4.6 zeroing bias the S2 champion also drops) never commits
# past -delta and reads STANDING through real gait. subtracting the rest-anchor median
# recenters it -- measured a corpus Pareto win (walk 0.63→0.69, stand 0.90→0.93, §10.4).
REST_ANCHOR_S = 3.0

# stride-adaptive analysis span (§10.6). a fixed 2 s window holds <2 strides once the stride
# period nears 2 s, so swap_count reads 0-1 and the rule under-calls slow gait -- the regime
# the deployment population (assistive / rehab) actually lives in. the fix: size the swap
# window to ~2 detected strides, capped; standing has no detectable period and keeps the base
# window (so it is not lengthened into its neighbours). measured to recover walk-recall
# 0.69→0.86 for stand-recall 0.93→0.92 (§10.6) -- most of a long window's gain, little of its cost.
MAX_SWAP_WINDOW_S = 6.0
ADAPTIVE_PERIODICITY_FLOOR = 0.35 # autocorr peak height to accept a cell as periodic (else standing)

# span-grow fallback (§10.7). the §10.6 detector under-fires: on slow/large strides stride_period
# returns None, the window stays at 2 s, holds one hump, and swap_count reads 1 -- 94% of the
# missed WALK windows are stuck at the base span, swinging a median 40° they never get to alternate
# twice inside. the fix: when the base verdict is NOT walking, re-count over the full max span and
# accept WALKING only if that span GENUINELY alternates -- two committed crossings, both halves
# swinging past GROW_MIN_PTP_DEG, and the legs anti-correlated past GROW_MIN_ANTIPHASE. the last
# gate is the §10.2/§11 discriminator: two stray standing weight-shifts also give swap_count 2 over
# 6 s, but their legs are not antiphase, so requiring leg anti-correlation keeps this from merging
# them into a false WALK. NOT a period detector -- it sidesteps the thing that failed. measured to
# recover walk-recall 0.86→0.92 (0.69→0.88 on files that do not begin at rest) for +27 STAND windows
# corpus-wide (~10:1), all routed through the disagreement audit (§10.7).
GROW_MIN_PTP_DEG = 8.0     # both window-halves must swing this far: a real both-sides stride, not jitter
GROW_MIN_ANTIPHASE = 0.5   # -corr(L,R) over the span: real alternation, not two isolated weight-shifts

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


# stride period in samples: the first autocorrelation local max AFTER the acf first dips below
# zero. skipping the lag-0 shoulder is essential -- on slow gait a plain argmax grabs the
# monotonic-decay shoulder at the shortest lag, not the real periodic peak (measured: it
# returned 0.33 s, the floor, on 2.9 s strides). None when no full cycle is resolved in the
# span (never dips) or the peak is too weak to be gait (standing).
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


# does the full max span genuinely alternate? (§10.7 grow-fallback gate). the rescue for a
# slow/large stride the period detector could not extend for: two committed crossings over the
# max span, both halves swinging past GROW_MIN_PTP_DEG, and the legs anti-correlated past
# GROW_MIN_ANTIPHASE. l/r are the RAW leg angles over the same span (antiphase is offset-free).
def _grows_to_walking(d: np.ndarray, l: np.ndarray, r: np.ndarray, a: int, b: int) -> bool:
    seg = d[a:b]
    if seg.size < 8:
        return False
    h = seg.size // 2
    if min(float(np.ptp(seg[:h])), float(np.ptp(seg[h:]))) < GROW_MIN_PTP_DEG:
        return False # not a real both-sides swing -- jitter or a one-sided excursion
    if -_corr(l[a:b], r[a:b]) < GROW_MIN_ANTIPHASE:
        return False # legs not antiphase -- two isolated weight-shifts, not a stride
    return swap_count(seg, SWAP_DELTA_DEG) >= 2


# swap verdict at one cell over a span sized to ~2 detected strides (§10.6). d is the WHOLE
# segment's centered interleg signal, c the cell centre, l/r the RAW leg angles for the §10.7
# grow gate. periodic -> 2×stride span (capped at MAX_SWAP_WINDOW_S, floored at the base window);
# non-periodic (standing) -> base window. when that call is NOT walking but the full max span
# genuinely alternates, grow to it (§10.7) -- this is what recovers the slow/large gait the
# period detector missed. l/r default None (the grow gate is skipped) so any caller passing only
# d keeps the old §10.6 behaviour. returns (verdict, span_seconds).
def adaptive_swap_at(d: np.ndarray, c: int, spec: WindowSpec,
                     l: np.ndarray | None = None, r: np.ndarray | None = None) -> tuple[str, float]:
    base_half = spec.n // 2
    max_half = int(round(MAX_SWAP_WINDOW_S * spec.fs_hz / 2))
    boot = d[max(0, c - max_half):min(d.size, c + max_half)] # bootstrap span for period detection
    period = stride_period(boot, spec.fs_hz)
    half = base_half if period is None else int(np.clip(period, base_half, max_half))
    a, b = max(0, c - half), min(d.size, c + half)
    verdict = swap_verdict(swap_count(d[a:b], SWAP_DELTA_DEG))

    if verdict != WALKING and l is not None and r is not None: # §10.7 grow-fallback
        ga, gb = max(0, c - max_half), min(d.size, c + max_half)
        if _grows_to_walking(d, l, r, ga, gb):
            return WALKING, (gb - ga) / spec.fs_hz # report the grown span
    return verdict, 2 * half / spec.fs_hz


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
# the swap rule reads. interleg_center is the per-file rest zero (§10.4) subtracted before the
# swap count only -- the posture descriptors below stay on raw d, because the offset IS the
# posture. returns a flat dict; the five ANCHOR_NAMES keys are what the rate audit re-checks,
# the rest are the validated §10 / §10.1 descriptors reported alongside.
def window_anchors(win: pd.DataFrame, fs: float = CANONICAL_HZ,
                   interleg_center: float = 0.0) -> dict[str, float | int | str]:
    l_ang = win[FEATURES[0]].to_numpy(float) # L_ang_LPF
    r_ang = win[FEATURES[1]].to_numpy(float) # R_ang_LPF
    l_vel = win[FEATURES[2]].to_numpy(float) # L_angvel_LPF
    r_vel = win[FEATURES[3]].to_numpy(float) # R_angvel_LPF
    d = l_ang - r_ang

    # --- the swap rule (§10): the validated, zero-parameter walk/stand call ---
    # counted on the interleg RECENTERED on the per-file rest zero (§10.4): raw d carries a
    # per-subject DC offset that can hold it above -delta through real gait
    swaps = swap_count(d - interleg_center, SWAP_DELTA_DEG)

    # --- validated descriptors (§10.1), on raw d ---
    half = d.size // 2
    ileg_minhalf = (min(float(np.ptp(d[:half])), float(np.ptp(d[half:])))
                    if half >= 1 else 0.0) # both halves must move, not one big excursion (offset-free)
    interleg_offset = float(np.median(d)) # posture: which leg leads, feet-together vs split (raw)

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
        "interleg_center": float(interleg_center), # per-file rest zero subtracted before the swap count
    }


# the interleg signal L_ang - R_ang for a frame (or slice), as an array
def _interleg(frame: pd.DataFrame) -> np.ndarray:
    return frame[FEATURES[0]].to_numpy(float) - frame[FEATURES[1]].to_numpy(float)


# is a candidate rest span ACTUALLY at rest? the swap rule's own STANDING verdict, read on the
# locally centered span: 0 leg alternations (§10). reusing the validated rule instead of a new
# stillness threshold keeps this parameter-free and consistent with how "not walking" is defined
# everywhere else -- a span the rule would call STANDING is a clean zero; one it would call
# WALKING is early gait masquerading as the opening rest, and its median is a poisoned offset.
def _is_rest(d: np.ndarray) -> bool:
    return d.size > 0 and swap_count(d - float(np.median(d)), SWAP_DELTA_DEG) == 0


# the stillest true-rest span anywhere in the recording: scan every segment in REST_ANCHOR_S
# hops, keep the median of the 0-swap span with the smallest interleg range (ptp is offset-free,
# so it ranks stillness, not posture). within a segment only, so a span never straddles a gap
# (the iter_windows rule). None when the recording never rests. this is the fallback zero for the
# ~2/28 files that do NOT begin at rest (§10.2): the offset we want is the per-subject mounting/
# zeroing bias (§4.6), constant across the file, so any genuinely still span estimates it -- we
# just need one the swap rule agrees is still, not the opening one taken on faith.
def _stillest_rest_span(frame: pd.DataFrame, span: int) -> float | None:
    best_ptp, best_med = np.inf, None
    for _seg_id, seg in frame.groupby("segment", sort=True):
        d = _interleg(seg)
        for a in range(0, d.size - span + 1, span):
            s = d[a:a + span]
            if _is_rest(s) and (p := float(np.ptp(s))) < best_ptp:
                best_ptp, best_med = p, float(np.median(s))
    return best_med


# the per-file interleg zero + whether it was measured on a span that is actually at rest.
# median(L_ang - R_ang) over the recording's opening rest (REST_ANCHOR_S, §10.2) is the
# per-subject mounting/zeroing offset (§4.6) -- the same static bias the S2 champion drops as
# non-gait. but the "begins at rest" assumption fails on ~2/28 files, and there the opening
# median is taken over early gait, poisoning the zero SILENTLY. so verify it: TRUST the opening
# span only if the swap rule calls it STANDING; else fall back to the stillest true-rest span
# anywhere in the recording; else (the recording never rests) the whole-file median, returned
# UNTRUSTED so a downstream reader knows this zero is the graceful-degradation path, not a
# measured rest. label-free throughout -- reads only the signal, so it is lockbox-safe (§10.4).
def rest_anchor(trial: Trial, fs: float = CANONICAL_HZ) -> tuple[float, bool]:
    frame = trial.frame
    if frame.empty:
        return 0.0, False
    span = int(round(REST_ANCHOR_S * fs))
    seg0 = frame[frame["segment"] == frame["segment"].min()] # the recording's first segment
    d_open = _interleg(seg0.iloc[:span])
    if _is_rest(d_open): # recordings begin at rest (§10.2) -- the common case
        return float(np.median(d_open)), True
    fallback = _stillest_rest_span(frame, span) # opening isn't rest: find rest elsewhere
    if fallback is not None:
        return fallback, True
    return float(np.median(_interleg(frame))), False # never rests: robust but untrusted


# backward-compatible scalar zero (the rate audit needs only the number, §10.4)
def rest_offset(trial: Trial, fs: float = CANONICAL_HZ) -> float:
    return rest_anchor(trial, fs)[0]


# stride-adaptive swap verdict + span for every candidate cell, keyed by the natural
# (segment, t_start_ms) identifiers the per-window rows already carry. a keyed join, not a
# parallel positional loop, so it cannot drift out of step with iter_windows -- it needs the
# whole segment (the isolated-window path can't see 2 strides of context), which is why it
# lives here rather than in window_anchors.
def _adaptive_swaps(trial: Trial, spec: WindowSpec, center: float) -> dict:
    out: dict[tuple[int, float], tuple[str, float]] = {}
    for seg_id, seg in trial.frame.groupby("segment", sort=True):
        seg = seg.reset_index(drop=True)
        l = seg[FEATURES[0]].to_numpy(float)
        r = seg[FEATURES[1]].to_numpy(float)
        d = (l - r) - center
        times = seg[TIME_COL].to_numpy(float)
        for start in range(0, len(seg) - spec.n + 1, spec.step):
            out[(int(seg_id), float(times[start]))] = adaptive_swap_at(
                d, start + spec.n // 2, spec, l, r) # l/r enable the §10.7 grow gate
    return out


# per-window anchors for one trial as a table: window metadata (rev/trial/segment/t_start/
# human label) joined to every anchor + descriptor. reuses the S2 window iterator so the
# "never window across a gap" rule stays single-sourced. the per-file rest zero is measured
# once and subtracted from every window's swap count (§10.4); the stride-adaptive verdict
# (§10.6) is joined in alongside the fixed-window one, never replacing it. rest_offset_trusted
# rides along on every row (constant per trial): False means the zero fell back off the opening
# rest (§10.2), so a reader can discount that file's swap calls rather than trust them blind.
def trial_anchors(trial: Trial, spec: WindowSpec | None = None) -> pd.DataFrame:
    spec = spec or WindowSpec()
    center, rest_trusted = rest_anchor(trial, spec.fs_hz)
    adaptive = _adaptive_swaps(trial, spec, center)
    rows = []
    for meta, win in iter_windows(trial, spec):
        verdict, span_s = adaptive[(meta["segment"], meta["t_start_ms"])]
        rows.append({**meta, **window_anchors(win, spec.fs_hz, center),
                     "swap_verdict_adaptive": verdict, "swap_window_s": span_s,
                     "rest_offset_trusted": rest_trusted})
    return pd.DataFrame(rows)
