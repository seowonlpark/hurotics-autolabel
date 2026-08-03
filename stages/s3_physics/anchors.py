"""S3 anchor features: a model-free physics view of one window.

The centrepiece is the SWAP RULE (§10), and its value here is precisely that it is
**independent of S2**. It reads a different quantity (interleg angle, not 23 windowed
statistics), it has zero fitted parameters, and it never sees a label. When it agrees
with the classifier the two are corroborating rather than echoing; when it disagrees,
that disagreement is evidence, not noise. S4 spends exactly that.

> Walking is the legs alternating. Not how far they swing — *whether they swap*.
> Count how many times `L_ang - R_ang` commits past `+1°` and then past `-1°`.
> 0 swaps -> STANDING. >=2 -> WALKING. Exactly 1 -> AMBIGUOUS.

`1` is genuinely ambiguous, not a fudge: one leg passing the other happens both when you
take a step and when you shift your weight. The rule ABSTAINS there rather than guessing,
which is the behaviour this repo wants (§7, "abstain rather than force").

Two corrections to the naive rule, both measured, both carried from prior work:

  - **Recentre on the per-file rest zero** (§10.4). A per-subject mounting offset can
    hold `L-R` above `-delta` through real gait and silently suppress every swap.
  - **Size the span to the stride** (§10.6/10.7). A fixed 2 s window under-calls slow
    gait — the rule's known weakness, a 0.22 Hz stride yields ~0.9 swaps per 2 s and
    abstains. Sizing the span to ~2 detected strides moved walk-recall 0.69 -> 0.86.

What is deliberately NOT here (this repo labels stand/walk and nothing else): no
discriminator registry, no stairs rule, no new-class seam. Those live in the sibling
experimental repo. See `caveats.md`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stages.s1_clean.config import CANONICAL_HZ
from stages.s2_ml.dataset import FEATURES, TIME_COL, Trial
# `_corr` and `rest_reference` are shared with S2 deliberately.
#
# `_corr`: a second copy drifted from it once already in a sibling repo, and that repo's
# grow gate calls `_corr` without importing it at all — a latent NameError on the slow
# gait path. It is vectorized over rows now, so scalar callers go through `_corr1`.
#
# `rest_reference`: S2's interleg features are centred on it, and if S3 measured its own
# rest zero the two stages would disagree about where "rest" is on the same recording —
# the swap count would then be taken about a different origin than the `ileg_*` features
# the model reads. One definition, both consumers.
from stages.s2_ml.features import GAIT_BAND_HZ, WindowSpec, _corr, rest_reference
from stages.s2_ml.rest import SWAP_DELTA_DEG, swap_count

# Stride-adaptive analysis span (§10.6). Standing keeps the base window; only motion
# grows one. Capped so a span can never swallow a whole bout and average two states.
MAX_SWAP_WINDOW_S = 6.0
ADAPTIVE_PERIODICITY_FLOOR = 0.35  # autocorr peak height to accept a cell as periodic

# Span-grow fallback (§10.7). When the base verdict is NOT walking, re-count over the
# full max span and accept WALKING only if the span GENUINELY alternates. All three
# gates must pass, because a slow walker and a person shifting their weight twice look
# identical to a bare swap count over a long span.
GROW_MIN_PTP_DEG = 8.0     # both halves must swing this far: a real stride, not jitter
GROW_MIN_ANTIPHASE = 0.5   # -corr(L,R): alternation, not two isolated weight shifts

# The four anchors PLAN.md requires a rate-invariance verdict for (§10.8). The swap rule
# and the §10.1 descriptors ride alongside but are not candidate anchors.
ANCHOR_NAMES = ("periodicity", "antiphase", "grav_stab", "gyro_energy")

# Swap-rule verdict bands (§10).
STANDING, AMBIGUOUS, WALKING = "STANDING", "AMBIGUOUS", "WALKING"


def swap_verdict(count: int) -> str:
    """0 -> STANDING, 1 -> AMBIGUOUS, >=2 -> WALKING. No fitted parameter anywhere."""
    if count == 0:
        return STANDING
    if count == 1:
        return AMBIGUOUS
    return WALKING


def _corr1(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson r for a single pair, through S2's vectorized `_corr`.

    A reshape rather than a reimplementation: the point is that S2 and S3 cannot compute
    correlation differently, including the shared convention that a constant series
    yields 0.0 instead of NaN.
    """
    if a.size < 2:
        return 0.0
    return float(_corr(a[None, :], b[None, :])[0])


def stride_period(x: np.ndarray, fs: float,
                  floor: float = ADAPTIVE_PERIODICITY_FLOOR) -> int | None:
    """Stride period in samples: the first autocorrelation local max AFTER the acf dips
    below zero.

    Skipping the lag-0 shoulder is essential rather than tidy — on slow gait a plain
    argmax grabs the decay shoulder and returns a period of a few samples, which then
    sizes the adaptive span far too short and re-creates the problem it exists to fix.

    None when no full cycle resolves in-span, or the peak is too weak to be gait.
    """
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


def _periodicity(x: np.ndarray, fs: float) -> float:
    """Rhythm STRENGTH, amplitude-independent: tallest normalized-autocorrelation peak at
    a non-zero lag whose period is a plausible stride (>=2 cycles in-window).

    Within-window only. Read it as "steady rhythm here", never "this subject has a
    rhythm" — §11.2 records a whole invented category ("walking with no rhythm") that
    came from reading a per-window periodicity dip as a property of the walker. The dip
    was a cadence CHANGE, in both trials, at the same protocol event.
    """
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


def _grows_to_walking(d: np.ndarray, l: np.ndarray, r: np.ndarray, a: int, b: int) -> bool:
    """Does the full max span GENUINELY alternate? (§10.7 grow gate.)

    Three independent gates, because the failure this guards against is specific: over a
    6 s span, a person who shifts their weight twice produces two crossings exactly like
    a slow walker. Amplitude (both halves swing) and antiphase (the legs oppose) are what
    separate them; the swap count alone cannot.
    """
    seg = d[a:b]
    if seg.size < 8:
        return False
    h = seg.size // 2
    if min(float(np.ptp(seg[:h])), float(np.ptp(seg[h:]))) < GROW_MIN_PTP_DEG:
        return False  # not a both-sides swing
    if -_corr1(l[a:b], r[a:b]) < GROW_MIN_ANTIPHASE:
        return False  # legs not antiphase
    return swap_count(seg, SWAP_DELTA_DEG) >= 2


def _adaptive_span(d: np.ndarray, c: int, spec: WindowSpec) -> tuple[int, int, int]:
    """Stride-sized span for one cell as (a, b, half). Clipped to [base window, max span].

    Single-sourced so the swap count and the adaptive periodicity read the SAME span —
    two spans would make the verdict and its own supporting evidence describe different
    stretches of time.
    """
    base_half = spec.n // 2
    max_half = int(round(MAX_SWAP_WINDOW_S * spec.fs_hz / 2))
    boot = d[max(0, c - max_half):min(d.size, c + max_half)]
    period = stride_period(boot, spec.fs_hz)
    half = base_half if period is None else int(np.clip(period, base_half, max_half))
    return max(0, c - half), min(d.size, c + half), half


def _adaptive_cell(d: np.ndarray, l: np.ndarray, r: np.ndarray, c: int,
                   spec: WindowSpec) -> tuple[str, float, float]:
    """(verdict, span_seconds, periodicity) for one cell over its stride-sized span."""
    a, b, half = _adaptive_span(d, c, spec)
    verdict = swap_verdict(swap_count(d[a:b], SWAP_DELTA_DEG))
    span_s = 2 * half / spec.fs_hz
    if verdict != WALKING:  # §10.7 grow fallback
        max_half = int(round(MAX_SWAP_WINDOW_S * spec.fs_hz / 2))
        ga, gb = max(0, c - max_half), min(d.size, c + max_half)
        if _grows_to_walking(d, l, r, ga, gb):
            verdict, span_s = WALKING, (gb - ga) / spec.fs_hz
    periodicity = 0.5 * (_periodicity(l[a:b], spec.fs_hz) + _periodicity(r[a:b], spec.fs_hz))
    return verdict, span_s, periodicity


def window_anchors(win: pd.DataFrame, fs: float = CANONICAL_HZ,
                   interleg_center: float = 0.0) -> dict[str, float | int | str]:
    """Every physics anchor + descriptor for one fixed window.

    `interleg_center` is subtracted before the swap count ONLY. For the descriptors the
    offset *is* the posture (§10.1: `interleg_offset` separates feet-together from split
    stance at +17°/-22°), so removing it there would delete the signal.
    """
    l_ang = win[FEATURES[0]].to_numpy(float)   # L_ang_LPF
    r_ang = win[FEATURES[1]].to_numpy(float)   # R_ang_LPF
    l_vel = win[FEATURES[2]].to_numpy(float)   # L_angvel_LPF
    r_vel = win[FEATURES[3]].to_numpy(float)   # R_angvel_LPF
    d = l_ang - r_ang

    swaps = swap_count(d - interleg_center, SWAP_DELTA_DEG)

    # Validated descriptors (§10.1), on raw d.
    half = d.size // 2
    ileg_minhalf = (min(float(np.ptp(d[:half])), float(np.ptp(d[half:])))
                    if half >= 1 else 0.0)
    interleg_offset = float(np.median(d))

    # The four audited anchors (§10.8).
    # antiphase: legs oppose when walking. NECESSARY, not sufficient (§6.2).
    antiphase = -_corr1(l_ang, r_ang)
    # grav_stab: steadiness of tilt. -> 1 standing, -> 0 walking.
    grav_stab = 1.0 / (1.0 + 0.5 * (float(l_ang.std()) + float(r_ang.std())))
    periodicity = 0.5 * (_periodicity(l_ang, fs) + _periodicity(r_ang, fs))
    # gyro_energy: TOTAL rotational energy. Scales with sample count, so it is a claim
    # about the sampling grid, not about the body — defined the way that FAILS the rate
    # audit, deliberately. It is the audit's negative control: an audit that has never
    # rejected anything is not evidence that the others passed (§11.1).
    gyro_energy = float(np.sum(l_vel ** 2) + np.sum(r_vel ** 2))

    return {
        "periodicity": periodicity,
        "antiphase": antiphase,
        "grav_stab": grav_stab,
        "gyro_energy": gyro_energy,
        "swap_count": int(swaps),
        "swap_verdict": swap_verdict(int(swaps)),
        "ileg_minhalf": ileg_minhalf,
        "interleg_offset": interleg_offset,
        "interleg_center": float(interleg_center),
    }


def trial_anchors(trial: Trial, spec: WindowSpec | None = None) -> pd.DataFrame:
    """Per-window anchors for one trial, on the same window grid S2 features use.

    The grid is `range(0, len(seg) - spec.n + 1, spec.step)` per segment — identical to
    the `sliding_window_view(...)[::step]` walk in `features.windows_of_trial`, and
    segments shorter than one window are skipped there too. So
    `(rev, trial, segment, t_start_ms)` joins the two stages exactly, and S4 asserts it
    with `validate="one_to_one"` rather than trusting this comment.

    The adaptive verdict is computed per segment (it needs ~2 strides of context either
    side, which a single window does not contain) and read back positionally within the
    same walk, so the two cannot fall out of step.
    """
    spec = spec or WindowSpec()
    frame = trial.frame.reset_index(drop=True)
    if frame.empty:
        return pd.DataFrame()

    # S2's rest zero, not a second opinion about where rest is (see the import note).
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
            verdict, span_s, per_adaptive = _adaptive_cell(
                d, l, r, start + spec.n // 2, spec)
            rows.append({
                "rev": trial.rev, "trial": trial.trial, "split": trial.split,
                "segment": int(seg_id), "t_start_ms": float(times[start]),
                **window_anchors(win, spec.fs_hz, center),
                "swap_verdict_adaptive": verdict, "swap_window_s": span_s,
                "periodicity_adaptive": per_adaptive,
                "rest_offset_trusted": rest_trusted,
            })
    return pd.DataFrame(rows)
