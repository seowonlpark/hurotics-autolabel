"""S3 rate-invariance audit: does an anchor describe the body, or the sampling grid?

PLAN's S3 gate is *no anchor feature without a rate-invariance verdict*, and §2.3 is why:
experiment id=69 read clustering that partitioned by "acquisition rate" across
99.4 / 99.7 / 100.0 / 500.0 Hz — which §2.2 later showed was **timestamp quantization**,
the same devices counting in binary sub-ms ticks. The geometry was measuring the clock.

So the test is: decimate every window to half rate through S1's own anti-aliased filter,
recompute each anchor, and ask whether the number moved. A body-defined anchor must not.
One that is really a claim about the sample count must.

The audit is gated to WALKING windows via the swap rule, which keeps it **label-free** —
standing has no cadence to compare, and using ground truth to select the audit set would
make a physics check depend on the annotations it is supposed to be independent of.

**The window is decimated WITH CONTEXT, and that is not an optimization [measured,
2026-08-03].** Decimating a bare 2 s window (200 samples) applies a 41-tap FIR through
`filtfilt`, whose edge transient runs ~123 samples — over half the window is the
filter's boundary behaviour rather than the signal. Measured on 2,966 walking windows,
that artefact alone moved `antiphase` by **0.1734** and condemned it `rate_dependent`.
Pulling the same window with 1.28 s of padding either side, decimating, then trimming
back drops the move to **0.0005**: `antiphase` is invariant, and the earlier verdict was
the audit measuring its own filter. Two sibling repos record the uncorrected number.

The control that says the fix did not simply blunt the test: `gyro_energy` is
**unchanged** at 0.4994 and still fails. It sums over samples, so halving the sample
count halves it — a claim about the grid, exactly as §2.3 predicts. A test that stopped
rejecting anything would be the §11.1 failure one level down.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import decimate

from stages.s1_clean.config import CANONICAL_HZ, DECIMATE_FILTER
from stages.s2_ml.dataset import FEATURES, Trial
from stages.s2_ml.features import WindowSpec, rest_reference
from stages.s3_physics.anchors import ANCHOR_NAMES, WALKING, window_anchors

# Halve the rate: a genuine bandwidth cut, not timestamp quantization at the same rate.
AUDIT_FACTOR = 2

# Change above this => the anchor tracks the grid, not the body.
AUDIT_TOL = 0.10

# How each anchor's change is measured. Mixing the two measures the wrong thing (§11.1):
# an absolute delta on a ratio-scale magnitude is meaningless, and a relative delta on a
# correlation blows up whenever the correlation passes through zero.
ANCHOR_METRIC = {
    "periodicity": "abs",   # normalized autocorr, [0, 1]
    "antiphase": "abs",     # -pearson r, [-1, 1]
    "grav_stab": "abs",     # stability score, (0, 1]
    "gyro_energy": "rel",   # sum of deg^2/s^2, ratio scale
}

# Denominator floor for ratio-scale anchors, so a near-zero native value cannot blow the
# delta up and report a rate dependence that is really a division artefact.
ANCHOR_FLOOR = {"gyro_energy": 1.0}

# Context taken from the segment either side of a window before decimating, in samples.
# `scipy.signal.decimate(..., ftype='fir')` builds a 2*10*factor+1 = 41-tap filter and
# `zero_phase` runs it through `filtfilt`, whose default padlen is 3*41 = 123. 128 clears
# that with room to spare. Windows without full context on BOTH sides are skipped rather
# than partially padded: a half-padded window would carry the artefact on one edge only,
# which is harder to reason about than not measuring it at all. The skipped count is
# reported, never silent.
AUDIT_PAD_SAMPLES = 128


def decimate_window(win: pd.DataFrame, factor: int = AUDIT_FACTOR) -> tuple[pd.DataFrame, float]:
    """Halve one window's rate with S1's anti-aliasing FIR (§2.5 — never `[::2]`).

    `zero_phase=True` so the filter moves no peak in time; if it did, the audit would be
    measuring its own filter's group delay and calling it a rate dependence.

    NOTE: on a bare window this leaves edge transients across a large fraction of the
    samples — see the module docstring. `audit_anchors` does not call this; it decimates
    a padded span through `_decimate_with_context`. Kept because it is the honest
    definition of "decimate this window", and callers outside the audit may want it.
    """
    cols = {c: decimate(win[c].to_numpy(float), factor, ftype=DECIMATE_FILTER,
                        zero_phase=True) for c in FEATURES}
    return pd.DataFrame(cols), CANONICAL_HZ / factor


def _decimate_with_context(seg: pd.DataFrame, start: int, n: int,
                           factor: int, pad: int) -> pd.DataFrame | None:
    """Decimate `seg[start:start+n]` using `pad` samples of real context either side.

    Returns None when the window sits too close to a segment boundary to be padded on
    both sides. Context is taken from within the segment ONLY — reaching across a gap
    would feed the filter time that was never recorded (§3.1).
    """
    a0, b0 = start - pad, start + n + pad
    if a0 < 0 or b0 > len(seg):
        return None
    lead, keep = pad // factor, n // factor
    big = {c: decimate(seg[c].to_numpy(float)[a0:b0], factor,
                       ftype=DECIMATE_FILTER, zero_phase=True) for c in FEATURES}
    if min(len(v) for v in big.values()) < lead + keep:
        return None
    return pd.DataFrame({c: v[lead:lead + keep] for c, v in big.items()})


def _delta(anchor: str, native: float, decimated: float) -> float:
    native, decimated = float(native), float(decimated)
    if ANCHOR_METRIC[anchor] == "abs":
        return abs(native - decimated)
    return abs(native - decimated) / (abs(native) + ANCHOR_FLOOR[anchor])


def audit_anchors(trials: list[Trial], spec: WindowSpec | None = None,
                  factor: int = AUDIT_FACTOR, tol: float = AUDIT_TOL,
                  pad: int = AUDIT_PAD_SAMPLES) -> dict[str, dict]:
    """One verdict per anchor. Never raises on a moved anchor — it reports.

    Iterates segments directly, because padding a window requires the samples around
    it. The grid is the same `range(0, len(seg) - spec.n + 1, spec.step)` walk that
    `features.windows_of_trial` and `anchors.trial_anchors` perform.
    """
    spec = spec or WindowSpec()
    deltas: dict[str, list[float]] = {a: [] for a in ANCHOR_NAMES}
    n_total = n_walking = n_unpadded = 0

    for trial in trials:
        frame = trial.frame.reset_index(drop=True)
        if frame.empty:
            continue
        # The same rest zero the anchor table and the S2 features are centred on, so the
        # audit never reads a trial on a different origin than the stage it audits (§10.4).
        _zeros, center, _trusted = rest_reference(frame, spec.fs_hz)
        for _seg_id, seg in frame.groupby("segment", sort=True):
            seg = seg.reset_index(drop=True)
            if len(seg) < spec.n:
                continue
            for start in range(0, len(seg) - spec.n + 1, spec.step):
                win = seg.iloc[start:start + spec.n]
                native = window_anchors(win, spec.fs_hz, center)
                n_total += 1
                if native["swap_verdict"] != WALKING:  # audit only where anchors describe
                    continue
                n_walking += 1
                dec = _decimate_with_context(seg, start, spec.n, factor, pad)
                if dec is None or len(dec) < 8:
                    n_unpadded += 1  # too close to a segment edge to pad both sides
                    continue
                decd = window_anchors(dec, spec.fs_hz / factor, center)
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
            "n_windows_walking": n_walking,
            "n_windows_audited": len(deltas[a]),
            "n_skipped_unpaddable": n_unpadded,
            "pad_samples": pad,
        }
    return report


def render(report: dict[str, dict]) -> str:
    r0 = report[ANCHOR_NAMES[0]]
    lines = [
        "## Rate-invariance audit", "",
        f"Every window decimated {CANONICAL_HZ:.0f} Hz -> {CANONICAL_HZ / AUDIT_FACTOR:.0f} Hz "
        f"through S1's anti-aliasing FIR, anchors recomputed, tolerance {AUDIT_TOL:g}. "
        f"Gated to the {r0['n_windows_walking']:,} windows the swap rule calls WALKING "
        f"(of {r0['n_windows_total']:,}) — standing has no cadence to compare, and the "
        f"gate keeps the audit label-free.", "",
        f"Each window is decimated with **{r0['pad_samples']} samples of real context** "
        f"either side, then trimmed back: on a bare window the filter's edge transient "
        f"covers over half the samples and reads as a rate dependence that is not there "
        f"(it condemned `antiphase` at Δ 0.1734 vs 0.0005 corrected). "
        f"{r0['n_skipped_unpaddable']:,} windows sat too close to a segment boundary to "
        f"pad on both sides and were skipped; {r0['n_windows_audited']:,} were audited.", "",
        "| anchor | metric | median Δ | p90 Δ | verdict |", "|---|---|---|---|---|",
    ]
    for a in ANCHOR_NAMES:
        r = report[a]
        p90 = f"{r['p90_delta']:.4f}" if r["p90_delta"] is not None else "—"
        lines.append(f"| `{a}` | {r['metric']} | {r['median_delta']:.4f} | {p90} "
                     f"| **{r['verdict']}** |")
    return "\n".join(lines)
