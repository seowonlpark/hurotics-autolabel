"""Per-file gyro trust + unit normalization.

Gyro is reliable — it is the derivative of angle (DOMAIN_NOTES 4.1) — but its unit
is inconsistent within a single file, and the Gyro axis LABELS are permuted relative
to the Deg labels (4.1b). This module resolves both by MEASUREMENT, not by asserting
the documented table: for each side it regresses d(Deg_A)/dt (deg/s) against every
Gyro axis, for every Deg axis A. The strongest-correlated gyro axis is A's
counterpart; the regression slope reveals the native unit (~1 -> deg/s, ~1/57.3 ->
rad/s). It then normalizes every gyro channel to deg/s.

**This module does not know what "sagittal" means, and must not pretend to.** It
measures a correspondence *internal* to the file: which Gyro axis measures the rate
of which Deg axis. Which axis is the sagittal (flexion) plane is a property of the
hardware revision with no in-file signature — 6.2 measured every signal-only rule
for it at BELOW CHANCE — and is resolved by variant lookup in
stages/s2_ml/transform.py. The two questions were once conflated in a field called
`sagittal_gyro_axis`, which reported the gyro matching Deg_Y and was therefore wrong
on fb5ea2c2, the majority variant, where sagittal is Deg_X.

When a file is too static for d(Deg_Y)/dt to carry signal, the fit is noise
(11.1, "density needs mass"): detection abstains and falls back to the documented
convention, recording that it did so. This is also the channel-trust check PLAN S1
requires, and its record is what the exception agent will read.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from stages.s1_clean.config import (
    CANONICAL_GYRO_UNIT,
    DOCUMENTED_GYRO_PERMUTATION,
    DOCUMENTED_GYRO_UNIT,
    DRIFT_MIN_SEGMENT_S,
    GYRO_AXES,
    RAD2DEG,
    SIDES,
    TRUST_R_FLOOR,
    UNIT_TO_DEGPS_SCALE,
    YAW_DRIFT_R_FLOOR,
)

# The two units gyro can arrive in; detection picks whichever slope is nearer.
_UNIT_SLOPES = {"deg/s": 1.0, "rad/s": 1.0 / RAD2DEG}


def _classify_unit(slope: float) -> str:
    """Nearest unit by |slope| in log space (1.0 vs 1/57.3 are ~1.76 decades apart)."""
    a = abs(slope)
    if a == 0 or not math.isfinite(a):
        return CANONICAL_GYRO_UNIT
    return min(_UNIT_SLOPES, key=lambda u: abs(math.log10(a) - math.log10(_UNIT_SLOPES[u])))


def _pooled(df: pd.DataFrame, side: str) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]] | None:
    """Pool d(Deg_A)/dt [deg/s] for every axis A, and each Gyro axis, for one side.

    The derivative is computed within each gap-free segment — never across a gap —
    so a segment column is required. Time is milliseconds.
    """
    deg_cols = {a: f"{side}_Deg_{a}" for a in GYRO_AXES}
    gyro_cols = {a: f"{side}_Gyro_{a}" for a in GYRO_AXES}
    if not all(c in df.columns for c in (*deg_cols.values(), *gyro_cols.values())):
        return None

    ddeg = {a: [] for a in GYRO_AXES}
    gyro = {a: [] for a in GYRO_AXES}
    for _, seg in df.groupby("segment", sort=True):
        if len(seg) < 3:
            continue
        t_s = seg["Time"].to_numpy(float) / 1000.0
        for a in GYRO_AXES:
            ddeg[a].append(np.gradient(seg[deg_cols[a]].to_numpy(float), t_s))
            gyro[a].append(seg[gyro_cols[a]].to_numpy(float))
    if not any(ddeg[a] for a in GYRO_AXES):
        return None
    return ({a: np.concatenate(v) for a, v in ddeg.items()},
            {a: np.concatenate(v) for a, v in gyro.items()})


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def detect_side(df: pd.DataFrame, side: str) -> dict | None:
    """Resolve one side's Deg->Gyro axis permutation + unit from the data.

    Deliberately NOT "the sagittal axis". This regression can only discover which
    Gyro axis measures the rate of which Deg axis — a correspondence internal to the
    file. WHICH axis is sagittal is a hardware-revision fact with no in-file
    signature (DOMAIN_NOTES 6.2 measured every signal-only rule at below chance);
    it is resolved by variant lookup in stages/s2_ml/transform.py. Conflating the
    two is what made the old `sagittal_gyro_axis` field wrong on fb5ea2c2.

    Returns None when the side's channels are absent. Otherwise a record whose
    `method` is 'detected' (confident) or 'fallback_documented' (too static).
    """
    got = _pooled(df, side)
    if got is None:
        return None
    ddeg, gyro = got

    # For each Deg axis, the best-matching Gyro axis and its r.
    #
    # The r-floor is applied PER AXIS, not once to the side. A Deg axis whose
    # derivative carries no signal has no argmax worth reading, and Deg_Z is
    # routinely exactly that: it is the yaw-like axis, which drifts rather than
    # oscillates (4.2), so d(Deg_Z)/dt is often noise even in a file that is
    # vigorously walking. Scoring the side as a whole and then reporting all three
    # axes would launder that noise into a confident-looking anomaly — the 11.1
    # "density needs mass" failure, one level down.
    match: dict[str, str | None] = {}
    r_by_axis: dict[str, float] = {}
    for A in GYRO_AXES:
        best_axis, best_r = None, 0.0
        for a in GYRO_AXES:
            r = _corr(ddeg[A], gyro[a])
            if np.isfinite(r) and abs(r) > abs(best_r):
                best_axis, best_r = a, r
        r_by_axis[A] = best_r
        match[A] = best_axis if (best_axis and abs(best_r) >= TRUST_R_FLOOR) else None

    resolved = [A for A in GYRO_AXES if match[A] is not None]

    # Unit comes from the strongest resolved pair — unit is a per-side property,
    # so the best-resolved axis is the best evidence for it.
    anchor = max(resolved, key=lambda A: abs(r_by_axis[A])) if resolved else None
    best_r = r_by_axis.get(anchor, 0.0) if anchor else 0.0
    confident = anchor is not None

    if confident:
        slope = float(np.polyfit(ddeg[anchor], gyro[match[anchor]], 1)[0])
        unit = _classify_unit(slope)
        method = "detected"
    else:
        slope = float("nan")
        unit = DOCUMENTED_GYRO_UNIT.get(side, CANONICAL_GYRO_UNIT)
        method = "fallback_documented"

    # Only axes that actually answered may contradict the documented map. An
    # abstaining axis is silent, not dissenting.
    conflicts = [A for A in resolved if match[A] != DOCUMENTED_GYRO_PERMUTATION[A]]
    doc_unit = DOCUMENTED_GYRO_UNIT.get(side)
    return {
        # Deg axis -> the Gyro axis measuring its rate; None = abstained, too
        # little signal on that axis to read. NOT sagittality (see module docstring).
        "gyro_axis_by_deg_axis": match,
        "resolved_deg_axes": resolved,
        "r_by_deg_axis": {A: round(r, 4) for A, r in r_by_axis.items()},
        # A permutation is a bijection; claimable only when all three axes answered.
        # Two resolved Deg axes pointing at one Gyro axis is a degenerate detection,
        # not an exotic device.
        "is_bijection": (len(resolved) == len(GYRO_AXES)
                         and len({match[A] for A in resolved}) == len(GYRO_AXES)),
        "conflicts_with_documented": conflicts,
        "matches_documented_permutation": not conflicts,
        "anchor_deg_axis": anchor,
        "unit": unit,
        "scale_to_degps": UNIT_TO_DEGPS_SCALE[unit],
        "r": round(best_r, 4),
        "slope": None if not np.isfinite(slope) else round(slope, 6),
        "n": int(ddeg[anchor].size) if anchor else 0,
        "confident": confident,
        "method": method,
        "matches_documented": (not conflicts and unit == doc_unit),
    }


def detect_drift(df: pd.DataFrame) -> dict:
    """Flag Deg channels that track session time (integration drift, not orientation).

    Per side, per Deg axis, the duration-weighted |corr(Deg, Time)| over segments
    long enough (>= DRIFT_MIN_SEGMENT_S) for a drift slope to mean anything. A
    sagittal channel oscillates -> near 0; a drifting channel tracks time -> near 1.
    Flags, never drops (DOMAIN_NOTES 4.2): this is a feature-time exclusion signal.
    """
    sides: dict[str, dict] = {}
    for side in SIDES:
        axes: dict[str, dict] = {}
        for a in GYRO_AXES:
            col = f"{side}_Deg_{a}"
            if col not in df.columns:
                continue
            num, den = 0.0, 0.0
            for _, seg in df.groupby("segment", sort=True):
                t = seg["Time"].to_numpy(float)
                dur = float(t[-1] - t[0]) / 1000.0
                if dur < DRIFT_MIN_SEGMENT_S:
                    continue
                r = _corr(seg[col].to_numpy(float), t)
                if np.isfinite(r):
                    num, den = num + abs(r) * dur, den + dur
            if den == 0:
                axes[a] = {"time_corr_absr": None, "drift_contaminated": False,
                           "secs": 0.0, "inconclusive": True}
            else:
                wr = num / den
                axes[a] = {"time_corr_absr": round(wr, 4),
                           "drift_contaminated": bool(wr >= YAW_DRIFT_R_FLOOR),
                           "secs": round(den, 1)}
        sides[side] = axes

    contaminated = [f"{s}_Deg_{a}" for s, ax in sides.items()
                    for a, r in ax.items() if r["drift_contaminated"]]
    return {"min_segment_s": DRIFT_MIN_SEGMENT_S, "sides": sides, "contaminated": contaminated}


def detect_and_normalize(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Detect trust per side, normalize every gyro channel to deg/s.

    Returns (normalized_df, trust). Normalization is unit-only (a per-side scalar);
    axes are NOT reordered — the resolved permutation is recorded for downstream.
    Sign is left intact and carried in `r` so polarity is never silently flipped.
    """
    out = df.copy()
    sides: dict[str, dict] = {}
    for side in SIDES:
        rec = detect_side(df, side)
        if rec is None:
            continue
        scale = rec["scale_to_degps"]
        if scale != 1.0:
            cols = [f"{side}_Gyro_{a}" for a in GYRO_AXES if f"{side}_Gyro_{a}" in out.columns]
            out[cols] = out[cols] * scale
        sides[side] = rec

    trust = {
        "canonical_gyro_unit": CANONICAL_GYRO_UNIT,
        "sides": sides,
        "abstained": [s for s, r in sides.items() if r["method"] == "fallback_documented"],
        "anomalies": [s for s, r in sides.items() if r["confident"] and not r["matches_documented"]],
        "drift": detect_drift(df),
    }
    return out, trust
