"""Per-file gyro trust + unit normalization.

Gyro is reliable — it is the derivative of angle (DOMAIN_NOTES 4.1) — but its unit
and sagittal axis vary within a single file (4.1b). This module resolves both by
MEASUREMENT, not by asserting the documented table: for each side it regresses
d(Deg_Y)/dt (deg/s) against every Gyro axis. The strongest-correlated axis is the
sagittal gyro; the regression slope reveals the native unit (~1 -> deg/s,
~1/57.3 -> rad/s). It then normalizes every gyro channel to deg/s.

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
    DOCUMENTED_GYRO_UNIT,
    DOCUMENTED_SAGITTAL_GYRO_AXIS,
    DRIFT_MIN_SEGMENT_S,
    GYRO_AXES,
    RAD2DEG,
    SAGITTAL_DEG_AXIS,
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


def _pooled(df: pd.DataFrame, side: str) -> tuple[np.ndarray, dict[str, np.ndarray]] | None:
    """Pool d(Deg_Y)/dt [deg/s] and each Gyro axis across segments for one side.

    The derivative is computed within each gap-free segment — never across a gap —
    so a segment column is required. Time is milliseconds.
    """
    deg_col = f"{side}_Deg_{SAGITTAL_DEG_AXIS}"
    gyro_cols = {a: f"{side}_Gyro_{a}" for a in GYRO_AXES}
    if deg_col not in df.columns or not all(c in df.columns for c in gyro_cols.values()):
        return None

    dY, gyro = [], {a: [] for a in GYRO_AXES}
    for _, seg in df.groupby("segment", sort=True):
        if len(seg) < 3:
            continue
        t_s = seg["Time"].to_numpy(float) / 1000.0
        dY.append(np.gradient(seg[deg_col].to_numpy(float), t_s))
        for a in GYRO_AXES:
            gyro[a].append(seg[gyro_cols[a]].to_numpy(float))
    if not dY:
        return None
    return np.concatenate(dY), {a: np.concatenate(v) for a, v in gyro.items()}


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def detect_side(df: pd.DataFrame, side: str) -> dict | None:
    """Resolve one side's sagittal gyro axis + unit from the data.

    Returns None when the side's channels are absent. Otherwise a record whose
    `method` is 'detected' (confident) or 'fallback_documented' (too static).
    """
    got = _pooled(df, side)
    if got is None:
        return None
    dY, gyro = got

    best_axis, best_r = None, 0.0
    for a in GYRO_AXES:
        r = _corr(dY, gyro[a])
        if np.isfinite(r) and abs(r) > abs(best_r):
            best_axis, best_r = a, r

    confident = best_axis is not None and abs(best_r) >= TRUST_R_FLOOR
    if confident:
        slope = float(np.polyfit(dY, gyro[best_axis], 1)[0])
        unit = _classify_unit(slope)
        axis, method = best_axis, "detected"
    else:
        slope = float("nan")
        unit = DOCUMENTED_GYRO_UNIT.get(side, CANONICAL_GYRO_UNIT)
        axis, method = DOCUMENTED_SAGITTAL_GYRO_AXIS, "fallback_documented"

    doc_axis = DOCUMENTED_SAGITTAL_GYRO_AXIS
    doc_unit = DOCUMENTED_GYRO_UNIT.get(side)
    return {
        "sagittal_gyro_axis": axis,
        "unit": unit,
        "scale_to_degps": UNIT_TO_DEGPS_SCALE[unit],
        "r": round(best_r, 4),
        "slope": None if not np.isfinite(slope) else round(slope, 6),
        "n": int(dY.size),
        "confident": confident,
        "method": method,
        "matches_documented": (axis == doc_axis and unit == doc_unit),
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
    axes are NOT reordered — the resolved sagittal axis is recorded for downstream.
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
