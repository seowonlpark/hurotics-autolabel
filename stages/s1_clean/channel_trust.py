# per-file gyro trust + unit normalization, all MEASURED; sagittality is transform.py's, not ours

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

# the two units gyro arrives in; detection picks the nearer slope
_UNIT_SLOPES = {"deg/s": 1.0, "rad/s": 1.0 / RAD2DEG}


# nearest in log space- 1.0 vs 1/57.3 are ~1.76 decades apart
def _classify_unit(slope: float) -> str:
    a = abs(slope)
    if a == 0 or not math.isfinite(a):
        return CANONICAL_GYRO_UNIT
    return min(_UNIT_SLOPES, key=lambda u: abs(math.log10(a) - math.log10(_UNIT_SLOPES[u])))


# derivative is taken WITHIN each segment, never across a gap; Time is ms
def _pooled(df: pd.DataFrame, side: str) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]] | None:
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


# one side's Deg->Gyro permutation + unit from the data; None if absent, else detected/fallback
def detect_side(df: pd.DataFrame, side: str) -> dict | None:
    got = _pooled(df, side)
    if got is None:
        return None
    ddeg, gyro = got

    # r-floor PER AXIS, not once per side: Deg_Z is routinely noise and would launder into an anomaly
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

    # unit is per-side, so the best-resolved axis is the best evidence for it
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

    # only axes that answered may contradict- abstaining is silent, not dissenting
    conflicts = [A for A in resolved if match[A] != DOCUMENTED_GYRO_PERMUTATION[A]]
    doc_unit = DOCUMENTED_GYRO_UNIT.get(side)
    return {
        # None = abstained; NOT sagittality
        "gyro_axis_by_deg_axis": match,
        "resolved_deg_axes": resolved,
        "r_by_deg_axis": {A: round(r, 4) for A, r in r_by_axis.items()},
        # only claimable when all three answered- two axes on one gyro is degenerate, not exotic
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


# flag Deg channels tracking session time- oscillating ~0, drifting ~1; never drops
def detect_drift(df: pd.DataFrame) -> dict:
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


# unit-only: axes are NOT reordered and sign is left intact, both recorded instead of flipped
def detect_and_normalize(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
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
