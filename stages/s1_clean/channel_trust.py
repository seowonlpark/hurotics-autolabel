# per-file gyro trust + unit normalization
# resolves the Deg->Gyro axis permutation and unit by MEASUREMENT (regress d(Deg)/dt
# against each Gyro axis), then normalizes every gyro channel to deg/s.
# does NOT determine sagittality -- that's a hardware fact with no in-file signature;
# transform.py fixes the sagittal read to the Y plane for every file (Section 6.3). too-static
# files abstain to the documented convention. see DOMAIN_NOTES Section 4.1/4.1b/6.2 and README.

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

# the two units gyro can arrive in; detection picks whichever slope is nearer
_UNIT_SLOPES = {"deg/s": 1.0, "rad/s": 1.0 / RAD2DEG}


# nearest unit by |slope| in log space (1.0 vs 1/57.3 are ~1.76 decades apart)
def _classify_unit(slope: float) -> str:
    a = abs(slope)
    if a == 0 or not math.isfinite(a):
        return CANONICAL_GYRO_UNIT
    return min(_UNIT_SLOPES, key=lambda u: abs(math.log10(a) - math.log10(_UNIT_SLOPES[u])))


# pool d(Deg_A)/dt [deg/s] and each Gyro axis for one side; derivative is taken within
# each gap-free segment, never across a gap, so a segment column is required
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


# pearson r; nan when either series is constant or too short
def _corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


# resolve one side's Deg->Gyro axis permutation + unit from the data (NOT sagittality)
# None if the side's channels are absent; else method 'detected' or 'fallback_documented'
def detect_side(df: pd.DataFrame, side: str) -> dict | None:
    got = _pooled(df, side)
    if got is None:
        return None
    ddeg, gyro = got

    # for each Deg axis, the best-matching Gyro axis and its r.
    # r-floor is applied PER AXIS, not once to the side: Deg_Z is yaw-like and drifts
    # rather than oscillates (Section 4.2), so d(Deg_Z)/dt is often noise even mid-walk --
    # scoring the whole side would launder that into a fake anomaly (Section 11.1)
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

    # unit is a per-side property, so read it off the strongest resolved pair
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

    # only axes that actually answered may contradict the documented map;
    # an abstaining axis is silent, not dissenting
    conflicts = [A for A in resolved if match[A] != DOCUMENTED_GYRO_PERMUTATION[A]]
    doc_unit = DOCUMENTED_GYRO_UNIT.get(side)
    return {
        # Deg axis -> Gyro axis measuring its rate; None = abstained. NOT sagittality
        "gyro_axis_by_deg_axis": match,
        "resolved_deg_axes": resolved,
        "r_by_deg_axis": {A: round(r, 4) for A, r in r_by_axis.items()},
        # a permutation is a bijection; claimable only when all three axes answered
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


# flag Deg channels that track session time (integration drift, not orientation):
# duration-weighted |corr(Deg, Time)| over long-enough segments. flags, never drops (Section 4.2)
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


# detect trust per side, normalize every gyro channel to deg/s
# unit-only (a per-side scalar); axes are NOT reordered, the permutation is just recorded;
# sign is left intact and carried in `r` so polarity is never silently flipped
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
