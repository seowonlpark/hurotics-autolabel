# the raw -> rev2 feature bridge: reproduce the labeled columns from a raw CSV
# reproduces HUROTICS' MATLAB (LPF.m / csv2mat.m) exactly, verified to ~1e-13 (§6.2);
# drift from the training features is silent train/serve skew, so verify_transform.py
# regression-tests it. two traps: the filter is CAUSAL (filtfilt would be wrong), and the
# path always reads the sagittal/Y plane, flagging anomalies rather than guessing (§6.3).
# permutation (measured per file) is separate from the axis choice (§4.1b/§6.3). see README.

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from stages.s1_clean.config import DOCUMENTED_GYRO_PERMUTATION

# csv2mat.m: f_ang = 1, f_angvel = 1 (for locomotion); f_angvel = 10 is the GCP
# setting and must NOT be used here
FC_ANG_HZ = 1.0
FC_ANGVEL_HZ = 1.0

# policy (2026-07-21): the raw->features path ALWAYS reads the sagittal/Y plane (Deg_Y +
# its rate Gyro_Z). every export is sagittal; the apparent per-header "convention" was a
# raw-axis naming inconsistency, not recoverable from any in-file signal (§6.3). anomalies
# are flagged (check_axis_trust / drift), never guessed. see README / DOMAIN_NOTES.
SAGITTAL_DEG_AXIS = "Y"
SAGITTAL_GYRO_AXIS = DOCUMENTED_GYRO_PERMUTATION[SAGITTAL_DEG_AXIS] # "Z", via X->X,Y->Z,Z->Y

FEATURE_COLUMNS = ("L_ang_LPF", "R_ang_LPF", "L_angvel_LPF", "R_angvel_LPF")


# raised when a file's measured permutation breaks on the sagittal Deg_Y axis this path
# reads -- refuse rather than read a channel that isn't the rate it claims to be (§4.1b)
class AxisConflictError(Exception):
    pass


# explicit opt-out for callers with no per-file trust record; a sentinel not None so
# skipping the check is a decision at the call site, not a silent default
TRUST_UNCHECKED = "trust_unchecked"


# the MATLAB filter coefficient: a = 2*pi*dt*fc / (2*pi*dt*fc + 1)
def alpha(dt_s: float, fc_hz: float) -> float:
    w = 2.0 * math.pi * dt_s * fc_hz
    return w / (w + 1.0)


# first-order causal IIR, seeded y[0] = x[0]; exact transcription of LPF.m
def lpf(x: np.ndarray, dt_s: float, fc_hz: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x.copy()
    a = alpha(dt_s, fc_hz)
    # lfilter state form: zi = (1-a)*y[-1]; seeding with x[0] gives y[0] = x[0]
    y, _ = lfilter([a], [1.0, -(1.0 - a)], x, zi=[(1.0 - a) * x[0]])
    return y


# the dt their pipeline actually uses: the LAST inter-sample interval, in seconds (§6.2)
# reproduced to match the training features; see safe_dt for the honest version
def matlab_dt(time_ms: np.ndarray) -> float:
    t = np.asarray(time_ms, dtype=float)
    return float(t[-1] - t[-2]) / 1000.0


# median interval -- what timestamp.m should have used; robust to a bad final dt
def safe_dt(time_ms: np.ndarray) -> float:
    t = np.asarray(time_ms, dtype=float)
    return float(np.median(np.diff(t))) / 1000.0


# the channel_trust.json S1 wrote beside a raw file's clean parquet; missing is an error
# (the file was never cleaned), not a permissive empty default
def load_trust(raw_path: Path, repo_root: Path | None = None) -> dict:
    root = repo_root or Path(__file__).resolve().parents[2]
    p = (root / "data" / "clean" / raw_path.parent.name /
         f"{raw_path.stem}.channel_trust.json")
    if not p.exists():
        raise FileNotFoundError(
            f"no channel_trust record at {p} for {raw_path.name}. The file was never "
            f"cleaned (quarantined, or S1 has not run). Run S1, or pass "
            f"TRUST_UNCHECKED if skipping the axis check is genuinely intended."
        )
    return json.loads(p.read_text(encoding="utf-8"))


# refuse a file whose measured permutation breaks on the sagittal Deg_Y axis; only L/R
# matter (the sides raw_to_features reads). trust is the file's channel_trust.json
def check_axis_trust(trust: dict | str) -> None:
    if trust is TRUST_UNCHECKED:
        return
    if not isinstance(trust, dict):
        raise TypeError(
            f"trust must be a channel_trust record or TRUST_UNCHECKED, got {type(trust)}"
        )
    broken = {}
    for side in ("L", "R"):
        rec = trust.get("sides", {}).get(side)
        if rec and SAGITTAL_DEG_AXIS in rec.get("conflicts_with_documented", []):
            broken[side] = rec.get("gyro_axis_by_deg_axis", {}).get(SAGITTAL_DEG_AXIS)
    if broken:
        detail = ", ".join(f"{s}_Deg_{SAGITTAL_DEG_AXIS} -> {s}_Gyro_{g}" for s, g in broken.items())
        raise AxisConflictError(
            f"this file's measured permutation breaks on the sagittal axis: {detail}, but "
            f"the documented map says Deg_{SAGITTAL_DEG_AXIS} -> Gyro_{SAGITTAL_GYRO_AXIS}. "
            f"Reading it would feed the classifier a channel it was not trained on, and "
            f"nothing downstream would notice. Resolve by hand (DOMAIN_NOTES §4.1b) — do "
            f"not suppress this."
        )


# build the four rev2 features from a name-resolved raw frame, always from the sagittal
# Deg_Y plane + its rate Gyro_Z (§6.3). dt_s defaults to the median; pass matlab_dt() to
# reproduce the training pipeline bit-for-bit. trust is required (TRUST_UNCHECKED to skip)
def raw_to_features(df: pd.DataFrame, *, trust: dict | str,
                    dt_s: float | None = None,
                    time_col: str = "Time") -> pd.DataFrame:
    check_axis_trust(trust)
    if dt_s is None:
        dt_s = safe_dt(df[time_col].to_numpy(float))

    out = {}
    for side in ("L", "R"):
        deg_col = f"{side}_Deg_{SAGITTAL_DEG_AXIS}"
        gyro_col = f"{side}_Gyro_{SAGITTAL_GYRO_AXIS}"
        for col in (deg_col, gyro_col):
            if col not in df.columns:
                raise KeyError(f"raw frame is missing {col!r}")
        out[f"{side}_ang_LPF"] = lpf(df[deg_col].to_numpy(float), dt_s, FC_ANG_HZ)
        out[f"{side}_angvel_LPF"] = lpf(df[gyro_col].to_numpy(float), dt_s, FC_ANGVEL_HZ)

    return pd.DataFrame({time_col: df[time_col].to_numpy(float), **out})
