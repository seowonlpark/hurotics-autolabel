"""The raw -> rev2 feature bridge: reproduce the labeled columns from a raw CSV.

The model trains on the labeled rev2 view (four rotational features) but must RUN on
raw device CSVs. This module is the bridge, and it reproduces HUROTICS' MATLAB
(`LPF.m` / `csv2mat.m`) exactly — verified to ~1e-13 against 19 paired recordings
(DOMAIN_NOTES §6.2). Any drift between this and the training features is silent
train/serve skew, so it is regression-tested against those pairs.

Two things here are easy to get wrong:

  1. **The filter is CAUSAL.** First-order IIR, one pole, fc = 1 Hz. It has phase lag
     by construction. `filtfilt` would be "better" signal processing and WRONG here —
     it would shift the features relative to the labels the model learned.
  2. **The sagittal axis is a DEVICE property, not a signal property.** It is fixed by
     how the IMU sat in that hardware revision, so it is resolved by schema variant,
     never guessed from the waveform. Signal-only detection was measured and scored
     BELOW CHANCE (§6.2) — do not reintroduce it. Unknown variant => abstain.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.signal import lfilter

# csv2mat.m: f_ang = 1; f_angvel = 1;  ("for locomotion classification").
# f_angvel = 10 is the GCP variant and must NOT be used for this task.
FC_ANG_HZ = 1.0
FC_ANGVEL_HZ = 1.0

# variant_id -> (Deg axis for *_ang_LPF, Gyro axis for *_angvel_LPF).
# Measured on paired recordings, 19/19 exact (§6.2). Extend only with new evidence:
# one paired raw+annotated file for the revision, never a guess.
SAGITTAL_AXIS_BY_VARIANT = {
    "fb5ea2c2": ("X", "X"),   # rev13 / rev14 — the §4.1b anomaly family
    "0fda484e": ("Y", "Z"),   # rev7 / rev8   — the documented Y<->Z convention
    "4bfd6ab2": ("Y", "Z"),   # rev4          — same convention
}

FEATURE_COLUMNS = ("L_ang_LPF", "R_ang_LPF", "L_angvel_LPF", "R_angvel_LPF")


class UnknownVariantError(Exception):
    """Raised when a file's schema variant has no measured axis mapping.

    Deliberately fatal rather than defaulted: guessing the axis feeds the classifier a
    channel that is not the one it was trained on, and nothing downstream would notice.
    """


def alpha(dt_s: float, fc_hz: float) -> float:
    """The MATLAB coefficient: a = 2*pi*dt*fc / (2*pi*dt*fc + 1)."""
    w = 2.0 * math.pi * dt_s * fc_hz
    return w / (w + 1.0)


def lpf(x: np.ndarray, dt_s: float, fc_hz: float) -> np.ndarray:
    """First-order causal IIR, seeded y[0] = x[0]. Exact transcription of LPF.m.

    y[n] = a*x[n] + (1-a)*y[n-1]
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x.copy()
    a = alpha(dt_s, fc_hz)
    # lfilter state form: zi = (1-a)*y[-1]; seeding with x[0] gives y[0] = x[0].
    y, _ = lfilter([a], [1.0, -(1.0 - a)], x, zi=[(1.0 - a) * x[0]])
    return y


def matlab_dt(time_ms: np.ndarray) -> float:
    """The dt their pipeline actually uses: the LAST inter-sample interval, in seconds.

    `timestamp.m` loops the whole time vector but overwrites `del_t` each pass, so only
    the final interval survives — and `csv2mat.m` passes the full vector once, so one dt
    filters the entire trial (§6.2). Reproduced here because matching the training
    features matters more than being correct; see `safe_dt` for the honest version.
    """
    t = np.asarray(time_ms, dtype=float)
    return float(t[-1] - t[-2]) / 1000.0


def safe_dt(time_ms: np.ndarray) -> float:
    """Median interval — what `timestamp.m` should have used.

    Robust to a single bad final interval, which in the MATLAB can be dt=0 and freeze
    the filter output flat for a whole trial. Use this on the canonical grid, where it
    equals the nominal dt anyway.
    """
    t = np.asarray(time_ms, dtype=float)
    return float(np.median(np.diff(t))) / 1000.0


def resolve_axes(variant_id: str) -> tuple[str, str]:
    """(deg_axis, gyro_axis) for a schema variant, or raise. Never guesses."""
    axes = SAGITTAL_AXIS_BY_VARIANT.get(variant_id)
    if axes is None:
        raise UnknownVariantError(
            f"no measured sagittal-axis mapping for variant {variant_id!r}. "
            f"Known: {sorted(SAGITTAL_AXIS_BY_VARIANT)}. Signal-based detection scores "
            f"below chance (DOMAIN_NOTES 6.2) — add a mapping from one paired "
            f"raw+annotated recording instead of guessing."
        )
    return axes


def raw_to_features(df: pd.DataFrame, variant_id: str, dt_s: float | None = None,
                    time_col: str = "Time") -> pd.DataFrame:
    """Build the four rev2 features from a name-resolved raw device frame.

    `dt_s` defaults to the median interval; pass `matlab_dt(...)` to reproduce the
    training pipeline bit-for-bit on an un-resampled raw file.
    """
    deg_axis, gyro_axis = resolve_axes(variant_id)
    if dt_s is None:
        dt_s = safe_dt(df[time_col].to_numpy(float))

    out = {}
    for side in ("L", "R"):
        deg_col, gyro_col = f"{side}_Deg_{deg_axis}", f"{side}_Gyro_{gyro_axis}"
        for col in (deg_col, gyro_col):
            if col not in df.columns:
                raise KeyError(f"raw frame is missing {col!r} (variant {variant_id})")
        out[f"{side}_ang_LPF"] = lpf(df[deg_col].to_numpy(float), dt_s, FC_ANG_HZ)
        out[f"{side}_angvel_LPF"] = lpf(df[gyro_col].to_numpy(float), dt_s, FC_ANGVEL_HZ)

    return pd.DataFrame({time_col: df[time_col].to_numpy(float), **out})
