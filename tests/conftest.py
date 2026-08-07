# synthetic frames only- these tests never read data/raw, so they run on a clean checkout

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stages.s1_clean.config import DOCUMENTED_GYRO_PERMUTATION, GYRO_AXES, RAD2DEG, SIDES

# one distinct frequency per axis so the three deg axes cannot be mistaken for each other
_AXIS_HZ = {"X": 0.3, "Y": 0.7, "Z": 1.1}


# gyro is the analytic derivative of deg routed through the documented permutation- detection should agree
def _device_frame(secs=30.0, sides=SIDES, unit="deg/s", hz=100.0):
    n = int(secs * hz)
    t_ms = np.arange(n) * (1000.0 / hz)
    t_s = t_ms / 1000.0
    scale = 1.0 if unit == "deg/s" else 1.0 / RAD2DEG
    cols = {"Time": t_ms, "segment": np.zeros(n, dtype=int)}
    for side in sides:
        for A in GYRO_AXES:
            w = 2 * np.pi * _AXIS_HZ[A]
            cols[f"{side}_Deg_{A}"] = 30.0 * np.sin(w * t_s)
            cols[f"{side}_Gyro_{DOCUMENTED_GYRO_PERMUTATION[A]}"] = 30.0 * w * np.cos(w * t_s) * scale
    return pd.DataFrame(cols)


# a post-resample frame: Time in ms, a segment column, and a matched Deg/Gyro pair per side
@pytest.fixture
def device_frame():
    return _device_frame
