# boundary conditions for gyro trust; abstention is the interesting path, not the happy one

from __future__ import annotations

import numpy as np

from stages.s1_clean.channel_trust import _corr, detect_and_normalize, detect_drift, detect_side
from stages.s1_clean.config import (
    DRIFT_MIN_SEGMENT_S,
    GYRO_AXES,
    RAD2DEG,
    TRUST_R_FLOOR,
    YAW_DRIFT_R_FLOOR,
)


def test_documented_permutation_is_detected_from_the_data(device_frame):
    rec = detect_side(device_frame(), "L")
    assert rec["confident"] and rec["is_bijection"]
    assert rec["matches_documented_permutation"] and rec["conflicts_with_documented"] == []
    assert abs(rec["r"]) >= TRUST_R_FLOOR


def test_rad_per_s_is_detected_and_scaled_to_degps(device_frame):
    df = device_frame(unit="rad/s")
    out, trust = detect_and_normalize(df)
    assert trust["sides"]["L"]["unit"] == "rad/s"
    assert trust["sides"]["L"]["scale_to_degps"] == RAD2DEG
    # the scale is applied, not just recorded- otherwise downstream reads rad as deg
    assert np.allclose(out["L_Gyro_X"], df["L_Gyro_X"] * RAD2DEG)


def test_an_absent_side_returns_none_rather_than_abstaining(device_frame):
    assert detect_side(device_frame(sides=("L",)), "R") is None


def test_segments_under_three_rows_contribute_nothing(device_frame):
    assert detect_side(device_frame(secs=0.02), "L") is None


def test_an_all_nan_side_falls_back_to_documented(device_frame):
    df = device_frame()
    for A in GYRO_AXES:
        df[f"L_Deg_{A}"] = np.nan
    rec = detect_side(df, "L")
    assert not rec["confident"] and rec["method"] == "fallback_documented"
    assert rec["resolved_deg_axes"] == [] and rec["slope"] is None


def test_a_static_axis_abstains_instead_of_inventing_a_match(device_frame):
    df = device_frame()
    df["L_Deg_Y"] = 0.0
    rec = detect_side(df, "L")
    # Y answers nothing, but abstaining is silent- it must not read as a conflict
    assert rec["gyro_axis_by_deg_axis"]["Y"] is None
    assert not rec["is_bijection"] and rec["conflicts_with_documented"] == []


def test_a_ramping_deg_channel_is_flagged_as_drift(device_frame):
    df = device_frame(secs=30.0)
    df["L_Deg_Z"] = df["Time"] / 1000.0
    drift = detect_drift(df)
    assert "L_Deg_Z" in drift["contaminated"]
    assert drift["sides"]["L"]["Z"]["time_corr_absr"] >= YAW_DRIFT_R_FLOOR
    assert "L_Deg_X" not in drift["contaminated"]


def test_drift_is_inconclusive_below_the_segment_floor(device_frame):
    drift = detect_drift(device_frame(secs=DRIFT_MIN_SEGMENT_S - 1.0))
    axis = drift["sides"]["L"]["X"]
    assert axis["inconclusive"] and axis["time_corr_absr"] is None
    assert drift["contaminated"] == []


def test_corr_refuses_a_constant_or_too_short_input():
    assert np.isnan(_corr(np.arange(2.0), np.arange(2.0)))
    assert np.isnan(_corr(np.zeros(50), np.arange(50.0)))


def test_a_single_bad_row_does_not_sink_the_side(device_frame):
    df = device_frame()
    df.loc[100, [f"L_Deg_{A}" for A in GYRO_AXES]] = np.nan
    rec = detect_side(df, "L")
    assert rec["confident"] and rec["matches_documented_permutation"]
    # np.gradient smears one bad row over its neighbours, so a handful of samples go, not the side
    assert rec["n"] >= len(df) - 10


def test_the_fit_reports_the_samples_it_actually_saw(device_frame):
    clean = detect_side(device_frame(), "L")
    df = device_frame()
    df.loc[100, [f"L_Deg_{A}" for A in GYRO_AXES]] = np.nan
    assert detect_side(df, "L")["n"] < clean["n"]
