# boundary conditions for segment-and-resample; the two xfails are real holes, not style complaints

from __future__ import annotations

import numpy as np
import pandas as pd

from stages.s1_clean.config import CANONICAL_DT_MS, MIN_SEGMENT_S
from stages.s1_clean.resample import measure_hz, nominal_rate, resample_file, segment_at_gaps


# Time in ms plus one linear-interp channel- the least resample_file will accept
def _frame(t_ms, **channels):
    t = np.asarray(t_ms, dtype=float)
    cols = {"Time": t, "L_Deg_X": np.sin(t / 500.0)}
    cols.update(channels)
    return pd.DataFrame(cols)


def test_measure_hz_is_nan_on_a_degenerate_time_base():
    assert np.isnan(measure_hz(np.array([0.0])))
    assert np.isnan(measure_hz(np.array([], dtype=float)))
    assert np.isnan(measure_hz(np.array([5.0, 5.0, 5.0])))


def test_nominal_rate_snaps_only_to_known_acquisition_rates():
    assert nominal_rate(100.0) == 100.0
    assert nominal_rate(495.0) == 500.0
    assert nominal_rate(300.0) is None
    assert nominal_rate(float("nan")) is None


def test_gap_splits_where_dt_jumps():
    t = np.concatenate([np.arange(500) * 10.0, 6000.0 + np.arange(500) * 10.0])
    assert segment_at_gaps(t) == [(0, 500), (500, 1000)]
    assert all(s.usable for s in resample_file(_frame(t), "Time")[1])


def test_single_sample_is_recorded_unusable_not_crashed():
    out, segs = resample_file(_frame([0.0]), "Time")
    assert len(segs) == 1 and not segs[0].usable
    assert "matches no known acquisition rate" in segs[0].reason
    assert out.empty


def test_segment_under_the_duration_floor_is_recorded_unusable():
    out, segs = resample_file(_frame([0.0, 10.0]), "Time")
    assert not segs[0].usable and f"< {MIN_SEGMENT_S} s" in segs[0].reason
    # unusable drops from the output but survives in the table- that is the whole point of the split
    assert out.empty and segs[0].n_source_rows == 2


def test_clean_100hz_is_grid_aligned_not_resampled():
    out, segs = resample_file(_frame(np.arange(1000) * 10.0), "Time")
    assert segs[0].method == "grid_aligned" and segs[0].usable
    assert len(out) == 999
    assert np.allclose(np.diff(out["Time"]), CANONICAL_DT_MS)


def test_500hz_decimates_down_to_100hz():
    out, segs = resample_file(_frame(np.arange(5000) * 2.0), "Time")
    assert segs[0].method.startswith("decimate_5x")
    assert np.allclose(np.diff(out["Time"]), CANONICAL_DT_MS)
    assert 990 <= len(out) <= 1000


def test_label_takes_nearest_never_a_blend():
    labels = np.repeat([0.0, 1.0], 500)
    out, _ = resample_file(_frame(np.arange(1000) * 10.0, Label=labels), "Time")
    assert set(np.unique(out["Label"])) <= {0.0, 1.0}


def test_every_segment_gets_its_index_stamped():
    t = np.concatenate([np.arange(500) * 10.0, 6000.0 + np.arange(500) * 10.0])
    out, _ = resample_file(_frame(t), "Time")
    assert set(np.unique(out["segment"])) == {0, 1}


def test_empty_frame_is_not_an_indexerror():
    assert segment_at_gaps(np.array([], dtype=float)) == []
    out, segs = resample_file(pd.DataFrame({"Time": [], "L_Deg_X": []}), "Time")
    assert out.empty and segs == []


def test_a_dead_channel_is_flagged_and_the_time_base_is_still_judged_on_its_own():
    n = 3000
    df = _frame(np.arange(n) * 10.0, L_Gyro_X=np.full(n, np.nan))
    _, segs = resample_file(df, "Time")
    assert segs[0].dead_channels == ["L_Gyro_X"]
    # usable is about the clock- folding channel health into it would misfile the drop in breakdown
    assert segs[0].usable


def test_a_live_channel_is_never_flagged_dead():
    _, segs = resample_file(_frame(np.arange(3000) * 10.0), "Time")
    assert segs[0].dead_channels == []


def test_partial_nan_is_not_a_dead_channel():
    n = 3000
    v = np.sin(np.arange(n) / 50.0)
    v[100:200] = np.nan
    _, segs = resample_file(_frame(np.arange(n) * 10.0, L_Gyro_X=v), "Time")
    assert segs[0].dead_channels == []
