# breakdown buckets S1's drops by matching resample's reason PROSE- so the prose is an interface

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stages.s1_clean.resample import resample_file

REPO_ROOT = Path(__file__).resolve().parents[1]

# breakdown pulls in dataset, which reads data/corpus.json at IMPORT- skip rather than fail collection
if not (REPO_ROOT / "data" / "corpus.json").is_file():
    pytest.skip("stages.breakdown needs data/corpus.json at import time",
                allow_module_level=True)

from stages.breakdown import PREAMBLE_MAX_ROWS, RATE_REJECT_MARKERS, bucket_dropped


def _dropped(t_ms) -> list[dict]:
    t = np.asarray(t_ms, dtype=float)
    df = pd.DataFrame({"Time": t, "L_Deg_X": np.sin(t / 500.0)})
    return [s.to_dict() for s in resample_file(df, "Time")[1] if not s.usable]


# 300 Hz is a plausible-looking rate that snaps to nothing- the interesting reject, not a crash
def _rate_reject() -> list[dict]:
    return _dropped(np.arange(3000) * (1000.0 / 300.0))


# 5-row preamble, a usable run, then a 50-row fragment- the two shapes that are NOT a rate problem
def _short_drops() -> list[dict]:
    return _dropped(np.concatenate([np.arange(5) * 10.0,
                                    5000.0 + np.arange(1500) * 10.0,
                                    25000.0 + np.arange(50) * 10.0]))


def test_a_rate_reject_says_something_breakdown_recognises():
    drops = _rate_reject()
    assert len(drops) == 1
    assert any(m in drops[0]["reason"] for m in RATE_REJECT_MARKERS)


def test_a_degenerate_time_base_is_a_rate_reject_too():
    drops = _dropped([0.0])
    assert any(m in drops[0]["reason"] for m in RATE_REJECT_MARKERS)


def test_the_duration_floor_reason_carries_no_rate_marker():
    # if it ever did, every short segment would be counted as a hardware finding
    for s in _short_drops():
        assert not any(m in s["reason"] for m in RATE_REJECT_MARKERS)


def test_bucket_dropped_separates_the_three_causes():
    rate, preamble, fragments = bucket_dropped(_short_drops() + _rate_reject())
    assert len(rate) == 1 and len(preamble) == 1 and len(fragments) == 1
    assert preamble[0]["index"] == 0 and preamble[0]["n_source_rows"] <= PREAMBLE_MAX_ROWS
    assert fragments[0]["index"] == 2 and fragments[0]["n_source_rows"] == 50


def test_every_drop_lands_in_exactly_one_bucket():
    drops = _short_drops() + _rate_reject()
    buckets = bucket_dropped(drops)
    assert sum(len(b) for b in buckets) == len(drops)


def test_a_short_segment_at_index_zero_is_only_preamble_while_it_is_short():
    long_preamble = [{"index": 0, "n_source_rows": PREAMBLE_MAX_ROWS + 1, "reason": "0.5 s < 1.0 s"}]
    _, preamble, fragments = bucket_dropped(long_preamble)
    assert preamble == [] and len(fragments) == 1


# breakdown prints this string in prose, not just in a counted table- see the anti-aliasing note
def test_the_500hz_method_string_is_the_one_the_prose_quotes():
    df = pd.DataFrame({"Time": np.arange(5000) * 2.0, "L_Deg_X": np.sin(np.arange(5000) / 50.0)})
    _, segs = resample_file(df, "Time")
    assert segs[0].method == "decimate_5x_fir"
    assert "`decimate_5x_fir`" in (REPO_ROOT / "stages" / "breakdown.py").read_text(encoding="utf-8")


# a marker nothing emits matches nothing, so the bucket it feeds silently reads as empty
def test_every_marker_is_still_emitted_by_some_stage():
    src = "\n".join(p.read_text(encoding="utf-8")
                    for p in (REPO_ROOT / "stages").rglob("*.py")
                    if p.name != "breakdown.py")
    assert all(m in src for m in RATE_REJECT_MARKERS)
