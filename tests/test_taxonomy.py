# the error taxonomy is the comparability yardstick (a faithful port of the incumbent's bucketing)
# and the source of decide()'s steady_confusion tiebreaker. its precedence assignment must
# partition every row into exactly one bucket. we assert that invariant plus a couple of
# unambiguous cases, rather than hand-encoding counts that could bake a bug in as "expected".

import numpy as np

from stages.s2_ml.dataset import STAND, WALK
from stages.s2_ml.taxonomy import CORRECT, ERROR_BUCKETS, aggregate, bucket_errors, row_buckets

_MS = 100.0  # 10 Hz spacing, well above the flicker/lag ms thresholds


def _t(n):
    return np.arange(n, dtype=float) * _MS


def test_every_row_lands_in_exactly_one_bucket():
    gt = np.array([STAND, STAND, WALK, WALK, WALK, STAND, STAND, WALK])
    pred = np.array([STAND, WALK, WALK, STAND, WALK, STAND, WALK, WALK])
    bucket = row_buckets(gt, pred, _t(len(gt)))
    # nothing is left unclassified: steady_confusion absorbs the remainder
    assert "unclassified" not in set(bucket)
    known = {CORRECT, *ERROR_BUCKETS}
    assert all(b in known for b in bucket)


def test_correct_plus_error_equals_n():
    gt = np.array([STAND, STAND, WALK, WALK, WALK, STAND])
    pred = np.array([STAND, WALK, WALK, STAND, WALK, STAND])
    res = bucket_errors(gt, pred, _t(len(gt)))
    assert res["correct_rows"] + res["total_error_rows"] == len(gt)


def test_all_correct_has_no_errors():
    gt = np.array([STAND, STAND, WALK, WALK])
    res = bucket_errors(gt, gt.copy(), _t(len(gt)))
    assert res["total_error_rows"] == 0
    assert res["correct_rows"] == 4
    assert res["dominant"] is None
    assert all(v == 0.0 for v in res["fractions"].values())


def test_empty_input_is_safe():
    res = bucket_errors(np.array([]), np.array([]), np.array([]))
    assert res["total_error_rows"] == 0 and res["dominant"] is None


def test_aggregate_sums_across_segments():
    gt = np.array([STAND, WALK, WALK, STAND])
    a = bucket_errors(gt, np.array([WALK, WALK, WALK, STAND]), _t(4))
    b = bucket_errors(gt, np.array([STAND, STAND, WALK, STAND]), _t(4))
    agg = aggregate([a, b])
    for bucket in ERROR_BUCKETS:
        assert agg["counts"][bucket] == a["counts"][bucket] + b["counts"][bucket]
    assert agg["correct_rows"] == a["correct_rows"] + b["correct_rows"]
    total = agg["correct_rows"] + agg["total_error_rows"]
    assert agg["row_accuracy"] == round(agg["correct_rows"] / total, 4)
