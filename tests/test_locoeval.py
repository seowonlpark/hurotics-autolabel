# locoeval is the blind measure layer -- the headline macro-F1 the whole champion/challenger loop
# gates on. these tests pin the metric arithmetic against a hand-computed example, and check the
# transition report's flicker/boundary-offset behaviour on constructed switches.

import numpy as np
import pandas as pd
import pytest

from stages.s2_ml.dataset import STAND, WALK
from stages.s2_ml.locoeval import _prf, evaluate, transition_report


def test_prf_empty_denominators():
    assert _prf(0, 0, 0) == (0.0, 0.0, 0.0)


def test_prf_perfect():
    assert _prf(5, 0, 0) == (1.0, 1.0, 1.0)


def test_evaluate_hand_computed():
    y_true = np.array([STAND, STAND, WALK, WALK])
    y_pred = np.array([STAND, WALK, WALK, WALK])
    r = evaluate(y_true, y_pred)

    assert r.n == 4
    assert r.accuracy == pytest.approx(0.75)
    # stand: p=1.0 r=0.5 f=0.6667 ; walk: p=0.6667 r=1.0 f=0.8
    by_label = {c.label: c for c in r.per_class}
    assert by_label["stand"].precision == pytest.approx(1.0)
    assert by_label["stand"].recall == pytest.approx(0.5)
    assert by_label["walk"].recall == pytest.approx(1.0)
    assert r.macro_f1 == pytest.approx((0.6666667 + 0.8) / 2, abs=1e-4)
    assert r.balanced_accuracy == pytest.approx(0.75)
    # confusion rows = truth
    assert r.confusion["stand"] == {"stand": 1, "walk": 1}
    assert r.confusion["walk"] == {"stand": 0, "walk": 2}


def test_evaluate_empty_is_safe():
    r = evaluate(np.array([]), np.array([]))
    assert r.n == 0
    assert r.accuracy == 0.0


def test_evaluate_per_rev_breakdown():
    # each rev carries BOTH classes, so a perfect prediction scores a true macro-F1 of 1.0
    y_true = np.array([STAND, WALK, STAND, WALK])
    y_pred = np.array([STAND, WALK, STAND, WALK])
    groups = np.array(["revA", "revA", "revB", "revB"])
    r = evaluate(y_true, y_pred, groups=groups)
    assert set(r.per_rev_macro_f1) == {"revA", "revB"}
    assert r.per_rev_macro_f1["revA"] == pytest.approx(1.0)


def test_per_rev_single_class_is_capped_at_half():
    # KNOWN QUIRK: macro_f1 always averages over {stand, walk}, so a rev (or subset) that
    # contains only one class is capped at 0.5 even when every prediction is correct -- the
    # absent class contributes F1=0. per_rev_macro_f1 is a diagnostic, not the promotion gate,
    # but a single-class rev's per-rev score should be read with this in mind.
    y = np.array([STAND, STAND])
    r = evaluate(y, y.copy(), groups=np.array(["revA", "revA"]))
    assert r.per_rev_macro_f1["revA"] == pytest.approx(0.5)


def _tr_frame(labels):
    n = len(labels)
    return pd.DataFrame({
        "rev": ["rev1"] * n, "trial": [1] * n, "segment": [0] * n,
        "t_start_ms": [100.0 * i for i in range(n)], "label": labels,
    })


def test_transition_report_late_boundary():
    # truth switches STAND->WALK at index 3; pred makes the same switch one window late
    truth = [STAND, STAND, STAND, WALK, WALK, WALK]
    pred = np.array([STAND, STAND, STAND, STAND, WALK, WALK])
    rep = transition_report(_tr_frame(truth), pred)
    assert rep["true_transitions"] == 1
    b = rep["boundary_error_windows"]
    assert b["late"] == 1 and b["early"] == 0 and b["on_time"] == 0
    assert b["median"] == pytest.approx(1.0)
    # the late transition is not counted as flicker
    assert rep["flicker_switches"] == 0


def test_transition_report_flicker():
    # truth is constant WALK; a single-window blip to STAND is a spurious switch
    truth = [WALK] * 6
    pred = np.array([WALK, WALK, STAND, WALK, WALK, WALK])
    rep = transition_report(_tr_frame(truth), pred)
    assert rep["true_transitions"] == 0
    assert rep["flicker_switches"] == 2  # into the blip and back out
    assert rep["flicker_rate"] == pytest.approx(2 / 5)
