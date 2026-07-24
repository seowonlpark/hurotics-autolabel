# the S2 gates are the only path to champion, and run on agent-authored specs: validate_spec whitelists
# keys off the estimator, select_features refuses a silent no-op, decide() is the margin-based promotion
# rule (plus its steady_confusion tiebreaker). pure functions, so pinned exactly.

import pytest

from stages.s2_ml.experiment import (
    PROMOTION_MARGIN,
    STEADY_CONFUSION_MARGIN,
    ExperimentResult,
    ExperimentSpec,
    _spec_from_dict,
    decide,
    select_features,
    validate_spec,
)


def _spec(**kw):
    base = dict(name="cand", rationale="a stated mechanism")
    base.update(kw)
    return ExperimentSpec(**base)


def _result(macro_f1, steady=None):
    tax = None if steady is None else {"fractions": {"steady_confusion": steady}}
    return ExperimentResult(_spec(), macro_f1, 0.9, 0.9, {}, n_features=3,
                            n_train_windows=100, taxonomy=tax)


# validate_spec

def test_valid_spec_passes():
    validate_spec(_spec(model_params={"n_estimators": 300}, window_s=2.0))


def test_missing_name_or_rationale_rejected():
    with pytest.raises(ValueError):
        validate_spec(ExperimentSpec(name="", rationale="r"))
    with pytest.raises(ValueError):
        validate_spec(ExperimentSpec(name="n", rationale=""))


def test_unknown_model_param_rejected():
    with pytest.raises(ValueError):
        validate_spec(_spec(model_params={"learning_rate": 0.1}))


def test_out_of_bounds_numeric_param_rejected():
    with pytest.raises(ValueError):
        validate_spec(_spec(model_params={"n_estimators": 5}))  # below the (10, 2000) floor


def test_categorical_param_allowed():
    # max_features has bounds None -- categorical, not range-checked
    validate_spec(_spec(model_params={"max_features": "sqrt"}))


def test_window_out_of_range_rejected():
    with pytest.raises(ValueError):
        validate_spec(_spec(window_s=20.0))  # outside (0.5, 10.0)


# select_features

def test_select_features_drops_named():
    assert select_features(["a", "b", "c"], ["b"]) == ["a", "c"]


def test_unknown_drop_is_error_not_noop():
    with pytest.raises(ValueError):
        select_features(["a", "b"], ["typo"])


def test_dropping_everything_rejected():
    with pytest.raises(ValueError):
        select_features(["a", "b"], ["a", "b"])


# decide

def test_no_champion_establishes_baseline():
    promote, why = decide(_result(0.5), None)
    assert promote is True
    assert "baseline" in why


def test_clears_margin_promotes():
    champ = {"macro_f1": 0.80}
    promote, _ = decide(_result(0.80 + PROMOTION_MARGIN + 0.001), champ)
    assert promote is True


def test_below_margin_rejected():
    champ = {"macro_f1": 0.80}
    promote, why = decide(_result(0.802), champ)  # +0.002 < 0.005 margin
    assert promote is False
    assert "margin" in why


def test_worse_challenger_rejected():
    champ = {"macro_f1": 0.80}
    promote, _ = decide(_result(0.70), champ)
    assert promote is False


def test_tie_broken_by_steady_confusion_drop():
    # a macro-F1 tie, but steady_confusion falls past its margin -> prefer the fail-passive model
    champ = {"macro_f1": 0.80, "taxonomy": {"fractions": {"steady_confusion": 0.20}}}
    challenger = _result(0.801, steady=0.20 - STEADY_CONFUSION_MARGIN - 0.01)
    promote, why = decide(challenger, champ)
    assert promote is True
    assert "steady_confusion" in why


def test_tie_without_enough_steady_drop_rejected():
    champ = {"macro_f1": 0.80, "taxonomy": {"fractions": {"steady_confusion": 0.20}}}
    challenger = _result(0.801, steady=0.195)  # only 0.005 drop, below the 0.02 margin
    promote, _ = decide(challenger, champ)
    assert promote is False


def test_tie_without_taxonomy_cannot_use_tiebreaker():
    champ = {"macro_f1": 0.80}  # no taxonomy on either side
    promote, _ = decide(_result(0.801, steady=None), champ)
    assert promote is False


# _spec_from_dict

def test_spec_from_dict_drops_unknown_keys():
    spec = _spec_from_dict({"name": "x", "rationale": "y", "legacy_field": 1})
    assert spec.name == "x" and spec.rationale == "y"
