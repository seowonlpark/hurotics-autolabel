# the S3/S4 provenance gates keep an agent's claim out of the record unless it points at a real window.
# a finding with no window/statement or an inverted time range is flagged ok=False; a hypothesis leaning
# on a rate_dependent anchor without saying so is warned. these tests exercise both paths.

from agents.s3_physics import _evidence_strength, _valid_evidence, validate_hypothesis
from agents.s4_fusion import measure_finding, validate_finding
from stages.s2_ml.dataset import STAND, WALK

VERDICTS = {"periodicity": "invariant", "antiphase": "rate_dependent"}


def _evidence(**kw):
    ev = dict(rev="rev2", trial=1, t_start_s=10.0, t_end_s=20.0, anchors=["antiphase"])
    ev.update(kw)
    return ev


# s3 evidence + hypothesis

def test_valid_evidence_accepts_complete_item():
    assert _valid_evidence(_evidence()) is True


def test_evidence_missing_window_rejected():
    assert _valid_evidence(_evidence(t_end_s=None)) is False


def test_evidence_inverted_window_rejected():
    assert _valid_evidence(_evidence(t_start_s=20.0, t_end_s=10.0)) is False


def test_evidence_without_anchors_rejected():
    assert _valid_evidence(_evidence(anchors=[])) is False


def _hyp(**kw):
    h = dict(statement="a claim", confidence="high", evidence=[_evidence()],
             relies_on=["periodicity"], label_audit=False, caveat="what falsifies it")
    h.update(kw)
    return h


def test_clean_hypothesis_passes_without_warnings():
    v = validate_hypothesis(_hyp(), VERDICTS)
    assert v["validation"]["ok"] is True
    assert v["validation"]["warnings"] == []
    assert v["rate_invariance"] == {"periodicity": "invariant"}


def test_hypothesis_without_statement_flagged():
    v = validate_hypothesis(_hyp(statement=""), VERDICTS)
    assert v["validation"]["ok"] is False


def test_hypothesis_bad_evidence_flagged():
    v = validate_hypothesis(_hyp(evidence=[{"rev": "rev2"}]), VERDICTS)
    assert v["validation"]["ok"] is False


def test_leaning_on_rate_dependent_anchor_warns():
    # relies on antiphase (rate_dependent) without acknowledging the verdict -> kept but warned
    v = validate_hypothesis(_hyp(relies_on=["antiphase"]), VERDICTS)
    assert v["validation"]["ok"] is True
    assert v["validation"]["warnings"]


def test_acknowledging_rate_dependence_clears_warning():
    v = validate_hypothesis(
        _hyp(relies_on=["antiphase"], caveat="this may track the sampling grid, not the body"),
        VERDICTS)
    assert v["validation"]["warnings"] == []


# s4 fusion finding

def _finding(**kw):
    f = dict(statement="a claim", classification="label_problem", confidence="high",
             rev="rev2", trial=1, t_start_s=10.0, t_end_s=20.0, caveat="x")
    f.update(kw)
    return f


def test_valid_finding_passes():
    assert validate_finding(_finding())["validation"]["ok"] is True


def test_finding_bad_classification_flagged():
    assert validate_finding(_finding(classification="made_up"))["validation"]["ok"] is False


def test_finding_inverted_window_flagged():
    v = validate_finding(_finding(t_start_s=20.0, t_end_s=10.0))
    assert v["validation"]["ok"] is False


def test_finding_missing_time_flagged():
    v = validate_finding(_finding(t_end_s=None))
    assert v["validation"]["ok"] is False


# s3 evidence strength (confidence grounded in code)
# the agent's self-reported confidence must not stand unchecked: code measures the evidence behind
# it -- windows and, crucially, how many revs they span -- and flags a 'high' resting on one window.

def test_evidence_strength_single_window_is_weak():
    s = _evidence_strength(_hyp(evidence=[_evidence()], confidence="high"))
    assert s["tier"] == "weak" and s["n_windows"] == 1 and s["n_revs"] == 1
    assert s["exceeds_evidence"] is True  # 'high' on one window outruns its evidence


def test_evidence_strength_recurring_across_revs_is_strong():
    ev = [_evidence(rev="rev2"), _evidence(rev="rev3"), _evidence(rev="rev6")]
    s = _evidence_strength(_hyp(evidence=ev, confidence="high"))
    assert s["tier"] == "strong" and s["n_revs"] == 3 and s["exceeds_evidence"] is False


def test_evidence_strength_two_windows_one_rev_is_moderate():
    s = _evidence_strength(_hyp(evidence=[_evidence(), _evidence(t_start_s=30, t_end_s=40)]))
    assert s["tier"] == "moderate"


def test_validate_hypothesis_attaches_evidence_strength():
    v = validate_hypothesis(_hyp(), VERDICTS)
    assert "evidence_strength" in v and v["evidence_strength"]["tier"] == "weak"


# s4 finding measured against ground truth
# a finding's fusion_verdict is checked against the labels on the windows it cites, and a finding
# that lands on no scored window is dropped. rows carry the fused table's per-window facts.

def _rows(*specs):
    # each spec: (t_start_s, s2_pred, s3_class, true, fused_label, is_low)
    return [dict(t_start_s=t, s2_pred=s2, s3_class=s3, true=tr, fused_label=fl, is_low=low)
            for (t, s2, s3, tr, fl, low) in specs]


def test_measure_s2_should_win_corroborated_when_s2_beats_fused():
    # over [10,20): s2 right on both, fused wrong on both -> taking s2 would have helped
    rows = _rows((10, STAND, WALK, STAND, WALK, True), (12, STAND, WALK, STAND, WALK, True))
    m = measure_finding(_finding(t_start_s=10, t_end_s=20, fusion_verdict="s2_should_win"), rows)
    assert m["n_matched"] == 2 and m["s2_accuracy"] == 1.0 and m["fused_accuracy"] == 0.0
    assert m["verdict_check"] == "corroborated"


def test_measure_s2_should_win_contradicted_when_fused_beats_s2():
    rows = _rows((10, WALK, WALK, STAND, STAND, False))
    m = measure_finding(_finding(t_start_s=10, t_end_s=20, fusion_verdict="s2_should_win"), rows)
    assert m["verdict_check"] == "contradicted"


def test_measure_s3_should_win_unverifiable_when_no_s3_class():
    # every window is s3-ambiguous (s3_class None) -> s3 cannot be scored -> unverifiable
    rows = _rows((10, STAND, None, STAND, STAND, True))
    m = measure_finding(_finding(t_start_s=10, t_end_s=20, fusion_verdict="s3_should_win"), rows)
    assert m["s3_accuracy"] is None and m["verdict_check"] == "unverifiable"


def test_measure_abstain_correct_when_no_alternative_beats_fused():
    # s2 and (classifiable) s3 each score no better than the fused label -> abstaining lost nothing
    rows = _rows((10, WALK, WALK, STAND, STAND, True), (12, STAND, STAND, WALK, WALK, True))
    m = measure_finding(_finding(t_start_s=10, t_end_s=20, fusion_verdict="abstain_correct"), rows)
    assert m["verdict_check"] == "corroborated"


def test_measure_no_matched_window_is_unverifiable():
    rows = _rows((100, STAND, WALK, STAND, WALK, True))  # far outside the cited window
    m = measure_finding(_finding(t_start_s=10, t_end_s=20, fusion_verdict="s2_should_win"), rows)
    assert m["n_matched"] == 0 and m["verdict_check"] == "unverifiable"


def test_validate_finding_with_rows_fails_when_no_scored_window():
    v = validate_finding(_finding(t_start_s=10, t_end_s=20), rows=[])
    assert v["validation"]["ok"] is False
    assert any("no scored window" in r for r in v["validation"]["reasons"])


def test_validate_finding_flags_high_confidence_contradicted_by_truth():
    rows = _rows((10, WALK, WALK, STAND, STAND, False))  # s2 wrong, fused right
    v = validate_finding(
        _finding(t_start_s=10, t_end_s=20, confidence="high", fusion_verdict="s2_should_win"),
        rows=rows)
    assert v["validation"]["ok"] is True  # format + grounding fine; the claim is just flagged
    assert v["confidence_flags"] == ["high confidence contradicted by ground truth"]
    assert v["measurement"]["verdict_check"] == "contradicted"


def test_validate_finding_clean_grounded_passes_without_flags():
    rows = _rows((10, STAND, WALK, STAND, WALK, True))  # s2 beats fused, verdict holds
    v = validate_finding(
        _finding(t_start_s=10, t_end_s=20, confidence="high", fusion_verdict="s2_should_win"),
        rows=rows)
    assert v["validation"]["ok"] is True and v["confidence_flags"] == []
