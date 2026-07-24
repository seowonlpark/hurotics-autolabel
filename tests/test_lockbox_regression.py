# Regression LOCK on the spent lockbox (rev8, rev13) -- NOT a generalization test.
#
# rev8/rev13 were opened once and are spent (DOMAIN_NOTES 12.5): they can never again produce an
# unbiased number, and the model must never be tuned against them. What is still legitimate is
# pinning the ALREADY-RECORDED 12.5 result to the code, so a later edit to the deterministic fusion
# or scoring layer cannot quietly change the number the record reports.
#
# The fixture (tests/fixtures/lockbox_windows.csv, made by make_lockbox_fixture.py) freezes the real
# per-window fusion inputs -- the deployment-fit S2 label+proba and the model-free S3 verdict -- so
# these tests re-apply the fuser and the scorer to real distributions WITHOUT refitting the model.
# They assert the exact numbers in DOMAIN_NOTES 12.5 and runs/s4_fusion/lockbox_result.md. If a
# change is meant to move these, regenerate the fixture and update the constants here deliberately.

from pathlib import Path

import pandas as pd
import pytest

from stages.s2_ml.dataset import DEFAULT_LOCKBOX_REVS, STAND, WALK
from stages.s3_physics.anchors import WALKING
from stages.s4_fusion.fuse import HIGH, LOW, MED
from stages.s4_fusion.lockbox import (
    HEADLINE, LOCKBOX, OPENED_REVS, REFERENCE, SEALED_REVS, SPENT_REVS)
from stages.s4_fusion.run import apply_fusion, calibration, score

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "lockbox_windows.csv"


@pytest.fixture(scope="module")
def fused():
    df = pd.read_csv(FIXTURE)
    return apply_fusion(df)


def _block(g):
    true = g["true"].to_numpy(int)
    keep = ~g["abstain"].to_numpy(bool)
    return {
        "s2": score(true, g["s2_pred"].to_numpy(int)),
        "fused": score(true, g["fused_label"].to_numpy(int)),
        "acting": score(true, g["fused_label"].to_numpy(int), keep=keep),
        "calibration": calibration(g),
    }


def test_lockbox_registry_is_consistent():
    # lockbox rev status is now ONE authored registry (lockbox.LOCKBOX); membership is
    # dataset.DEFAULT_LOCKBOX_REVS. the SEALED/SPENT/OPENED tuples are derived from LOCKBOX and the
    # module raises at import if the registry and the dataset membership disagree -- so importing at
    # all already proves that tie. this pins the remaining intent: valid roles and the partition.
    assert all(r.role in (HEADLINE, REFERENCE) for r in LOCKBOX)
    lockbox = set(DEFAULT_LOCKBOX_REVS)
    sealed, spent, opened = set(SEALED_REVS), set(SPENT_REVS), set(OPENED_REVS)
    # every lockbox rev is either the headline or reference-only, never both
    assert sealed | spent == lockbox
    assert sealed.isdisjoint(spent)
    # a rev spent on the axis decision has by definition been seen; you only open lockbox revs
    assert spent <= opened <= lockbox


def test_fixture_shape(fused):
    # the sealed/spent revs and their window counts are part of the recorded result
    assert dict(fused.groupby("rev").size()) == {"rev8": 198, "rev13": 1235}


def test_rev8_headline_matches_record(fused):
    # DOMAIN_NOTES 12.5 headline (rev8, never seen): the honest one-shot number, now locked
    b = _block(fused[fused["rev"] == "rev8"])
    assert b["s2"]["macro_f1"] == 0.8217
    assert b["fused"]["macro_f1"] == 0.7978
    assert b["fused"]["accuracy"] == 0.8636
    assert b["fused"]["stand_recall"] == 0.5577
    assert b["fused"]["walk_recall"] == 0.9726
    # the confidence-gated deployable: abstain on LOW
    assert b["acting"]["coverage"] == 0.9444
    assert b["acting"]["accuracy"] == 0.8984


def test_rev8_tier_calibration_matches_record(fused):
    cal = _block(fused[fused["rev"] == "rev8"])["calibration"]
    assert cal[HIGH]["n"] == 175
    assert cal[MED]["n"] == 12
    assert cal[LOW]["n"] == 11
    # HIGH is well-calibrated, LOW is near-chance -- the signal 12.5 says generalized
    assert cal[HIGH]["accuracy"] == 0.92
    assert cal[LOW]["accuracy"] == 0.2727


def test_rev13_reference_matches_record(fused):
    # rev13 is spent/reference-only; on it fusion is a no-op (S2 already calls every miss WALK)
    b = _block(fused[fused["rev"] == "rev13"])
    assert b["s2"]["macro_f1"] == 0.764
    assert b["fused"]["macro_f1"] == 0.764
    assert b["fused"]["walk_recall"] == 1.0
    assert b["fused"]["stand_recall"] == 0.3846


def test_abstain_gate_catches_more_errors_than_it_acts_on(fused):
    # the deployable's core claim (12.5 finding 2): LOW/abstain windows are far more error-prone
    # than the windows it acts on. A property of the real held-out data, locked as a guard.
    err = fused["fused_label"] != fused["true"]
    abstain = fused["abstain"].to_numpy(bool)
    assert err[abstain].mean() > 3 * err[~abstain].mean()


def test_physics_veto_direction_on_real_windows(fused):
    # the measured policy (fuse.py): where S2 says STAND but physics says WALKING, the fused call
    # is WALK and the window abstains (LOW). Lock it against a real slice so an edit can't invert
    # the veto and still pass the synthetic unit tests.
    slc = fused[(fused["s2_pred"] == STAND) & (fused["s3"] == WALKING)]
    assert len(slc) > 0
    assert (slc["fused_label"] == WALK).all()
    assert (slc["confidence"] == LOW).all()
    assert slc["abstain"].all()
