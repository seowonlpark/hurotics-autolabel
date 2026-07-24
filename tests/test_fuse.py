# the fuser: one call + one confidence from S2 (learned) and S3 (physics). its policy is measured, not
# intuitive: physics VETOES toward WALK but its STANDING call is ignored against S2, and any disagreement
# abstains (LOW). these cases lock that table so a future edit can't turn the veto around.

from stages.s2_ml.dataset import STAND, WALK
from stages.s3_physics.anchors import AMBIGUOUS, STANDING, WALKING
from stages.s4_fusion.fuse import HIGH, LOW, MED, fuse


def test_agreement_stand_is_high():
    f = fuse(STAND, 0.9, STANDING)
    assert f.label == STAND and f.confidence == HIGH and f.abstain is False


def test_agreement_walk_is_high():
    f = fuse(WALK, 0.9, WALKING)
    assert f.label == WALK and f.confidence == HIGH and f.abstain is False


def test_physics_walking_vetoes_s2_stand():
    # S2 says STAND, physics says WALKING -> label flips to WALK, and it is a LOW disagreement
    f = fuse(STAND, 0.97, WALKING)
    assert f.label == WALK and f.confidence == LOW and f.abstain is True


def test_physics_standing_ignored_against_s2_walk():
    # S2 says WALK, physics says STANDING -> label stays WALK (S3 standing is not trusted),
    # but the two disagree, so it still abstains
    f = fuse(WALK, 0.9, STANDING)
    assert f.label == WALK and f.confidence == LOW and f.abstain is True


def test_ambiguous_physics_is_medium():
    assert fuse(STAND, 0.9, AMBIGUOUS).confidence == MED
    assert fuse(WALK, 0.9, AMBIGUOUS).confidence == MED


def test_s2_proba_carried_through():
    assert fuse(WALK, 0.83, WALKING).s2_proba == 0.83
