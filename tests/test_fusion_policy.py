# the fuser is the deployable, and its policy is the measured S2xS3 agreement (Section 12), now made
# explicit and N-class ready as FusionPolicy / derive_policy (Section 13). the load-bearing property:
# deriving the policy from a contingency shaped like the real corpus reproduces the FROZEN 2-class
# default cell-for-cell -- so routing production through derive_policy changed no result. these tests
# lock that equivalence, plus the fail-passive guards that keep a thin new-class cell from inventing a
# call. test_fuse.py still pins the frozen default itself; this pins the generalization around it.

import pytest

from stages.s2_ml.dataset import STAND, WALK
from stages.s3_physics.anchors import AMBIGUOUS, STANDING, WALKING
from stages.s4_fusion.fuse import (
    HIGH, LOW, MED, MIN_CELL_SUPPORT, derive_policy, fuse, tier_of)

# the six S2xS3 cells and their measured majority-true class (Section 12 / the live contingency): this
# is the reading the hand 2-class default encodes. n well above MIN_CELL_SUPPORT so every cell is
# trusted, mirroring the real corpus (smallest real cell is 71 windows).
CELL_MAJORITY = {
    (STAND, STANDING): STAND,
    (STAND, AMBIGUOUS): STAND,
    (STAND, WALKING): WALK,   # physics vetoes toward WALK: 78% of this cell is truly WALK
    (WALK, STANDING): WALK,   # S3 STANDING ignored against S2: 79% truly WALK
    (WALK, AMBIGUOUS): WALK,
    (WALK, WALKING): WALK,
}
ALL_S3 = (STANDING, AMBIGUOUS, WALKING)


# build measured rows for one cell: `n` windows, a strict `majority` supermajority, the rest the other
# class -- enough signal that derive_policy trusts the cell
def _cell_rows(s2, s3, majority, n=100):
    other = WALK if majority == STAND else STAND
    return [(s2, s3, majority)] * (n - 5) + [(s2, s3, other)] * 5


def _corpus_rows():
    rows = []
    for (s2, s3), maj in CELL_MAJORITY.items():
        rows += _cell_rows(s2, s3, maj)
    return rows


# --- the equivalence that makes the production diff empty ---

def test_derived_policy_matches_frozen_default_on_every_cell():
    policy = derive_policy(_corpus_rows())
    for s2 in (STAND, WALK):
        for s3 in ALL_S3:
            derived = fuse(s2, 0.9, s3, policy)
            frozen = fuse(s2, 0.9, s3)  # policy=None -> the shipped 2-class rule
            assert derived.label == frozen.label, (s2, s3)
            assert derived.confidence == frozen.confidence == tier_of(s2, s3)
            assert derived.abstain == frozen.abstain


def test_derived_cell_labels_are_the_measured_majority():
    policy = derive_policy(_corpus_rows())
    for cell, maj in CELL_MAJORITY.items():
        assert policy.label_for(*cell) == maj
        assert policy.support[cell] == 100


# --- structural tier: reproduces the frozen tiering, generalizes with no fitting ---

@pytest.mark.parametrize("s2,s3,tier", [
    (STAND, STANDING, HIGH), (WALK, WALKING, HIGH),   # same class named -> agreement
    (STAND, AMBIGUOUS, MED), (WALK, AMBIGUOUS, MED),  # S3 abstains -> no second opinion
    (STAND, WALKING, LOW), (WALK, STANDING, LOW),     # different classes -> disagreement
])
def test_tier_is_structural(s2, s3, tier):
    assert tier_of(s2, s3) == tier
    assert fuse(s2, 0.9, s3).confidence == tier


# --- fail-passive guards: a thin or tied cell must not set a label, it keeps S2 ---

def test_thin_cell_is_not_trusted_and_keeps_s2():
    thin = [(STAND, WALKING, WALK)] * (MIN_CELL_SUPPORT - 1)  # a real veto cell, but too few windows
    policy = derive_policy(thin)
    assert (STAND, WALKING) not in policy.label_of      # not trusted
    assert policy.label_for(STAND, WALKING) == STAND    # fail-passive to S2


def test_tied_cell_is_not_trusted():
    tied = [(STAND, WALKING, STAND)] * 50 + [(STAND, WALKING, WALK)] * 50  # 50/50, no majority
    policy = derive_policy(tied)
    assert (STAND, WALKING) not in policy.label_of
    assert policy.label_for(STAND, WALKING) == STAND


def test_unseen_cell_fails_passive():
    # a cell the policy never measured (e.g. a class combination absent from train+val) keeps S2
    policy = derive_policy(_cell_rows(WALK, WALKING, WALK))
    assert policy.label_for(STAND, STANDING) == STAND
