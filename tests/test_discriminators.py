# the discriminator registry is the seam that lets the physics view carry more than stand/walk
# (Section 13). two things must stay pinned: (1) the swap rule is registered VERBATIM -- its verdict
# read back through the registry is exactly swap_verdict(swap_count), so the refactor that added the
# registry changed no result; (2) the declarative extension path (ThresholdRule / from_spec) fires
# correctly and its validation gate rejects anything a proposed class must not smuggle in.

import pytest

from stages.s3_physics.anchors import (
    AMBIGUOUS, DISCRIMINATOR_ANCHORS, STANDING, SWAP_DISCRIMINATOR, WALKING,
    swap_count, swap_verdict,
)
from stages.s3_physics import discriminators as disc


# --- the swap rule as the first registered discriminator (the zero-change guarantee) ---

def test_swap_registered_as_code_callable():
    d = disc.get("swap")
    assert d is SWAP_DISCRIMINATOR
    assert d.kind == "callable" and d.origin == "code"
    assert d.emits == WALKING
    assert d.classes == (STANDING, AMBIGUOUS, WALKING)


@pytest.mark.parametrize("count,verdict", [
    (0, STANDING), (1, AMBIGUOUS), (2, WALKING), (5, WALKING)])
def test_swap_verdict_bands_unchanged_through_registry(count, verdict):
    # verdict_of read back through the registry must equal the bare swap_verdict -- this is the
    # property that makes the anchors.csv golden diff come out empty on the swap_verdict column
    assert SWAP_DISCRIMINATOR.verdict_of({"swap_count": count}) == verdict
    assert swap_verdict(count) == verdict


def test_registry_returns_a_copy():
    disc.registry()["swap"] = None  # mutating the returned dict must not touch the live registry
    assert disc.get("swap") is SWAP_DISCRIMINATOR


def test_duplicate_registration_is_loud():
    with pytest.raises(ValueError, match="already registered"):
        disc.register(SWAP_DISCRIMINATOR)


# --- the declarative extension path: a new class's physics test, as data ---

def test_threshold_rule_fires_on_all_clauses():
    # a bilateral squat sketch: legs move TOGETHER (antiphase < 0) and posture sweeps (grav_stab low)
    d = disc.from_spec("squat", "SQUAT",
                       [{"anchor": "antiphase", "op": "<", "value": 0.0},
                        {"anchor": "grav_stab", "op": "<", "value": 0.5}],
                       DISCRIMINATOR_ANCHORS)
    assert d.kind == "threshold" and d.origin == "proposed"  # not a trusted built-in
    assert d.verdict_of({"antiphase": -0.3, "grav_stab": 0.2}) == "SQUAT"
    assert d.verdict_of({"antiphase": -0.3, "grav_stab": 0.9}) == ""   # one clause fails -> abstain
    assert d.verdict_of({"antiphase": 0.4, "grav_stab": 0.2}) == ""    # other clause fails


def test_threshold_missing_anchor_does_not_fire():
    d = disc.from_spec("x", "X", [{"anchor": "periodicity", "op": ">", "value": 0.5}],
                       DISCRIMINATOR_ANCHORS)
    assert d.verdict_of({}) == ""  # absent anchor -> rule does not fire, never a KeyError


@pytest.mark.parametrize("spec,msg", [
    ([], "non-empty"),
    ("notalist", "non-empty"),
    ([{"anchor": "antiphase", "op": "<"}], "anchor/op/value"),
    ([{"anchor": "made_up", "op": "<", "value": 0}], "unknown anchor"),
    ([{"anchor": "antiphase", "op": "~=", "value": 0}], "unknown op"),
    ([{"anchor": "antiphase", "op": "<", "value": "NaN"}], "finite"),
    ([{"anchor": "antiphase", "op": "<", "value": "abc"}], "numeric"),
])
def test_validate_spec_rejects_bad_specs(spec, msg):
    with pytest.raises(ValueError, match=msg):
        disc.validate_spec(spec, DISCRIMINATOR_ANCHORS)


def test_known_vocabulary_is_the_audited_anchors_plus_descriptors():
    # the vocabulary a declarative rule may reference must include the four audited anchors
    for a in ("periodicity", "antiphase", "grav_stab", "gyro_energy"):
        assert a in DISCRIMINATOR_ANCHORS
