# S4 fusion: one call + one confidence from the S2 learned label and the S3 physics verdict.
# this is the deployable artifact -- and the confidence signal the incumbent lacks (the reason
# the project exists). deterministic, no data access: the whole policy is reviewable here.
#
# every rule is measured, not assumed (DOMAIN_NOTES Section 12, the S2xS3 contingency):
#   - AGREEMENT is 98% correct -> HIGH; DISAGREEMENT is ~coin-flip for either model's own
#     label -> LOW (abstain). agreement beats S2's own probability at matched coverage, so S3
#     carries independent information, not a rehash of S2's confidence.
#   - S3's WALKING verdict is a reliable WALK signal (78-99% across cells); its STANDING verdict
#     is NOT (contaminated by slow gait it under-calls). so physics VETOES toward WALK, and its
#     STANDING call is ignored against S2 -- the reverse of the intuitive "trust physics on
#     standing", which measured *below* S2 alone.
#   - "take whichever model is more confident" was measured and REJECTED: S2 is confident-wrong
#     exactly where the two disagree (proba 0.97, accuracy 0.90 when S3 says STANDING), so the
#     tier is set by agreement, never by whose confidence is louder.

from __future__ import annotations

from dataclasses import dataclass

from stages.s2_ml.dataset import STAND, WALK
from stages.s3_physics.anchors import AMBIGUOUS, STANDING, WALKING

# confidence tiers; a controller acts on HIGH/MED and fails passive on LOW
HIGH, MED, LOW = "high", "medium", "low"


@dataclass(frozen=True)
class Fusion:
    label: int # STAND / WALK -- the fused call (still emitted on LOW, but flagged)
    confidence: str # HIGH / MED / LOW, from S2-vs-S3 agreement
    abstain: bool # True on LOW: the label is ~coin-flip, a safety-critical controller holds
    s2_proba: float # S2's own class probability, carried as a graded within-tier score


# fuse one window. label = the measured per-cell majority: STAND only when S2 says STAND AND
# physics does not call WALKING; every other cell is WALK (both disagreement cells run ~79%
# WALK, so S3=WALKING vetoes S2=STAND, and S3=STANDING is ignored against S2=WALK).
def fuse(s2_pred: int, s2_proba: float, s3_verdict: str) -> Fusion:
    label = STAND if (s2_pred == STAND and s3_verdict != WALKING) else WALK

    agree = ((s2_pred == STAND and s3_verdict == STANDING)
             or (s2_pred == WALK and s3_verdict == WALKING))
    if agree:
        confidence = HIGH
    elif s3_verdict == AMBIGUOUS:
        confidence = MED
    else: # one says stand, the other walk -- the coin-flip cell
        confidence = LOW

    return Fusion(label, confidence, confidence == LOW, float(s2_proba))
