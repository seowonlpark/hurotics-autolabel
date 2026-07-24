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
#
# the label rule above is not hand-wired to two classes: it is exactly "the winning class of each
# (S2 label, S3 verdict) cell", read off the measured majority-true of that cell. FusionPolicy /
# derive_policy make that reading explicit and N-class ready (Section 13), and derive_policy on the
# current corpus reproduces the frozen 2-class default byte-for-byte (locked by test_fusion_policy).
# the tier is structural (tier_of) and needs no fitting, so a new class earns a calibrated confidence
# the moment its S3 verdict can name it.

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable

from stages.s2_ml.dataset import STAND, WALK
from stages.s3_physics.anchors import STANDING, WALKING

# confidence tiers; a controller acts on HIGH/MED and fails passive on LOW
HIGH, MED, LOW = "high", "medium", "low"

# an S3 verdict maps to the class it ASSERTS; AMBIGUOUS asserts none (absent here), so it is never
# agreement nor disagreement -- it is the "no second opinion" tier. widen this map when a new class
# gets its own physics verdict (Section 13), and the tier logic below extends with no other change.
S3_TO_CLASS: dict[str, int] = {STANDING: STAND, WALKING: WALK}

# a cell thinner than this is not trusted to set its own label from a majority -- fail-passive keeps
# S2 ([S2-5]). every validated 2-class cell has >= 71 windows (Section 12), so this bound only guards a
# thin NEW-class cell from inventing a call out of a handful of noisy windows; it does not touch the
# stand/walk policy, whose cells all clear it by a wide margin.
MIN_CELL_SUPPORT = 20


@dataclass(frozen=True)
class Fusion:
    label: int # STAND / WALK -- the fused call (still emitted on LOW, but flagged)
    confidence: str # HIGH / MED / LOW, from S2-vs-S3 agreement
    abstain: bool # True on LOW: the label is ~coin-flip, a safety-critical controller holds
    s2_proba: float # S2's own class probability, carried as a graded within-tier score


# the confidence tier, STRUCTURAL -- read off S2/S3 AGREEMENT, never either model's own probability
# ([S4-2], Section 12): the two naming the SAME class -> HIGH; S3 abstaining (AMBIGUOUS) -> MED (no
# second opinion, keep S2); the two naming DIFFERENT classes -> LOW (a controller holds). this needs
# no fitting and generalizes to any class set unchanged -- it is why a new class gets a calibrated
# tier for free, the moment its S3 verdict can name it. reproduces the frozen 2-class tiering exactly.
def tier_of(s2_pred: int, s3_verdict: str) -> str:
    s3_class = S3_TO_CLASS.get(s3_verdict)
    if s3_class is not None and s3_class == s2_pred:
        return HIGH
    if s3_class is None:
        return MED
    return LOW


# the fused-label policy: the winning class of each (S2 label, S3 verdict) cell, READ OFF the measured
# majority-true of that cell (Section 12), not assumed. this is the NxN generalization of the hand 2x2
# -- stand/walk is one instance. a cell without a trusted majority (too thin, or a tie) is simply
# absent, and label_for falls back to S2 (fail-passive). FROZEN by contract: derive it ONCE from the
# deployment-fit population (train+val) and apply it unchanged; NEVER re-derive it on the lockbox,
# which would fit the held-out set the one honest number depends on (Section 7, Section 12.5).
@dataclass(frozen=True)
class FusionPolicy:
    label_of: dict                                # (s2_pred, s3_verdict) -> fused label
    support: dict = field(default_factory=dict)   # cell -> n windows it was measured on (provenance)

    # the fused label for one cell; fail-passive to S2 where the policy has no trusted majority
    def label_for(self, s2_pred: int, s3_verdict: str) -> int:
        return self.label_of.get((s2_pred, s3_verdict), s2_pred)


# build a FusionPolicy from measured (s2_pred, s3_verdict, true) rows: each cell's fused label is its
# STRICT majority true class, kept only with >= MIN_CELL_SUPPORT windows and no tie -- otherwise the
# cell is left out so label_for fails passive to S2. this is the literal "read off the measured
# agreement" the 2-class policy was written from by hand ([S4-2]); on the current corpus it reproduces
# that hand policy exactly. derive from train+val, then FREEZE (see FusionPolicy) -- do not feed it
# lockbox rows.
def derive_policy(rows: Iterable[tuple[int, str, int]]) -> FusionPolicy:
    counts: dict[tuple[int, str], Counter] = defaultdict(Counter)
    for s2, s3, true in rows:
        counts[(int(s2), s3)][int(true)] += 1
    label_of: dict[tuple[int, str], int] = {}
    support: dict[tuple[int, str], int] = {}
    for cell, c in counts.items():
        n = sum(c.values())
        support[cell] = n
        top, top_n = c.most_common(1)[0]
        tied = sum(1 for v in c.values() if v == top_n) > 1
        if n >= MIN_CELL_SUPPORT and not tied: # trust a clear, well-supported majority only
            label_of[cell] = int(top)
    return FusionPolicy(label_of=label_of, support=support)


# fuse one window into a call + confidence. label: from a measured FusionPolicy when one is given,
# else the FROZEN 2-class default -- STAND only when S2 says STAND AND physics does not veto toward
# WALKING; every other cell WALK (Section 12). the default equals derive_policy() on the current corpus
# (locked by test_fusion_policy), so passing no policy is byte-identical to the shipped 2-class fuser.
# tier + abstain are structural (tier_of) and identical either way.
def fuse(s2_pred: int, s2_proba: float, s3_verdict: str,
         policy: FusionPolicy | None = None) -> Fusion:
    if policy is None:
        label = STAND if (s2_pred == STAND and s3_verdict != WALKING) else WALK
    else:
        label = policy.label_for(s2_pred, s3_verdict)

    confidence = tier_of(s2_pred, s3_verdict)
    return Fusion(label, confidence, confidence == LOW, float(s2_proba))
