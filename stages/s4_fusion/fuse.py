# S4 fusion: one call + one confidence from the S2 label and S3 verdict; the deployable artifact.
# deterministic, no data access. every rule is measured (Section 12): agreement -> HIGH, S3 abstaining ->
# MED, disagreement -> LOW; physics vetoes toward WALK only. N-class ready via FusionPolicy (Section 13).

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable

from stages.s2_ml.dataset import STAND, WALK
from stages.s3_physics.anchors import STANDING, WALKING

# confidence tiers; a controller acts on HIGH/MED and fails passive on LOW
HIGH, MED, LOW = "high", "medium", "low"

# an S3 verdict maps to the class it ASSERTS; AMBIGUOUS asserts none, the "no second opinion" tier.
# widen this map when a new class gets its own physics verdict (Section 13); tier logic extends unchanged.
S3_TO_CLASS: dict[str, int] = {STANDING: STAND, WALKING: WALK}

# a cell thinner than this is not trusted to set its own label; fail-passive keeps S2. guards a thin
# new-class cell from inventing a call, without touching the stand/walk policy (Section 12).
MIN_CELL_SUPPORT = 20


@dataclass(frozen=True)
class Fusion:
    label: int # STAND / WALK -- the fused call (still emitted on LOW, but flagged)
    confidence: str # HIGH / MED / LOW, from S2-vs-S3 agreement
    abstain: bool # True on LOW: the label is ~coin-flip, a safety-critical controller holds
    s2_proba: float # S2's own class probability, carried as a graded within-tier score


# the confidence tier, read off S2/S3 agreement not either model's probability (Section 12): same class
# -> HIGH; S3 abstaining -> MED (keep S2); different classes -> LOW (a controller holds). no fitting,
# generalizes to any class set unchanged.
def tier_of(s2_pred: int, s3_verdict: str) -> str:
    s3_class = S3_TO_CLASS.get(s3_verdict)
    if s3_class is not None and s3_class == s2_pred:
        return HIGH
    if s3_class is None:
        return MED
    return LOW


# the fused-label policy: the winning class of each (S2 label, S3 verdict) cell, read off the measured
# majority-true of that cell (Section 12). a cell without a trusted majority is absent, so label_for falls
# back to S2. FROZEN by contract: derive once from train+val, never re-derive on the lockbox (Section 7).
@dataclass(frozen=True)
class FusionPolicy:
    label_of: dict                                # (s2_pred, s3_verdict) -> fused label
    support: dict = field(default_factory=dict)   # cell -> n windows it was measured on (provenance)

    # the fused label for one cell; fail-passive to S2 where the policy has no trusted majority
    def label_for(self, s2_pred: int, s3_verdict: str) -> int:
        return self.label_of.get((s2_pred, s3_verdict), s2_pred)


# build a FusionPolicy from measured (s2_pred, s3_verdict, true) rows: each cell's fused label is its
# strict majority true class, kept only with >= MIN_CELL_SUPPORT windows and no tie; otherwise the cell
# is left out so label_for fails passive to S2. derive from train+val, then FREEZE (see FusionPolicy).
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
        if n >= MIN_CELL_SUPPORT and not tied: # trust a clear, well-supported majority
            label_of[cell] = int(top)
    return FusionPolicy(label_of=label_of, support=support)


# fuse one window into a call + confidence. label: from a measured FusionPolicy when given, else the
# FROZEN 2-class default (STAND only when S2 says STAND and physics does not veto toward WALKING). tier
# + abstain are structural (tier_of), identical either way.
def fuse(s2_pred: int, s2_proba: float, s3_verdict: str,
         policy: FusionPolicy | None = None) -> Fusion:
    if policy is None:
        label = STAND if (s2_pred == STAND and s3_verdict != WALKING) else WALK
    else:
        label = policy.label_for(s2_pred, s3_verdict)

    confidence = tier_of(s2_pred, s3_verdict)
    return Fusion(label, confidence, confidence == LOW, float(s2_proba))
