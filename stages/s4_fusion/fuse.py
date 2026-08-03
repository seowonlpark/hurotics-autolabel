"""S4 fusion: one call, one confidence, one reason — the deployable artifact.

The goal this stage exists to serve: **be right about the windows it claims, and say so
when it isn't sure.** A confident wrong label is worse than an abstention, because a
controller acts on it. So every window gets a call, a confidence tier, and — when the
tier is low — the reason and the alternative it was weighing.

Deterministic, no data access, no model. It reads two independent opinions:

  - **S2**, a Random Forest over 23 windowed statistics, with a class probability.
  - **S3**, the swap rule: zero fitted parameters, a different quantity (interleg angle),
    and no exposure to labels at all.

Why two: measured on 4,812 out-of-fold windows, S2's own probability cannot find its
worst errors, and the physics can.

    S2 vs physics    n       S2 accuracy   S2 mean probability
    agree            4,226   0.9785        0.926
    physics abstains   315   0.8635        0.801
    DISAGREE           271   0.3395        0.713

When the swap rule contradicts the classifier, the classifier is wrong two times in
three — while still reporting 0.71 confidence. That is the entire case for this stage.

**The veto is asymmetric, and that is measured, not aesthetic.** Splitting the 271
disagreements by direction:

    S2 says STAND, physics says WALK   n=189   physics right 0.788
    S2 says WALK, physics says STAND   n= 82   physics right 0.366

Physics asserting WALKING is strong evidence; physics asserting STANDING is weak. The
asymmetry has a mechanism: >=2 committed alternations is *positive* evidence that the
legs swapped, while 0 swaps is the *absence* of evidence, which a slow or small-amplitude
stride produces just as readily as genuine standing. Graded by swap count, the same story:
disagreements where physics counted 2-4 swaps, physics is right 0.82; where it counted 0,
0.46 — a coin flip.

So the label follows physics toward WALK only. Whole-corpus accuracy of each option:

    S2 alone                            0.9350
    follow physics both directions      0.9530
    physics vetoes toward WALK only     0.9576   <- adopted
    physics vetoes toward STAND only    0.9304
"""

from __future__ import annotations

from dataclasses import dataclass

from stages.s2_ml.dataset import STAND, WALK
from stages.s3_physics.anchors import AMBIGUOUS, STANDING, WALKING

# Confidence tiers. A controller acts on HIGH, may act on MEDIUM, holds on LOW.
HIGH, MEDIUM, LOW = "high", "medium", "low"

# An S3 verdict maps to the class it ASSERTS. AMBIGUOUS asserts nothing — it is the
# "no second opinion" case, not a vote for either class.
S3_ASSERTS: dict[str, int] = {STANDING: STAND, WALKING: WALK}

# Below this class probability the model is split enough that agreement with physics is
# no longer enough to call it HIGH. Tuned against the corpus to maximise coverage at the
# accuracy target; see `stages/s4_fusion/run.py`, which re-measures it and will say so if
# it drifts.
DEFAULT_PROBA_FLOOR = 0.70

# Reason codes. Deterministic and closed — an agent may explain one, never invent one.
R_CONTRADICTED = "physics_contradicts"      # S3 asserts the other class
R_PHYSICS_ABSTAINS = "physics_abstains"     # exactly 1 swap: a step or a weight shift
R_MODEL_SPLIT = "model_split"               # S2's own probability is low
R_REST_UNTRUSTED = "rest_zero_untrusted"    # recording never rests; calibration is fallback
R_GROWN_SPAN = "verdict_from_grown_span"    # WALKING only via the §10.7 grow


@dataclass(frozen=True)
class Fusion:
    """One window's verdict. `label` is always populated — abstention is a flag, not a hole.

    The caller gets a usable call either way, which is what "guess, but tell me when you
    are guessing" requires. `alternative` is the class the losing opinion argued for, so a
    reviewer sees what the disagreement was *about* rather than only that there was one.
    """

    label: int
    confidence: str
    abstain: bool
    reasons: tuple[str, ...]
    alternative: int | None
    s2_proba: float


def fuse(s2_pred: int, s2_proba: float, s3_verdict: str, *,
         rest_trusted: bool = True, grown: bool = False,
         proba_floor: float = DEFAULT_PROBA_FLOOR) -> Fusion:
    """Fuse one window. Pure function of its arguments — no I/O, no model, no state."""
    asserts = S3_ASSERTS.get(s3_verdict)
    reasons: list[str] = []
    alternative: int | None = None

    if asserts is not None and asserts != s2_pred:
        # Contradiction. Emit the call the evidence favours by direction (see module
        # docstring), flag it low, and name the option that lost.
        label = WALK if (s2_pred == STAND and s3_verdict == WALKING) else s2_pred
        alternative = s2_pred if label != s2_pred else asserts
        reasons.append(R_CONTRADICTED)
        confidence = LOW
    else:
        label = s2_pred
        if asserts is None:                      # physics abstained
            reasons.append(R_PHYSICS_ABSTAINS)
            confidence = MEDIUM if s2_proba >= proba_floor else LOW
            alternative = WALK if s2_pred == STAND else STAND
        else:                                    # both agree
            confidence = HIGH if s2_proba >= proba_floor else MEDIUM

    if s2_proba < proba_floor:
        reasons.append(R_MODEL_SPLIT)
    # Quality caveats. They never change the label — they explain why a call is soft, and
    # a flag that silently rewrote the answer would be the opposite of an audit trail.
    if not rest_trusted:
        reasons.append(R_REST_UNTRUSTED)
    if grown and s3_verdict == WALKING:
        reasons.append(R_GROWN_SPAN)

    return Fusion(int(label), confidence, confidence == LOW,
                  tuple(reasons), alternative, float(s2_proba))


def explain(f: Fusion) -> str:
    """One human sentence for a fused window. Used in the report and the agent queue."""
    name = {STAND: "stand", WALK: "walk"}
    parts = {
        R_CONTRADICTED: "the physics reads the opposite class",
        R_PHYSICS_ABSTAINS: "the physics saw one leg crossing — a step or a weight shift",
        R_MODEL_SPLIT: f"the model was split ({f.s2_proba:.2f})",
        R_REST_UNTRUSTED: "this recording never rests, so its calibration is a fallback",
        R_GROWN_SPAN: "walking was only detected over a widened span (slow gait)",
    }
    why = "; ".join(parts[r] for r in f.reasons if r in parts)
    head = f"{name.get(f.label, f.label)} ({f.confidence})"
    if not why:
        return head
    alt = (f" — the alternative was {name.get(f.alternative, f.alternative)}"
           if f.alternative is not None else "")
    return f"{head}: {why}{alt}"
