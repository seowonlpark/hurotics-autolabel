"""S4 fusion: one call, one confidence, one reason — the deployable artifact.

The goal this stage exists to serve: **be right about the windows it claims, and say so
when it isn't sure.** A confident wrong label is worse than an abstention, because a
controller acts on it. So every window gets a call, a confidence tier, and — when the
tier is low — the reason and the alternative it was weighing.

Deterministic, no data access, no model. It reads two independent opinions:

  - **S2**, a Random Forest over 23 windowed statistics, with a class probability.
  - **S3**, the swap rule: zero fitted parameters, a different quantity (interleg angle),
    and no exposure to labels at all.

Why two: measured on 4,812 out-of-fold windows, S2's own probability does not find its
worst errors as well as an independent opinion does.

    S2 vs physics    n       S2 accuracy   S2 mean probability
    agree            4,307   0.9807        0.947
    physics abstains   316   0.8734        0.856
    DISAGREE           189   0.5344        0.708

A window where the swap rule contradicts the classifier is a coin flip that the
classifier reports 0.71 confidence on. Spending the same number of abstentions on the
disagreements rather than on the lowest-probability windows catches materially more real
errors — 0.4656 of the disagreeing windows are wrong versus 0.3757 of the equally-sized
lowest-probability set. That, not the label override below, is what this stage is for.

**These numbers are the SECOND measurement, and the first is why they are quoted.**
Before the S2 feature set absorbed `ileg_swaps` and the rest-anchored interleg block,
the same table read n=271 disagreements at S2 accuracy 0.3395, with physics right 0.788
in the stand->walk direction. Teaching the model the swap count moved most of that signal
inside S2, exactly as it should: fewer disagreements, and the survivors closer to even.
A constant justified against the first measurement and never re-checked would now be
folklore — re-run `stages/s4_fusion/run.py`, which re-measures the whole curve.

**Physics does NOT override the label, and that is a deliberate removal.** An earlier
version let it veto toward WALK, justified when physics was right 0.788 of the time in
that direction. After the feature rework the same measurement read **0.566 on 76
windows** — worth about +0.2 points of accuracy, which is not an edge, it is noise with a
rationale attached. Dropped (Lu, 2026-08-03).

The mechanism that motivated it is still real: >=2 committed alternations is *positive*
evidence the legs swapped, while 0 swaps is the *absence* of evidence, which slow or
small-amplitude gait produces as readily as standing. That asymmetry is why physics
disagreement is worth flagging. It is not enough to overrule a model that now reads the
swap count itself.

So the split of responsibility is clean: **S2 decides what the window is, S3 decides
whether to believe it.** The losing opinion is still reported as `alternative`, so a
reviewer sees what the disagreement was about rather than only that there was one.
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

    # The label is always S2's. Physics sets the confidence, never the call.
    label = s2_pred

    if asserts is not None and asserts != s2_pred:
        reasons.append(R_CONTRADICTED)
        alternative = asserts                    # what the physics argued for instead
        confidence = LOW
    elif asserts is None:                        # physics abstained
        reasons.append(R_PHYSICS_ABSTAINS)
        alternative = WALK if s2_pred == STAND else STAND
        confidence = MEDIUM if s2_proba >= proba_floor else LOW
    else:                                        # both agree
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
