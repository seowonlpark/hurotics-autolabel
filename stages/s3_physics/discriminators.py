# S3 discriminator registry: the seam that lets the physics view carry MORE than stand/walk.
# a discriminator is a named physical test that maps a window's anchors to a verdict. today there
# is exactly one -- the swap rule (Section 10) -- and it is registered here as a bespoke CALLABLE,
# its implementation unchanged (it lives in anchors.py, this only catalogs it). the point of the
# registry is the OTHER kind: a new class gets a physics second-opinion by declaring a
# ThresholdRule over the EXISTING anchor vocabulary -- antiphase, grav_stab, periodicity, ... --
# not by hand-writing a new verdict function. a declarative rule can be rate-audited and reviewed
# exactly like an S2 challenger spec ([S2-3]): bounded ops, known anchors, no arbitrary code, so
# code can measure it and a human still decides (the S4 governance contract, Section 11.2).
#
# this module is pure infrastructure -- it imports nothing from anchors.py, so anchors.py can
# register the swap rule at import time without a cycle. see README / DOMAIN_NOTES Section 10, Section 13.

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

# the comparison ops a declarative rule may use -- bounded on purpose, the S2 validate_spec
# discipline ([S2-3]): a discriminator spec is data, so its vocabulary is closed and reviewable.
_OPS: dict[str, Callable[[float, float], bool]] = {
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
}


# a declarative discriminator body: ALL clauses must hold for the rule to fire. each clause is
# (anchor, op, value), read over the flat anchor dict window_anchors already produces. no arbitrary
# code -- this is why a proposed class's physics test can be audited instead of trusted.
@dataclass(frozen=True)
class ThresholdRule:
    clauses: tuple[tuple[str, str, float], ...]

    def holds(self, anchors: dict) -> bool:
        for name, op, value in self.clauses:
            x = anchors.get(name)
            if x is None or not _OPS[op](float(x), float(value)):
                return False
        return True


# one registered discriminator. verdict_of maps a window's anchor dict to a verdict string; a
# bespoke `callable` (the swap rule) carries validated physics too subtle for a threshold, while a
# `threshold` is built from a declarative spec via from_spec. origin marks provenance: `code` is a
# built-in the pipeline trusts, `proposed` is an agent/human spec that -- like every S4 proposal --
# still routes to a human before it can change a call (Section 11.2).
@dataclass(frozen=True)
class Discriminator:
    name: str                          # registry key
    emits: str                         # the verdict/class label asserted when the test fires
    verdict_of: Callable[[dict], str]  # anchor dict -> verdict string
    classes: tuple[str, ...]           # every verdict label verdict_of can return
    kind: str                          # "callable" (bespoke physics) or "threshold" (declarative)
    origin: str = "code"               # "code" (built-in) or "proposed" (needs_human)


_REGISTRY: dict[str, Discriminator] = {}


# register a discriminator; a duplicate name is a bug (two classes fighting over one key), not a
# silent overwrite -- fail loud, the [X-3] no-silent-mutation discipline applied to the catalog.
def register(d: Discriminator) -> Discriminator:
    if d.name in _REGISTRY:
        raise ValueError(f"discriminator {d.name!r} already registered")
    _REGISTRY[d.name] = d
    return d


def get(name: str) -> Discriminator:
    return _REGISTRY[name]


# a copy of the catalog (callers must not mutate the live registry)
def registry() -> dict[str, Discriminator]:
    return dict(_REGISTRY)


# the gate a declarative spec must clear before it becomes a discriminator: a non-empty list of
# {anchor, op, value} clauses, every anchor drawn from the known vocabulary, every op bounded, every
# threshold a finite number. the S3 analogue of S2 validate_spec ([S2-3]) -- a proposed physics test
# is data, and data that does not clear this gate never runs. raises ValueError on the first fault.
def validate_spec(spec: object, known_anchors: set[str]) -> None:
    if not isinstance(spec, list) or not spec:
        raise ValueError("discriminator spec must be a non-empty list of clauses")
    for c in spec:
        if not isinstance(c, dict) or set(c) != {"anchor", "op", "value"}:
            raise ValueError(f"clause must have exactly anchor/op/value, got {c!r}")
        if c["anchor"] not in known_anchors:
            raise ValueError(f"unknown anchor {c['anchor']!r}; known: {sorted(known_anchors)}")
        if c["op"] not in _OPS:
            raise ValueError(f"unknown op {c['op']!r}; allowed: {sorted(_OPS)}")
        try:
            v = float(c["value"])
        except (TypeError, ValueError):
            raise ValueError(f"threshold value must be numeric, got {c['value']!r}")
        if v != v or v in (float("inf"), float("-inf")):
            raise ValueError(f"threshold value must be finite, got {c['value']!r}")


# build a discriminator from a validated declarative spec. it emits `emits` when every clause holds
# and `negative` (default "", i.e. abstain) otherwise. origin defaults to `proposed`: a spec that
# came from outside the built-in physics is needs_human until a person adopts it (Section 11.2). this
# is the whole extension path -- a new class's physics second-opinion, with no new verdict code.
def from_spec(name: str, emits: str, spec: list, known_anchors: set[str],
              negative: str = "", origin: str = "proposed") -> Discriminator:
    validate_spec(spec, known_anchors)
    rule = ThresholdRule(tuple((c["anchor"], c["op"], float(c["value"])) for c in spec))
    classes = (emits, negative) if negative else (emits,)
    return Discriminator(name=name, emits=emits,
                         verdict_of=lambda a: emits if rule.holds(a) else negative,
                         classes=classes, kind="threshold", origin=origin)
