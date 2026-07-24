# S3 discriminator registry: a discriminator is a named physical test mapping anchors to a verdict. the
# swap rule is a bespoke callable; a new class declares a ThresholdRule over the anchor vocabulary, so its
# test is auditable data. pure infrastructure, imports nothing from anchors.py to avoid a cycle (Section 13).

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

# comparison ops a declarative rule may use; bounded on purpose so a spec stays closed and reviewable
_OPS: dict[str, Callable[[float, float], bool]] = {
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
}


# a declarative discriminator body: ALL clauses must hold for the rule to fire. each clause is
# (anchor, op, value), read over the flat anchor dict window_anchors produces.
@dataclass(frozen=True)
class ThresholdRule:
    clauses: tuple[tuple[str, str, float], ...]

    def holds(self, anchors: dict) -> bool:
        for name, op, value in self.clauses:
            x = anchors.get(name)
            if x is None or not _OPS[op](float(x), float(value)):
                return False
        return True


# one registered discriminator. verdict_of maps a window's anchor dict to a verdict string; kind is
# `callable` (bespoke physics) or `threshold` (declarative spec). origin marks provenance: `code` is a
# trusted built-in, `proposed` still routes to a human before it can change a call (Section 11.2).
@dataclass(frozen=True)
class Discriminator:
    name: str                          # registry key
    emits: str                         # the verdict/class label asserted when the test fires
    verdict_of: Callable[[dict], str]  # anchor dict -> verdict string
    classes: tuple[str, ...]           # every verdict label verdict_of can return
    kind: str                          # "callable" (bespoke physics) or "threshold" (declarative)
    origin: str = "code"               # "code" (built-in) or "proposed" (needs_human)


_REGISTRY: dict[str, Discriminator] = {}


# register a discriminator; a duplicate name fails loud rather than silently overwriting
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


# the gate a declarative spec must clear: a non-empty list of {anchor, op, value} clauses, every anchor
# known, every op bounded, every threshold finite. raises ValueError on the first fault.
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


# build a discriminator from a validated declarative spec: emits `emits` when every clause holds,
# else `negative` (default "", abstain). origin defaults to `proposed`, needs_human until adopted.
def from_spec(name: str, emits: str, spec: list, known_anchors: set[str],
              negative: str = "", origin: str = "proposed") -> Discriminator:
    validate_spec(spec, known_anchors)
    rule = ThresholdRule(tuple((c["anchor"], c["op"], float(c["value"])) for c in spec))
    classes = (emits, negative) if negative else (emits,)
    return Discriminator(name=name, emits=emits,
                         verdict_of=lambda a: emits if rule.holds(a) else negative,
                         classes=classes, kind="threshold", origin=origin)
