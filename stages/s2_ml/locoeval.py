# locoeval: the blind measure layer, emits objective numbers with NO opinion (Section 7). headline
# metric is macro-F1 (the corpus is ~83% walk, so accuracy flatters "predict walk always"); -1 windows
# are excluded. reports only the unambiguous parts: per-class rates, confusion, transition timing.

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from stages.s2_ml.dataset import STAND, WALK

CLASS_NAMES = {STAND: "stand", WALK: "walk"}


# precision, recall, f1 from tp/fp/fn; 0 on empty denominators
def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


# per-class precision/recall/f1 with support
@dataclass
class ClassMetrics:
    label: str # class name
    support: int # true rows of this class
    precision: float
    recall: float
    f1: float


# numbers only -- no verdict, no recommendation
@dataclass
class EvalResult:
    n: int # windows scored
    macro_f1: float # headline metric (Section 5.4)
    accuracy: float
    balanced_accuracy: float
    per_class: list[ClassMetrics]
    confusion: dict[str, dict[str, int]] # rows = truth
    unknown_frac_mean: float # mean -1 share
    per_rev_macro_f1: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["per_class"] = [asdict(c) if not isinstance(c, dict) else c for c in self.per_class]
        return d


# macro-F1 and friends over {stand, walk}; blind -- computes, never judges
def evaluate(y_true: np.ndarray, y_pred: np.ndarray,
             groups: np.ndarray | None = None,
             unknown_frac: np.ndarray | None = None) -> EvalResult:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = [STAND, WALK]

    per_class, f1s, recalls = [], [], []
    for c in classes:
        tp = int(np.sum((y_true == c) & (y_pred == c)))
        fp = int(np.sum((y_true != c) & (y_pred == c)))
        fn = int(np.sum((y_true == c) & (y_pred != c)))
        p, r, f = _prf(tp, fp, fn)
        per_class.append(ClassMetrics(CLASS_NAMES[c], int(np.sum(y_true == c)), p, r, f))
        f1s.append(f)
        recalls.append(r)

    confusion = {
        CLASS_NAMES[t]: {CLASS_NAMES[p]: int(np.sum((y_true == t) & (y_pred == p)))
                         for p in classes}
        for t in classes
    }

    per_rev: dict[str, float] = {}
    if groups is not None:
        for g in pd.unique(np.asarray(groups)):
            m = np.asarray(groups) == g
            sub = evaluate(y_true[m], y_pred[m])
            per_rev[str(g)] = sub.macro_f1

    return EvalResult(
        n=int(y_true.size),
        macro_f1=float(np.mean(f1s)),
        accuracy=float(np.mean(y_true == y_pred)) if y_true.size else 0.0,
        balanced_accuracy=float(np.mean(recalls)),
        per_class=per_class,
        confusion=confusion,
        unknown_frac_mean=float(np.mean(unknown_frac)) if unknown_frac is not None else 0.0,
        per_rev_macro_f1=per_rev,
    )


# timing behaviour around true state changes -- the substrate for late/early/flicker.
# raw counts, not the Section 7 taxonomy (its precedence rules are unspecified here). flicker_rate
# = spurious switches per steady window; boundary_error = signed window offset (neg=early)
def transition_report(df: pd.DataFrame, y_pred: np.ndarray) -> dict:
    d = df.reset_index(drop=True).copy()
    d["pred"] = y_pred
    flick_switch = flick_windows = 0
    offsets: list[int] = []

    for (_rev, _trial, _seg), g in d.groupby(["rev", "trial", "segment"], sort=True):
        g = g.sort_values("t_start_ms")
        truth = g["label"].to_numpy()
        pred = g["pred"].to_numpy()
        if truth.size < 2:
            continue

        t_switch = np.flatnonzero(truth[1:] != truth[:-1]) + 1
        p_switch = np.flatnonzero(pred[1:] != pred[:-1]) + 1

        steady = np.ones(truth.size - 1, dtype=bool)
        for s in t_switch: # exclude the true boundary neighbourhood
            steady[max(0, s - 2):min(steady.size, s + 1)] = False
        flick_switch += int(np.sum((pred[1:] != pred[:-1]) & steady))
        flick_windows += int(steady.sum())

        for s in t_switch:
            if p_switch.size:
                offsets.append(int(p_switch[np.argmin(np.abs(p_switch - s))] - s))

    off = np.array(offsets) if offsets else np.array([], dtype=int)
    return {
        "true_transitions": int(off.size),
        "flicker_rate": float(flick_switch / flick_windows) if flick_windows else 0.0,
        "flicker_switches": flick_switch,
        "boundary_error_windows": {
            "median": float(np.median(off)) if off.size else 0.0,
            "mean_abs": float(np.mean(np.abs(off))) if off.size else 0.0,
            "early": int(np.sum(off < 0)),
            "on_time": int(np.sum(off == 0)),
            "late": int(np.sum(off > 0)),
        },
    }


# render an eval result (+ optional transitions) as the locoeval.md report
def render(result: EvalResult, transitions: dict | None = None, title: str = "locoeval") -> str:
    lines = [
        f"# {title}",
        "",
        f"- windows: **{result.n:,}**",
        f"- **macro-F1: {result.macro_f1:.4f}**  (headline, Section 5.4)",
        f"- accuracy: {result.accuracy:.4f}  -  balanced accuracy: {result.balanced_accuracy:.4f}",
        "",
        "| class | support | precision | recall | F1 |",
        "|---|---|---|---|---|",
    ]
    for c in result.per_class:
        lines.append(f"| {c.label} | {c.support:,} | {c.precision:.4f} | {c.recall:.4f} | {c.f1:.4f} |")

    lines += ["", "confusion (rows = truth):", "", "| | pred stand | pred walk |", "|---|---|---|"]
    for t, row in result.confusion.items():
        lines.append(f"| **{t}** | {row['stand']:,} | {row['walk']:,} |")

    if result.per_rev_macro_f1:
        lines += ["", "per-rev macro-F1 (each rev = one subject/day, Section 7):", ""]
        for rev, f1 in sorted(result.per_rev_macro_f1.items()):
            lines.append(f"- `{rev}`: {f1:.4f}")

    if transitions:
        b = transitions["boundary_error_windows"]
        lines += [
            "", "## Transition behaviour", "",
            f"- true transitions: **{transitions['true_transitions']}**",
            f"- flicker rate (spurious switches per steady window): **{transitions['flicker_rate']:.4f}**",
            f"- boundary error (windows): median {b['median']:+.1f}, mean|err| {b['mean_abs']:.2f}",
            f"- early {b['early']} - on-time {b['on_time']} - late {b['late']}",
            "",
            "*Not the Section 7 taxonomy - its precedence semantics are unspecified in this repo.*",
        ]
    return "\n".join(lines)


# write the eval result (+ optional transitions) as locoeval.json
def save(result: EvalResult, path: Path, transitions: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = result.to_dict()
    if transitions:
        payload["transitions"] = transitions
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
