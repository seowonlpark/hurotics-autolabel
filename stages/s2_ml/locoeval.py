"""locoeval: the blind measure layer.

DOMAIN_NOTES §7: this layer emits objective numbers and NO opinion. Every judgement —
"is this good", "should this ship" — belongs above it. Keeping that separation is what
stops a model from being adopted because a narrative sounded convincing.

Headline metric is **macro-F1** (§5.4): the corpus is ~82% walking, so a "predict walk
always" model scores >0.8 accuracy while being useless. Macro-F1 refuses to reward that.

The second headline is the **selective curve**. Once the classifier may abstain, a lone
accuracy number is meaningless without the coverage it was bought at, so the two are
always reported together — plus `worst_rev_accuracy`, the same accuracy on the single
worst held-out subject. Pooled accuracy averages a new subject together with subjects the
model has effectively seen; the worst-rev column is the honest floor for the next person
who wears the device, and the two disagree by several points at every threshold.

`-1` (human-unknown) never enters training and is excluded from metrics consistently (§7),
but its share is reported so the confidence signal can be scored against it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from stages.s2_ml.dataset import STAND, WALK

CLASS_NAMES = {STAND: "stand", WALK: "walk"}

DEFAULT_THRESHOLDS = (0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.98)


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


@dataclass
class ClassMetrics:
    label: str
    support: int
    precision: float
    recall: float
    f1: float


@dataclass
class EvalResult:
    """Numbers only. No verdict, no recommendation."""

    n: int
    macro_f1: float
    accuracy: float
    balanced_accuracy: float
    per_class: list[ClassMetrics]
    confusion: dict[str, dict[str, int]]
    unknown_frac_mean: float
    per_rev_macro_f1: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["per_class"] = [asdict(c) if not isinstance(c, dict) else c for c in self.per_class]
        return d


def evaluate(y_true: np.ndarray, y_pred: np.ndarray,
             groups: np.ndarray | None = None,
             unknown_frac: np.ndarray | None = None) -> EvalResult:
    """Macro-F1 and friends over {stand, walk}. Blind: computes, never judges."""
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
        g = np.asarray(groups)
        for name in pd.unique(g):
            m = g == name
            per_rev[str(name)] = evaluate(y_true[m], y_pred[m]).macro_f1

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


def selective_curve(y_true: np.ndarray, p_walk: np.ndarray,
                    groups: np.ndarray | None = None,
                    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS) -> list[dict]:
    """Coverage vs accuracy as the abstention threshold moves.

    `p_walk` is P(walk); confidence is `max(p, 1-p)`, so threshold 0.5 means "no
    abstention" and the first row is always the full-coverage baseline.
    """
    y_true = np.asarray(y_true)
    p_walk = np.asarray(p_walk, dtype=float)
    conf = np.maximum(p_walk, 1.0 - p_walk)
    correct = np.where(p_walk >= 0.5, WALK, STAND) == y_true
    g = np.asarray(groups) if groups is not None else None

    rows = []
    for thr in thresholds:
        keep = conf >= thr
        n_keep = int(keep.sum())
        worst = float("nan")
        if g is not None and n_keep:
            per = [correct[keep & (g == name)].mean() for name in pd.unique(g)
                   if (keep & (g == name)).any()]
            worst = float(min(per)) if per else float("nan")
        rows.append({
            "threshold": float(thr),
            "coverage": float(keep.mean()) if keep.size else 0.0,
            "n_labeled": n_keep,
            "selective_accuracy": float(correct[keep].mean()) if n_keep else float("nan"),
            "worst_rev_accuracy": worst,
            "errors_kept": int((~correct[keep]).sum()),
        })
    return rows


def render(result: EvalResult, curve: list[dict] | None = None,
           title: str = "locoeval") -> str:
    lines = [
        f"# {title}", "",
        f"- windows: **{result.n:,}**",
        f"- **macro-F1: {result.macro_f1:.4f}**  (headline, §5.4)",
        f"- accuracy: {result.accuracy:.4f}  ·  balanced accuracy: "
        f"{result.balanced_accuracy:.4f}",
        "", "| class | support | precision | recall | F1 |", "|---|---|---|---|---|",
    ]
    for c in result.per_class:
        lines.append(f"| {c.label} | {c.support:,} | {c.precision:.4f} | "
                     f"{c.recall:.4f} | {c.f1:.4f} |")

    lines += ["", "confusion (rows = truth):", "",
              "| | pred stand | pred walk |", "|---|---|---|"]
    for t, row in result.confusion.items():
        lines.append(f"| **{t}** | {row['stand']:,} | {row['walk']:,} |")

    if result.per_rev_macro_f1:
        lines += ["", "per-rev macro-F1 (each rev = one subject/day, §7):", ""]
        for rev, f1 in sorted(result.per_rev_macro_f1.items()):
            lines.append(f"- `{rev}`: {f1:.4f}")

    if curve:
        lines += ["", "## Selective accuracy", "",
                  "Accuracy on the rows the model commits to, against the share it commits "
                  "to. `worst rev` is the same accuracy on the single worst held-out "
                  "subject - the floor for a new person.", "",
                  "| threshold | coverage | selective acc | worst rev | errors kept |",
                  "|---|---|---|---|---|"]
        for r in curve:
            lines.append(f"| {r['threshold']:.2f} | {r['coverage']:.4f} | "
                         f"{r['selective_accuracy']:.4f} | {r['worst_rev_accuracy']:.4f} | "
                         f"{r['errors_kept']:,} |")
    return "\n".join(lines)


def save(result: EvalResult, path: Path, curve: list[dict] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = result.to_dict()
    if curve:
        payload["selective_curve"] = curve
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
