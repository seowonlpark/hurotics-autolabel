# locoeval: blind measure layer, NO opinion- macro-F1 leads, since ~86% walk rewards a stub

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


# numbers only; no verdict, no recommendation
@dataclass
class EvalResult:
    n: int
    macro_f1: float
    accuracy: float
    balanced_accuracy: float
    per_class: list[ClassMetrics]
    confusion: dict[str, dict[str, int]]
    unknown_frac_mean: float
    per_rev_macro_f1: dict[str, float] = field(default_factory=dict)
    # ALONGSIDE macro-F1, never instead: accuracy rewards the degenerate model, macro-F1 hides subjects
    per_rev_accuracy: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["per_class"] = [asdict(c) if not isinstance(c, dict) else c for c in self.per_class]
        return d


# macro-F1 and friends over {stand, walk}; computes, never judges
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
    per_rev_acc: dict[str, float] = {}
    if groups is not None:
        g = np.asarray(groups)
        for name in pd.unique(g):
            m = g == name
            sub = evaluate(y_true[m], y_pred[m])
            per_rev[str(name)] = sub.macro_f1
            per_rev_acc[str(name)] = sub.accuracy

    return EvalResult(
        n=int(y_true.size),
        macro_f1=float(np.mean(f1s)),
        accuracy=float(np.mean(y_true == y_pred)) if y_true.size else 0.0,
        balanced_accuracy=float(np.mean(recalls)),
        per_class=per_class,
        confusion=confusion,
        unknown_frac_mean=float(np.mean(unknown_frac)) if unknown_frac is not None else 0.0,
        per_rev_macro_f1=per_rev,
        per_rev_accuracy=per_rev_acc,
    )


# coverage vs accuracy as the threshold moves; confidence is max(p, 1-p), so 0.5 is no abstention
def selective_curve(y_true: np.ndarray, p_walk: np.ndarray,
                    groups: np.ndarray | None = None,
                    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS) -> list[dict]:
    y_true = np.asarray(y_true)
    p_walk = np.asarray(p_walk, dtype=float)
    conf = np.maximum(p_walk, 1.0 - p_walk)
    correct = np.where(p_walk >= 0.5, WALK, STAND) == y_true
    g = np.asarray(groups) if groups is not None else None

    rows = []
    for thr in thresholds:
        keep = conf >= thr
        n_keep = int(keep.sum())
        worst, worst_rev = float("nan"), None
        if g is not None and n_keep:
            # carry the NAME, not just the minimum- it cannot be recovered once dropped here
            per = [(float(correct[keep & (g == name)].mean()), str(name)) for name in pd.unique(g)
                   if (keep & (g == name)).any()]
            # ties break on the name, so the column does not wander between runs
            worst, worst_rev = min(per) if per else (float("nan"), None)
        rows.append({
            "threshold": float(thr),
            "coverage": float(keep.mean()) if keep.size else 0.0,
            "n_labeled": n_keep,
            "selective_accuracy": float(correct[keep].mean()) if n_keep else float("nan"),
            "worst_rev_accuracy": worst,
            "worst_rev": worst_rev,
            "errors_kept": int((~correct[keep]).sum()),
        })
    return rows


def render(result: EvalResult, curve: list[dict] | None = None,
           title: str = "locoeval") -> str:
    lines = [
        f"# {title}", "",
        f"- windows: **{result.n:,}**",
        f"- **macro-F1: {result.macro_f1:.4f}**  (headline)",
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
        lines += ["", "per-rev held-out scores (each rev = one subject/day):", "",
                  "| rev | macro-F1 | window accuracy |", "|---|---|---|"]
        for rev, f1 in sorted(result.per_rev_macro_f1.items()):
            acc = result.per_rev_accuracy.get(rev)
            acc_s = f"{acc * 100:.2f}%" if acc is not None else "-"
            lines.append(f"| `{rev}` | {f1:.4f} | {acc_s} |")

    if curve:
        lines += ["", "## Selective accuracy", "",
                  "Accuracy on the rows the model commits to, against the share it commits "
                  "to. `worst rev` is the same accuracy on the single worst held-out "
                  "subject - the floor for a new person.", "",
                  "| threshold | coverage | selective acc | worst rev | errors kept |",
                  "|---|---|---|---|---|"]
        for r in curve:
            worst = f"{r['worst_rev_accuracy']:.4f}"
            if r.get("worst_rev"):
                worst += f" (`{r['worst_rev']}`)"
            lines.append(f"| {r['threshold']:.2f} | {r['coverage']:.4f} | "
                         f"{r['selective_accuracy']:.4f} | {worst} | "
                         f"{r['errors_kept']:,} |")
    return "\n".join(lines)


def save(result: EvalResult, path: Path, curve: list[dict] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = result.to_dict()
    if curve:
        payload["selective_curve"] = curve
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
