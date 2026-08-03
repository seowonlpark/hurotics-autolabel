"""locoeval: the blind measure layer.

DOMAIN_NOTES §7: this layer emits objective numbers and NO opinion. Every judgement —
"is this good", "should this be promoted" — belongs to the agent/gate above it. Keeping
that separation is what stops a model from being promoted because a narrative sounded
convincing.

Headline metric is **macro-F1** (§5.4): the corpus is ~83% walking, so a
"predict walk always" model scores >0.8 accuracy while being useless. Macro-F1 refuses
to reward that.

`-1` (human-unknown) never enters training and is excluded from the metrics consistently
(§7), but its share is reported so a future confidence signal can be scored against it.

SCOPE NOTE: §7's precedence-ordered MECE row-level taxonomy is NOT here — it lives in
`taxonomy.py`, ported verbatim from `locoeval/diagnose.py`, and is fed row-level
predictions by `predict.py`. (An earlier version of this note said its semantics were
"not written down anywhere in this repo" and listed a chain ending in `remainder`;
both were wrong — `steady_confusion` absorbs what precedence leaves, and `omission`
splits three ways. See DOMAIN_NOTES §7.)

What stays here is the WINDOW-level view: per-class rates, the confusion matrix, and
`transition_report`'s timing counts. Window-level and row-level are different
resolutions of the same question and are not interchangeable — a 2 s window cannot
express a 200 ms flicker (see `taxonomy.py`).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from stages.s2_ml.dataset import STAND, WALK

CLASS_NAMES = {STAND: "stand", WALK: "walk"}


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


def _contiguous_chunks(g: pd.DataFrame, stride_ms: float) -> list[pd.DataFrame]:
    """Split one (rev, trial, segment) group where windows are not actually adjacent.

    The caller hands us a FILTERED frame — `TRANSITION` windows have already been
    dropped, because they are not training targets. Their rows are gone but the time
    they occupied is not, so two windows that are neighbours in the array can be
    seconds apart on the clock. Treating them as adjacent is §11 item 2 ("a bout
    analysis deleted UNKNOWNs *then* computed runs, silently merging across gaps") —
    and, measured on the current corpus, it merged 213 of 244 counted boundaries.

    So adjacency is decided by the clock, never by array position. A neighbour more
    than 1.5 strides away means at least one window was removed in between.
    """
    t = g["t_start_ms"].to_numpy(float)
    cuts = np.flatnonzero(np.diff(t) > 1.5 * stride_ms) + 1
    bounds = [0, *cuts.tolist(), len(g)]
    return [g.iloc[a:b] for a, b in zip(bounds[:-1], bounds[1:])]


def infer_stride_ms(df: pd.DataFrame) -> float:
    """Smallest positive window spacing in the frame — the stride, when anything survives."""
    d = np.diff(np.sort(df["t_start_ms"].to_numpy(float)))
    d = d[d > 0]
    return float(d.min()) if d.size else 0.0


def transition_report(df: pd.DataFrame, y_pred: np.ndarray,
                      stride_ms: float | None = None) -> dict:
    """Timing behaviour around true state changes, at WINDOW resolution.

    The §7 row-level partition is `taxonomy.py`'s; this is the cheap window-level view
    that comes free with the CV predictions. Read it as such: its unit is one window,
    so it can neither see a 200 ms flicker nor place a boundary finer than one stride.

    Measured only across genuinely adjacent windows (`_contiguous_chunks`). Boundaries
    that fall across a removed window are excluded and COUNTED, not silently dropped —
    a large `merged_boundaries_excluded` means most transitions live in the windows
    this split threw away, which is itself the finding.

    - `flicker_rate`: predicted switches per window inside label-steady runs. A steady
      truth run should produce zero switches; every switch is spurious.
    - `boundary_error_windows`: signed offset (in windows) between each true transition
      and the nearest predicted transition. Negative = early, positive = late.
    """
    d = df.reset_index(drop=True).copy()
    d["pred"] = y_pred
    stride = stride_ms if stride_ms else infer_stride_ms(d)
    flick_switch = flick_windows = excluded = 0
    offsets: list[int] = []

    for (_rev, _trial, _seg), g in d.groupby(["rev", "trial", "segment"], sort=True):
        g = g.sort_values("t_start_ms")
        if stride <= 0:
            continue
        chunks = _contiguous_chunks(g, stride)
        # A label change ACROSS a chunk boundary is a real transition we cannot time.
        for prev, nxt in zip(chunks[:-1], chunks[1:]):
            excluded += int(prev["label"].iloc[-1] != nxt["label"].iloc[0])

        for chunk in chunks:
            truth = chunk["label"].to_numpy()
            pred = chunk["pred"].to_numpy()
            if truth.size < 2:
                continue

            t_switch = np.flatnonzero(truth[1:] != truth[:-1]) + 1
            p_switch = np.flatnonzero(pred[1:] != pred[:-1]) + 1

            steady = np.ones(truth.size - 1, dtype=bool)
            for s in t_switch:                  # exclude the true boundary neighbourhood
                steady[max(0, s - 2):min(steady.size, s + 1)] = False
            flick_switch += int(np.sum((pred[1:] != pred[:-1]) & steady))
            flick_windows += int(steady.sum())

            for s in t_switch:
                if p_switch.size:
                    offsets.append(int(p_switch[np.argmin(np.abs(p_switch - s))] - s))

    off = np.array(offsets) if offsets else np.array([], dtype=int)
    return {
        "true_transitions": int(off.size),
        "merged_boundaries_excluded": excluded,
        "stride_ms": stride,
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


def render(result: EvalResult, transitions: dict | None = None, title: str = "locoeval") -> str:
    lines = [
        f"# {title}",
        "",
        f"- windows: **{result.n:,}**",
        f"- **macro-F1: {result.macro_f1:.4f}**  (headline, §5.4)",
        f"- accuracy: {result.accuracy:.4f}  ·  balanced accuracy: {result.balanced_accuracy:.4f}",
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
        lines += ["", "per-rev macro-F1 (each rev = one subject/day, §7):", ""]
        for rev, f1 in sorted(result.per_rev_macro_f1.items()):
            lines.append(f"- `{rev}`: {f1:.4f}")

    if transitions:
        b = transitions["boundary_error_windows"]
        lines += [
            "", "## Transition behaviour (window resolution)", "",
            f"- timeable transitions: **{transitions['true_transitions']}**"
            f"  ·  excluded as non-adjacent: **{transitions.get('merged_boundaries_excluded', 0)}**",
            f"- flicker rate (spurious switches per steady window): **{transitions['flicker_rate']:.4f}**",
            f"- boundary error (windows): median {b['median']:+.1f}, mean|err| {b['mean_abs']:.2f}",
            f"- early {b['early']} · on-time {b['on_time']} · late {b['late']}",
            "",
            "*Window-level, so a boundary resolves no finer than one stride and a sub-window*",
            "*flicker is inexpressible. The §7 row-level partition is `taxonomy.py`, scored on*",
            "*dense inference. Excluded boundaries fall inside `transition` windows, which are*",
            "*not training targets — a large count means the transitions live where this split*",
            "*cannot see them.*",
        ]
    return "\n".join(lines)


def save(result: EvalResult, path: Path, transitions: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = result.to_dict()
    if transitions:
        payload["transitions"] = transitions
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
