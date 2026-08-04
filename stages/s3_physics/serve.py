"""The swap rule on the SERVE grid: a physics verdict for a file that has no labels.

`label_audit` points the swap rule at the annotations, which is the only thing S3 has been
independent of since S2 absorbed the interleg block. This module points it at a raw
recording instead — the files a customer actually sends — so that two consumers can exist:

  - `s2_ml/label.py`'s physics floor/ceiling (default OFF, and see the warning below)
  - `s3_physics/plausibility.py`'s file-level sanity bounds (default ON)

**Read the warning before wiring this into a decision [measured, 2026-08-03].** The swap
rule is *not* an independent second opinion on the classifier. S2 reads `ileg_swaps`,
`ileg_minhalf` and `ileg_minquarter` off the same 1 Hz-filtered interleg angle this rule
reads, so the two are wrong together on exactly the windows where that signal is
ambiguous: physics contradicts only ~12% of S2's high-confidence errors, and at p >= 0.95
it catches **none** of them. Removing the interleg block from S2 to restore the separation
was tried and does not help — the correlation is in the signal, not the feature list
(`caveats.md` §1.1c). A per-window gate built on this will therefore fire almost nowhere
that matters, which is why `label.py` ships it off and measures it rather than trusting it.

The FILE-level use is a different instrument and does not inherit that objection. "This
recording contains no leg alternation anywhere, and the model called 80% of it walking" is
not a second opinion about a window; it is a statement that the two disagree about the
whole recording, which happens when a channel is swapped or an axis is misread — not when
a window is hard. `plausibility.py` is the consumer that matters.

## Why this returns windows and not rows

`segment_verdicts` mirrors `features.segment_features` exactly — same grid walk, same
`(values, starts)` shape — and stops there. Mapping windows onto rows is `label.py`'s
`_accumulate`, and it stays `label.py`'s: a row's physics has to be averaged over the same
covering windows, by the same difference-array, as that row's probability. A second
window-to-row mapping living here could drift from the one the probability uses, and then
a gate would compare two quantities that describe different rows while looking like they
describe the same one.
"""

from __future__ import annotations

import numpy as np

from stages.s2_ml.dataset import FEATURES
from stages.s2_ml.features import WindowSpec
from stages.s3_physics.anchors import AMBIGUOUS, STANDING, WALKING, adaptive_verdict

# Row-level verdict bands, applied to the swap count AVERAGED over the windows covering a
# row. Deliberately the same arithmetic `label.py` already applies to `ileg_swaps` for its
# `weight_shift_or_step` reason — `(swaps >= 0.5) & (swaps < 1.5)` — so the serve path has
# one notion of "the interleg signal crossed about once here", not two that round
# differently at the boundary.
#
# Averaging a categorical verdict is the alternative and is worse: a row covered by eight
# windows, five STANDING and three WALKING, has no majority verdict worth the name, while
# its mean swap count is a real number the bands read the same way they read any other.
ROW_STANDING_MAX = 0.5
ROW_WALKING_MIN = 1.5


def row_verdict(mean_swaps: np.ndarray) -> np.ndarray:
    """STANDING / AMBIGUOUS / WALKING per row, from the mean adaptive swap count.

    NaN (no window covers the row) yields AMBIGUOUS rather than a verdict: an uncovered
    row has no physics, and calling it standing because zero windows found zero swaps is
    the arithmetic of an empty set masquerading as evidence.
    """
    out = np.full(mean_swaps.shape, AMBIGUOUS, dtype=object)
    with np.errstate(invalid="ignore"):
        out[mean_swaps < ROW_STANDING_MAX] = STANDING
        out[mean_swaps >= ROW_WALKING_MIN] = WALKING
    out[~np.isfinite(mean_swaps)] = AMBIGUOUS
    return out


def segment_verdicts(chan: dict[str, np.ndarray], spec: WindowSpec,
                     ileg_zero: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(swaps, span_s, starts) for every window of one gap-free segment.

    The grid is `np.arange(n_windows) * step`, the identical walk `segment_features`
    performs, so `starts` can be compared element-for-element against the feature grid's
    — and `label.py` asserts exactly that rather than trusting this sentence.

    `ileg_zero` is required, not defaulted, for the reason `segment_features` gives: the
    per-file rest zero is what the swap count is taken about (§10.4), and substituting one
    silently is the train/serve skew this pipeline exists to avoid.
    """
    l = chan[FEATURES[0]]
    r = chan[FEATURES[1]]
    d = (l - r) - ileg_zero
    n_windows = max(0, (len(d) - spec.n) // spec.step + 1)
    starts = np.arange(n_windows) * spec.step

    swaps = np.empty(n_windows, dtype=float)
    span_s = np.empty(n_windows, dtype=float)
    for i, start in enumerate(starts):
        # `start + spec.n // 2` is the cell centre, the same one `trial_anchors` passes.
        _verdict, sp, sw, _a, _b = adaptive_verdict(d, l, r, int(start) + spec.n // 2, spec)
        swaps[i] = sw
        span_s[i] = sp
    return swaps, span_s, starts


__all__ = ["AMBIGUOUS", "STANDING", "WALKING", "row_verdict", "segment_verdicts",
           "ROW_STANDING_MAX", "ROW_WALKING_MIN"]
