"""Dense-stride inference: turn a windowed classifier into per-row predictions.

The model decides over a window; the error taxonomy (`taxonomy.py`) measures runs and
lags in milliseconds. Scoring 2 s window labels with millisecond thresholds would be
meaningless — a piecewise-constant-over-2 s prediction CANNOT emit a run shorter than
`FLICKER_MAX_MS`, so flicker would read zero by construction rather than by merit.

So inference slides the same window at a small stride and assigns each prediction to the
rows around its centre. Prediction resolution becomes the stride, not the window:
at the 100 ms default a spurious single-step run is 100 ms — below the 200 ms flicker
threshold, hence expressible. Training is unaffected; this is inference-side only.

Centre assignment, not leading-edge: a window's label describes the whole span it covers,
so attributing it to the middle keeps transitions unbiased. Attributing it to the start
would shift every predicted transition half a window late and manufacture `late` rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stages.s2_ml.dataset import FEATURES, HUMAN_UNKNOWN, LABEL_COL, TIME_COL, TRAIN_CLASSES
from stages.s2_ml.features import (
    WindowSpec,
    feature_names,
    rest_reference,
    segment_features,
)

# 100 ms: finer than FLICKER_MAX_MS (200 ms) so flicker is expressible, coarse enough
# that dense inference over the corpus stays minutes rather than hours.
DEFAULT_INFERENCE_STRIDE_S = 0.1


def dense_predict_segment(model, seg: pd.DataFrame, feats: list[str], spec: WindowSpec,
                          zeros: dict[str, float], ileg_zero: float,
                          stride_s: float = DEFAULT_INFERENCE_STRIDE_S) -> np.ndarray:
    """Per-row predictions for one gap-free segment.

    `zeros` / `ileg_zero` are the RECORDING's rest reference, passed in rather than
    measured here. A per-segment zero would calibrate each segment against itself, so the
    same posture would read differently either side of a gap — and it would not be the
    zero the model was trained against, which is train/serve skew by construction.

    Rows before the first window centre and after the last inherit the nearest
    prediction: the edges are genuinely unobservable at this window length, and
    extrapolating the nearest label is honest about that rather than inventing a class.
    """
    n_rows = len(seg)
    n_win = spec.n
    if n_rows < n_win:
        return np.full(n_rows, -1, dtype=int)  # too short to score; caller drops these

    # Same extractor as training, at a finer stride — not a second implementation. The
    # dense path used to loop a per-window function; that function no longer exists, and
    # reintroducing one here is exactly the train/serve skew `verify_features.py` guards.
    dense = WindowSpec(window_s=spec.window_s, stride_s=stride_s, fs_hz=spec.fs_hz)
    chan = {c: seg[c].to_numpy(float) for c in FEATURES}
    X, starts = segment_features(chan, dense, zeros, ileg_zero)
    if X.shape[0] == 0:
        return np.full(n_rows, -1, dtype=int)

    frame = pd.DataFrame(X, columns=feature_names())
    win_pred = model.predict(frame[feats].to_numpy(float))

    out = np.empty(n_rows, dtype=int)
    centres = [int(s) + n_win // 2 for s in starts]
    for k, c in enumerate(centres):
        lo = c if k == 0 else (centres[k - 1] + c) // 2
        hi = c + 1 if k == len(centres) - 1 else (c + centres[k + 1]) // 2
        out[lo:hi] = win_pred[k]
    out[:centres[0]] = win_pred[0]          # leading edge
    out[centres[-1]:] = win_pred[-1]        # trailing edge
    return out


def labeled_runs(gt: np.ndarray, pred: np.ndarray, t: np.ndarray):
    """Split into contiguous runs where ground truth is a trainable class.

    `-1` (human-unknown) is excluded from scoring (§5.2/§7) — the model has no `-1` to
    predict, so counting those rows as errors would measure the labeller's hesitation,
    not the model. Splitting (rather than deleting) preserves contiguity and timing, so
    a `-1` stretch reads as a recording boundary, which is what it effectively is.
    """
    ok = np.isin(gt, TRAIN_CLASSES)
    if not ok.any():
        return []
    edges = np.flatnonzero(np.diff(ok.astype(int)) != 0) + 1
    runs = []
    for s, e in zip([0, *edges], [*edges, len(ok)]):
        if ok[s] and e - s >= 2:
            runs.append((gt[s:e], pred[s:e], t[s:e]))
    return runs


def dense_predict_trial(model, frame: pd.DataFrame, feats: list[str], spec: WindowSpec,
                        stride_s: float = DEFAULT_INFERENCE_STRIDE_S):
    """Yield (gt, pred, time_ms) runs for one trial, segment by segment.

    The rest reference is measured ONCE for the recording and shared across its segments,
    matching how `features.windows_of_trial` calibrates at training time.
    """
    frame = frame.reset_index(drop=True)
    zeros, ileg_zero, _trusted = rest_reference(frame, spec.fs_hz)
    for _seg_id, seg in frame.groupby("segment", sort=True):
        seg = seg.reset_index(drop=True)
        pred = dense_predict_segment(model, seg, feats, spec, zeros, ileg_zero, stride_s)
        if np.all(pred == -1):
            continue
        gt = seg[LABEL_COL].to_numpy()
        t = seg[TIME_COL].to_numpy(float)
        yield from labeled_runs(gt, pred, t)


__all__ = [
    "DEFAULT_INFERENCE_STRIDE_S",
    "dense_predict_segment",
    "dense_predict_trial",
    "labeled_runs",
    "HUMAN_UNKNOWN",
]
