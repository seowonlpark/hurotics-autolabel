# dense-stride inference: turn a windowed classifier into per-row predictions
# slides the window at a small stride and assigns each prediction to rows around its
# centre, so prediction resolution is the stride, not the window -- otherwise the ms-scale
# error taxonomy couldn't see flicker. centre (not leading-edge) keeps transitions
# unbiased. inference-side only, training unaffected. see README.

from __future__ import annotations

import numpy as np
import pandas as pd

from stages.s2_ml.dataset import HUMAN_UNKNOWN, LABEL_COL, TIME_COL, TRAIN_CLASSES
from stages.s2_ml.features import WindowSpec, window_features

# 100 ms: finer than FLICKER_MAX_MS (200 ms) so flicker is expressible, coarse enough
# that dense inference over the corpus stays minutes not hours
DEFAULT_INFERENCE_STRIDE_S = 0.1


# per-row predictions for one gap-free segment; edge rows inherit the nearest window's
# prediction -- they're genuinely unobservable at this window length
def dense_predict_segment(model, seg: pd.DataFrame, feats: list[str], spec: WindowSpec,
                          stride_s: float = DEFAULT_INFERENCE_STRIDE_S) -> np.ndarray:
    n_rows = len(seg)
    step = max(1, int(round(stride_s * spec.fs_hz)))
    n_win = spec.n

    if n_rows < n_win:
        return np.full(n_rows, -1, dtype=int) # too short to score; caller drops these

    starts = list(range(0, n_rows - n_win + 1, step))
    # one window_features call per window, then project onto feats -- recomputing inside
    # the feature loop would redo every FFT once per feature, 20x+ the work
    rows = []
    for s in starts:
        f = window_features(seg.iloc[s:s + n_win], spec.fs_hz)
        rows.append([f[name] for name in feats])
    win_pred = model.predict(np.array(rows, dtype=float))

    out = np.empty(n_rows, dtype=int)
    centres = [s + n_win // 2 for s in starts]
    for k, c in enumerate(centres):
        lo = c if k == 0 else (centres[k - 1] + c) // 2
        hi = c + 1 if k == len(centres) - 1 else (c + centres[k + 1]) // 2
        out[lo:hi] = win_pred[k]
    out[:centres[0]] = win_pred[0] # leading edge
    out[centres[-1]:] = win_pred[-1] # trailing edge
    return out


# split into contiguous runs where ground truth is a trainable class; -1 (human-unknown)
# is excluded from scoring (Section 5.2/Section 7). splitting preserves contiguity so -1 reads as a boundary
def labeled_runs(gt: np.ndarray, pred: np.ndarray, t: np.ndarray):
    ok = np.isin(gt, TRAIN_CLASSES)
    if not ok.any():
        return []
    edges = np.flatnonzero(np.diff(ok.astype(int)) != 0) + 1
    runs = []
    for s, e in zip([0, *edges], [*edges, len(ok)]):
        if ok[s] and e - s >= 2:
            runs.append((gt[s:e], pred[s:e], t[s:e]))
    return runs


# yield (gt, pred, time_ms) runs for one trial, segment by segment
def dense_predict_trial(model, frame: pd.DataFrame, feats: list[str], spec: WindowSpec,
                        stride_s: float = DEFAULT_INFERENCE_STRIDE_S):
    for _seg_id, seg in frame.groupby("segment", sort=True):
        seg = seg.reset_index(drop=True)
        pred = dense_predict_segment(model, seg, feats, spec, stride_s)
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
