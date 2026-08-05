# dense-stride inference -> per-row predictions; resolution is the stride, centred not leading-edge

from __future__ import annotations

import numpy as np
import pandas as pd

from stages.s2_ml.dataset import FEATURES, LABEL_COL, TIME_COL, TRAIN_CLASSES
from stages.s2_ml.features import WindowSpec, feature_names, rest_reference, segment_features

# 100 ms: finer than FLICKER_MAX_MS so flicker is expressible; NOT train.DEFAULT_INFERENCE_STRIDE_S
DEFAULT_INFERENCE_STRIDE_S = 0.1


# per-row predictions for one segment; edge rows inherit the nearest window- genuinely unobservable
def dense_predict_segment(model, seg: pd.DataFrame, feats: list[str], spec: WindowSpec,
                          stride_s: float = DEFAULT_INFERENCE_STRIDE_S,
                          ref: tuple[dict[str, float], float] | None = None) -> np.ndarray:
    # rest posture is a property of the RECORDING: the caller measures it once and passes it down
    if ref is None:
        z, ileg, _trusted = rest_reference(seg, spec.fs_hz)
        ref = (z, ileg)
    zeros, ileg_zero = ref
    n_rows = len(seg)
    step = max(1, int(round(stride_s * spec.fs_hz)))
    n_win = spec.n

    if n_rows < n_win:
        return np.full(n_rows, -1, dtype=int) # too short to score; caller drops these

    # every window at once, through the SAME `segment_features` that built the training rows
    wspec = WindowSpec(window_s=spec.window_s, stride_s=stride_s, fs_hz=spec.fs_hz)
    chan = {c: seg[c].to_numpy(float) for c in FEATURES}
    X, starts = segment_features(chan, wspec, zeros, ileg_zero)
    if X.shape[0] == 0:
        return np.full(n_rows, -1, dtype=int)
    order = feature_names()
    win_pred = model.predict(X[:, [order.index(f) for f in feats]])
    starts = list(starts)

    out = np.empty(n_rows, dtype=int)
    centres = [s + n_win // 2 for s in starts]
    for k, c in enumerate(centres):
        lo = c if k == 0 else (centres[k - 1] + c) // 2
        hi = c + 1 if k == len(centres) - 1 else (c + centres[k + 1]) // 2
        out[lo:hi] = win_pred[k]
    out[:centres[0]] = win_pred[0] # leading edge
    out[centres[-1]:] = win_pred[-1] # trailing edge
    return out


# contiguous runs where truth is trainable; -1 is excluded from scoring and reads as a boundary
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
    # over the WHOLE recording, as at training time: a per-segment zero would skew every other one
    zeros, ileg_zero, _trusted = rest_reference(frame, spec.fs_hz)
    for _seg_id, seg in frame.groupby("segment", sort=True):
        seg = seg.reset_index(drop=True)
        pred = dense_predict_segment(model, seg, feats, spec, stride_s, (zeros, ileg_zero))
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
]
