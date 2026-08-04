# S2 windowing + feature extraction: the model-agnostic boundary
# never window across a gap; a mixed window is kept but excluded from training
# per-file calibration, never corpus-wide
# interleg exists because no absolute amplitude threshold separates the classes
# all windows of a segment at once- dense inference cannot afford a Python loop

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from stages.s1_clean.config import CANONICAL_HZ
from stages.s2_ml.dataset import (
    FEATURES,
    HUMAN_UNKNOWN,
    LABEL_COL,
    STAND,
    TIME_COL,
    TRAIN_CLASSES,
    Trial,
    WALK,
)
from stages.s2_ml.rest import (
    ANGLE_CHANNELS,
    REST_ANCHOR_S,
    SWAP_DELTA_DEG,
    is_rest,
    rest_span_frame,
)

# Non-overlapping for training: overlapping windows manufacture near-duplicate rows, which
# inflate the apparent sample count and flatter any metric computed on them
DEFAULT_WINDOW_S = 2.0
DEFAULT_STRIDE_S = 2.0

# A training window must be label-pure; anything less is a transition (rule 2)
PURITY_MIN = 1.0

# the band a stride can plausibly occupy for this population: cadence 16-102 steps/min
# deliberately NOT the healthy-adult (0.5, 3.0) Hz band
GAIT_BAND_HZ = (0.13, 3.0)
HF_BAND_HZ = (GAIT_BAND_HZ[1], 15.0)
CYCLE_BAND_S = (0.6, 2.5)
MAX_LAG_S = 1.0

TRANSITION = "transition"

# The tail excluded from each annotated class when measuring the ambiguity band below
# 1%, not 5%: at 5% the band narrows to roughly [6.5, 11.9]° and 11 of 41 trials contain
# no band window at all, which makes the per-trial policy statistic in
# `s3_physics.label_audit` undefined for a quarter of the corpus; at 1% one trial is empty
BAND_TAIL_PCT = 1.0

META_COLUMNS = {"rev", "trial", "split", "segment", "t_start_ms", "start_row",
                "label", "purity", "unknown_frac", "rest_trusted"}


# the interleg-amplitude interval where the two ANNOTATED classes overlap
# inside it BOTH human labels genuinely occur, so no amplitude rule separates it
# NO model output goes into either edge, which is what keeps label_audit model-free
# degenerate input returns (inf, -inf), which every lo <= x <= hi test reads as empty
def amplitude_band(minhalf, labels, tail_pct: float = BAND_TAIL_PCT) -> tuple[float, float]:
    minhalf = np.asarray(minhalf, dtype=float)
    labels = np.asarray(labels)
    walk = minhalf[(labels == WALK) & np.isfinite(minhalf)]
    stand = minhalf[(labels == STAND) & np.isfinite(minhalf)]
    if not len(walk) or not len(stand):
        return float("inf"), float("-inf")
    return (float(np.percentile(walk, tail_pct)),
            float(np.percentile(stand, 100.0 - tail_pct)))


@dataclass
class WindowSpec:
    window_s: float = DEFAULT_WINDOW_S
    stride_s: float = DEFAULT_STRIDE_S
    fs_hz: float = CANONICAL_HZ

    @property
    def n(self) -> int:
        return int(round(self.window_s * self.fs_hz))

    @property
    def step(self) -> int:
        return int(round(self.stride_s * self.fs_hz))


# rest.swap_count for every window [s, s+n) at once; swaps == sign changes with BOTH
# endpoints inside the window, and "both endpoints" is what a plain prefix sum gets wrong
# it also counts the change carried by the first committed sample, whose predecessor is
# outside; the final subtraction removes that; verify_features holds the two to agreement
def swap_counts(d: np.ndarray, starts: np.ndarray, n: int,
                delta: float = SWAP_DELTA_DEG) -> np.ndarray:
    s = np.where(d > delta, 1, np.where(d < -delta, -1, 0))
    nz = np.flatnonzero(s)
    change = np.zeros(d.size, dtype=np.int64)
    if nz.size > 1:
        change[nz[1:]] = (s[nz[1:]] != s[nz[:-1]]).astype(np.int64)
    pre = np.concatenate([[0], np.cumsum(change)])
    total = pre[starts + n] - pre[starts]
    if nz.size == 0:
        return total
    k = np.searchsorted(nz, starts)
    inside = k < nz.size
    first = np.where(inside, nz[np.minimum(k, nz.size - 1)], 0)
    return total - np.where(inside & (first < starts + n), change[first], 0)


def _corr(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    a = A - A.mean(1, keepdims=True)
    b = B - B.mean(1, keepdims=True)
    den = np.sqrt((a * a).sum(1)) * np.sqrt((b * b).sum(1))
    out = np.zeros(A.shape[0])
    ok = den > 0
    out[ok] = (a[ok] * b[ok]).sum(1) / den[ok]
    return out


def _power(A: np.ndarray) -> np.ndarray:
    return np.abs(np.fft.rfft((A - A.mean(1, keepdims=True)) * np.hanning(A.shape[1]),
                              axis=1)) ** 2


# (dominant frequency in the gait band, fraction of non-DC power inside it)
def _spectral(A: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    P = _power(A)
    freq = np.fft.rfftfreq(A.shape[1], 1.0 / fs)
    band = (freq >= GAIT_BAND_HZ[0]) & (freq <= GAIT_BAND_HZ[1])
    total = P[:, 1:].sum(1)
    dom = np.zeros(A.shape[0])
    frac = np.zeros(A.shape[0])
    ok = total > 0
    if band.any():
        Pb = P[:, band]
        dom[ok] = freq[band][Pb[ok].argmax(1)]
        frac[ok] = Pb[ok].sum(1) / total[ok]
    flat = ~np.any(A, axis=1)
    dom[flat] = 0.0
    frac[flat] = 0.0
    return dom, frac


# power fraction above the gait band; the transform already low-passed at 1 Hz so most of
# it is gone, but dropping the whole spectral block measurably lowers worst-subject accuracy
def _hf_ratio(A: np.ndarray, fs: float) -> np.ndarray:
    P = _power(A)
    freq = np.fft.rfftfreq(A.shape[1], 1.0 / fs)
    band = (freq >= HF_BAND_HZ[0]) & (freq <= HF_BAND_HZ[1])
    total = P[:, 1:].sum(1)
    out = np.zeros(A.shape[0])
    ok = total > 0
    if band.any():
        out[ok] = P[:, band][ok].sum(1) / total[ok]
    out[~np.any(A, axis=1)] = 0.0
    return out


# standardized k-th central moment; 0 where the window is constant
def _moment(A: np.ndarray, k: int) -> np.ndarray:
    c = A - A.mean(1, keepdims=True)
    s = A.std(1)
    out = np.zeros(A.shape[0])
    ok = s > 0
    out[ok] = (c[ok] ** k).mean(1) / s[ok] ** k
    return out


def _autocorr(A: np.ndarray) -> np.ndarray:
    n = A.shape[1]
    c = A - A.mean(1, keepdims=True)
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    F = np.fft.rfft(c, nfft, axis=1)
    return np.fft.irfft(F * np.conj(F), nfft, axis=1)[:, :n]


# stride period from the first in-band autocorrelation peak; sub-bin by construction,
# unlike _spectral's 1/window_s grid, which at 2 s is 0.5 Hz
def _cycle_s(A: np.ndarray, fs: float) -> np.ndarray:
    n = A.shape[1]
    lo, hi = int(CYCLE_BAND_S[0] * fs), min(int(CYCLE_BAND_S[1] * fs), n - 1)
    if hi <= lo:
        return np.zeros(A.shape[0])
    ac = _autocorr(A)
    out = np.zeros(A.shape[0])
    ok = A.std(1) > 0
    out[ok] = (ac[ok, lo:hi + 1].argmax(1) + lo) / fs
    return out


# inter-leg timing offset from the cross-correlation peak; level gait is a half cycle out
# of phase
def _lag_s(A: np.ndarray, B: np.ndarray, fs: float) -> np.ndarray:
    n = A.shape[1]
    sa, sb = A.std(1), B.std(1)
    ok = (sa > 0) & (sb > 0)
    out = np.zeros(A.shape[0])
    if not ok.any():
        return out
    a = (A[ok] - A[ok].mean(1, keepdims=True)) / sa[ok, None]
    b = (B[ok] - B[ok].mean(1, keepdims=True)) / sb[ok, None]
    m = min(int(MAX_LAG_S * fs), n - 1)
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    cc = np.fft.irfft(np.fft.rfft(a, nfft, axis=1) * np.conj(np.fft.rfft(b, nfft, axis=1)),
                      nfft, axis=1)
    full = np.concatenate([cc[:, nfft - m:], cc[:, :m + 1]], axis=1)
    out[ok] = (full.argmax(1) - m) / fs
    return out


# smallest interleg ptp among k equal sub-windows; NOT the whole-window ptp, since one
# weight shift reads high on that while only one part of the window moves- continuous gait
# moves in every part, which is what separates walking from a standing subject who turns
def _min_over_parts(D: np.ndarray, k: int) -> np.ndarray:
    n = D.shape[1] // k
    if n < 2:
        return np.zeros(D.shape[0])
    return np.minimum.reduce([D[:, i * n:(i + 1) * n].max(1) - D[:, i * n:(i + 1) * n].min(1)
                              for i in range(k)])


# the feature vector's column order; the single source of truth for it
def feature_names() -> list[str]:
    names: list[str] = []
    for c in FEATURES:
        # the static-offset family is deliberately absent: it carries the subject's
        # zeroing bias rather than gait, and keeping it hurt held-out subjects
        stats = ("std", "ptp") if c in ANGLE_CHANNELS else ("mean", "std", "ptp", "absmean")
        names += [f"{c}_{s}" for s in stats]
    names += ["ang_LR_corr", "angvel_LR_corr"]
    for side in ("L", "R"):
        names += [f"{side}_angvel_dom_hz", f"{side}_angvel_band_frac",
                  f"{side}_angvel_skew", f"{side}_angvel_kurt",
                  f"{side}_angvel_posfrac", f"{side}_angvel_peakratio",
                  f"{side}_angvel_hf_ratio", f"{side}_angacc_rms", f"{side}_cycle_s",
                  f"{side}_ang_p95_rest", f"{side}_ang_med_rest", f"{side}_ang_p95_p05"]
    names += ["angvel_LR_lag_s", "ileg_minhalf", "ileg_minquarter", "ileg_swaps"]
    return names


# (X, start_row) for every window of one gap-free segment; zeros/ileg_zero are required,
# not defaulted- substituting one silently is the train/serve skew this pipeline avoids
def segment_features(chan: dict[str, np.ndarray], spec: WindowSpec,
                     zeros: dict[str, float], ileg_zero: float
                     ) -> tuple[np.ndarray, np.ndarray]:
    n, step, fs = spec.n, spec.step, spec.fs_hz
    W = {c: sliding_window_view(chan[c], n)[::step] for c in FEATURES}
    starts = np.arange(W[FEATURES[0]].shape[0]) * step
    d = chan[ANGLE_CHANNELS[0]] - chan[ANGLE_CHANNELS[1]] - ileg_zero
    D = sliding_window_view(d, n)[::step]

    cols: dict[str, np.ndarray] = {}
    for c in FEATURES:
        A = W[c]
        if c not in ANGLE_CHANNELS:
            cols[f"{c}_mean"] = A.mean(1)
            cols[f"{c}_absmean"] = np.abs(A).mean(1)
        cols[f"{c}_std"] = A.std(1)
        cols[f"{c}_ptp"] = A.max(1) - A.min(1)

    cols["ang_LR_corr"] = _corr(W[ANGLE_CHANNELS[0]], W[ANGLE_CHANNELS[1]])
    cols["angvel_LR_corr"] = _corr(W[FEATURES[2]], W[FEATURES[3]])

    for side in ("L", "R"):
        V = W[f"{side}_angvel_LPF"]
        dom, frac = _spectral(V, fs)
        cols[f"{side}_angvel_dom_hz"] = dom
        cols[f"{side}_angvel_band_frac"] = frac
        cols[f"{side}_angvel_skew"] = _moment(V, 3)
        cols[f"{side}_angvel_kurt"] = _moment(V, 4) - 3.0
        cols[f"{side}_angvel_posfrac"] = (V > 0).mean(1)
        cols[f"{side}_angvel_peakratio"] = V.max(1) / (np.abs(V.min(1)) + 1e-6)
        cols[f"{side}_angvel_hf_ratio"] = _hf_ratio(V, fs)
        cols[f"{side}_angacc_rms"] = np.sqrt(((np.diff(V, axis=1) * fs) ** 2).mean(1))
        cols[f"{side}_cycle_s"] = _cycle_s(V, fs)

        A = W[f"{side}_ang_LPF"]
        p95 = np.percentile(A, 95, axis=1)
        p05 = np.percentile(A, 5, axis=1)
        z = zeros[f"{side}_ang_LPF"]
        # How far the thigh lifts above THIS subject's own standing posture
        cols[f"{side}_ang_p95_rest"] = p95 - z
        cols[f"{side}_ang_med_rest"] = np.median(A, 1) - z
        cols[f"{side}_ang_p95_p05"] = p95 - p05

    cols["angvel_LR_lag_s"] = _lag_s(W[FEATURES[2]], W[FEATURES[3]], fs)
    cols["ileg_minhalf"] = _min_over_parts(D, 2)
    cols["ileg_minquarter"] = _min_over_parts(D, 4)
    cols["ileg_swaps"] = swap_counts(d, starts, n).astype(float)

    names = feature_names()
    return np.column_stack([cols[k] for k in names]), starts


# (per-side rest posture, rest interleg offset, trusted) for one recording; untrusted means
# it never rests and the zeros fall back to whole-recording medians- returned, not raised,
# so the flag travels with the features and label.py can say so instead of guessing
def rest_reference(frame: pd.DataFrame, fs: float = CANONICAL_HZ
                   ) -> tuple[dict[str, float], float, bool]:
    span = int(round(REST_ANCHOR_S * fs))
    seg0 = frame[frame["segment"] == frame["segment"].min()]
    opening = seg0.iloc[:span]
    best = None
    if len(opening) == span and is_rest(
            (opening[ANGLE_CHANNELS[0]] - opening[ANGLE_CHANNELS[1]]).to_numpy(float), span):
        best = opening
    else:
        best = rest_span_frame(frame, span)
    if best is None:
        zeros = {c: float(frame[c].median()) for c in ANGLE_CHANNELS}
        return zeros, zeros[ANGLE_CHANNELS[0]] - zeros[ANGLE_CHANNELS[1]], False
    zeros = {c: float(best[c].median()) for c in ANGLE_CHANNELS}
    return zeros, zeros[ANGLE_CHANNELS[0]] - zeros[ANGLE_CHANNELS[1]], True


# (label, purity, unknown_fraction) per window; -1 is excluded before voting but its share
# is reported; a window not pure over {stand, walk} is TRANSITION, never a coin-flip vote
def label_windows(labels_2d: np.ndarray):
    unknown = (labels_2d == HUMAN_UNKNOWN).mean(1)
    counts = np.stack([(labels_2d == c).sum(1) for c in TRAIN_CLASSES], axis=1)
    valid = counts.sum(1)
    purity = np.where(valid > 0, counts.max(1) / np.maximum(valid, 1), 0.0)
    winner = np.array(TRAIN_CLASSES, dtype=object)[counts.argmax(1)]
    label = np.where(purity >= PURITY_MIN, winner, TRANSITION)
    return np.where(valid == 0, None, label), purity, unknown


# every window of one trial, never spanning a gap
def windows_of_trial(trial: Trial, spec: WindowSpec) -> pd.DataFrame:
    frame = trial.frame.reset_index(drop=True)
    if frame.empty:
        return pd.DataFrame()
    zeros, ileg_zero, trusted = rest_reference(frame, spec.fs_hz)
    out = []
    for seg_id, seg in frame.groupby("segment", sort=True):
        seg = seg.reset_index(drop=True)
        if len(seg) < spec.n:
            continue
        chan = {c: seg[c].to_numpy(float) for c in FEATURES}
        X, starts = segment_features(chan, spec, zeros, ileg_zero)
        if X.shape[0] == 0:
            continue
        lab, purity, unknown = label_windows(
            sliding_window_view(seg[LABEL_COL].to_numpy(), spec.n)[::spec.step])
        meta = pd.DataFrame({
            "rev": trial.rev, "trial": trial.trial, "split": trial.split,
            "segment": int(seg_id), "t_start_ms": seg[TIME_COL].to_numpy(float)[starts],
            "start_row": starts, "label": lab, "purity": purity,
            "unknown_frac": unknown, "rest_trusted": trusted,
        })
        out.append(pd.concat([meta, pd.DataFrame(X, columns=feature_names())], axis=1))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def build_windows(trials: list[Trial], spec: WindowSpec | None = None) -> pd.DataFrame:
    spec = spec or WindowSpec()
    frames = [w for t in trials if not (w := windows_of_trial(t, spec)).empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# feature columns only, never the metadata or the target
def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META_COLUMNS]
