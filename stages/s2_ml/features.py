# S2 windowing + feature extraction: the model-agnostic boundary
# feature selection lives HERE, not in the clean layer (§9); this is the surface the S2
# experimenter proposes changes to. rules: never window across a gap, a mixed window is
# `transition` (excluded from training, not forced), window length is an open tradeoff
# (§9 -- slow gait runs to 0.13 Hz). see README / DOMAIN_NOTES.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from stages.s1_clean.config import CANONICAL_HZ
from stages.s2_ml.dataset import (
    FEATURES,
    HUMAN_UNKNOWN,
    LABEL_COL,
    TIME_COL,
    TRAIN_CLASSES,
    Trial,
)

# non-overlapping by default: overlapping windows manufacture near-duplicate rows and
# flatter any metric computed on them
DEFAULT_WINDOW_S = 2.0
DEFAULT_STRIDE_S = 2.0

# a training window must be label-pure; anything less is a transition (rule 2)
PURITY_MIN = 1.0

# the band a stride can plausibly occupy for this population (§9): cadence 16-102
# steps/min. deliberately NOT the healthy-adult (0.5, 3.0) Hz band
GAIT_BAND_HZ = (0.13, 3.0)

TRANSITION = "transition"


# window/stride in samples, derived from seconds and rate
@dataclass
class WindowSpec:
    window_s: float = DEFAULT_WINDOW_S # window length in seconds
    stride_s: float = DEFAULT_STRIDE_S # hop between windows in seconds
    fs_hz: float = CANONICAL_HZ # sampling rate

    # window length in samples
    @property
    def n(self) -> int:
        return int(round(self.window_s * self.fs_hz))

    # stride in samples
    @property
    def step(self) -> int:
        return int(round(self.stride_s * self.fs_hz))


# (dominant frequency in the gait band, fraction of power inside the band); resolution is
# 1/window_s, so at 2 s the low edge is unresolvable -- the §9 tradeoff made visible
def _spectral(x: np.ndarray, fs: float) -> tuple[float, float]:
    x = x - x.mean()
    if x.size < 4 or not np.any(x):
        return 0.0, 0.0
    power = np.abs(np.fft.rfft(x * np.hanning(x.size))) ** 2
    freq = np.fft.rfftfreq(x.size, 1.0 / fs)
    band = (freq >= GAIT_BAND_HZ[0]) & (freq <= GAIT_BAND_HZ[1])
    total = power[1:].sum() # drop DC
    if not band.any() or total <= 0:
        return 0.0, 0.0
    return float(freq[band][np.argmax(power[band])]), float(power[band].sum() / total)


# pearson r; 0 when either series is constant or too short
def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


# features for one window: time-domain per channel + cross-leg + spectral. cross-leg corr
# earns its place physically -- walking swings the legs in antiphase, standing doesn't
def window_features(win: pd.DataFrame, fs: float) -> dict[str, float]:
    feats: dict[str, float] = {}
    arrays = {c: win[c].to_numpy(float) for c in FEATURES}

    for name, v in arrays.items():
        feats[f"{name}_mean"] = float(v.mean())
        feats[f"{name}_std"] = float(v.std())
        feats[f"{name}_ptp"] = float(np.ptp(v)) # np.ptp: ndarray.ptp() gone in NumPy 2 (§8)
        feats[f"{name}_absmean"] = float(np.abs(v).mean())

    feats["ang_LR_corr"] = _corr(arrays["L_ang_LPF"], arrays["R_ang_LPF"])
    feats["angvel_LR_corr"] = _corr(arrays["L_angvel_LPF"], arrays["R_angvel_LPF"])
    feats["ang_LR_offset"] = float(np.median(arrays["L_ang_LPF"]) - np.median(arrays["R_ang_LPF"]))

    for side in ("L", "R"):
        dom, frac = _spectral(arrays[f"{side}_angvel_LPF"], fs)
        feats[f"{side}_angvel_dom_hz"] = dom
        feats[f"{side}_angvel_band_frac"] = frac

    return feats


# (label, purity, unknown_fraction) for a window; -1 excluded before voting (§5.2) but its
# share reported. not pure over {stand, walk} => TRANSITION, never a coin-flip vote
def label_window(labels: np.ndarray) -> tuple[object, float, float]:
    unknown_frac = float(np.mean(labels == HUMAN_UNKNOWN))
    valid = labels[np.isin(labels, TRAIN_CLASSES)]
    if valid.size == 0:
        return None, 0.0, unknown_frac
    values, counts = np.unique(valid, return_counts=True)
    purity = float(counts.max() / valid.size)
    winner = int(values[counts.argmax()])
    return (winner if purity >= PURITY_MIN else TRANSITION), purity, unknown_frac


# every window of one trial, never spanning a gap
def windows_of_trial(trial: Trial, spec: WindowSpec) -> pd.DataFrame:
    rows = []
    frame = trial.frame
    if frame.empty:
        return pd.DataFrame()

    for seg_id, seg in frame.groupby("segment", sort=True):
        seg = seg.reset_index(drop=True)
        labels = seg[LABEL_COL].to_numpy()
        for start in range(0, len(seg) - spec.n + 1, spec.step):
            stop = start + spec.n
            label, purity, unk = label_window(labels[start:stop])
            if label is None:
                continue
            win = seg.iloc[start:stop]
            rows.append({
                "rev": trial.rev, "trial": trial.trial, "split": trial.split,
                "segment": int(seg_id), "t_start_ms": float(win[TIME_COL].iloc[0]),
                "label": label, "purity": purity, "unknown_frac": unk,
                **window_features(win, spec.fs_hz),
            })
    return pd.DataFrame(rows)


# windows for every trial, stacked
def build_windows(trials: list[Trial], spec: WindowSpec | None = None) -> pd.DataFrame:
    spec = spec or WindowSpec()
    frames = [w for t in trials if not (w := windows_of_trial(t, spec)).empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# feature columns only -- never the metadata or the target
def feature_columns(df: pd.DataFrame) -> list[str]:
    meta = {"rev", "trial", "split", "segment", "t_start_ms", "label", "purity", "unknown_frac"}
    return [c for c in df.columns if c not in meta]


# build windows at the default spec and print the split x label crosstab
def main() -> None:
    from stages.s2_ml.dataset import load_dataset

    spec = WindowSpec()
    df = build_windows(load_dataset(), spec)
    trainable = df[df["label"] != TRANSITION]
    print(f"[s2] window={spec.window_s}s stride={spec.stride_s}s @ {spec.fs_hz} Hz "
          f"-> {len(df):,} windows, {len(feature_columns(df))} features")
    print(f"[s2] trainable (label-pure): {len(trainable):,}  "
          f"transition (excluded from training): {len(df) - len(trainable):,}")
    print()
    print(pd.crosstab(df["split"], df["label"]).to_string())


if __name__ == "__main__":
    main()
