"""S2 windowing + feature extraction: the model-agnostic boundary.

Per DOMAIN_NOTES §9 this is where feature selection lives — NOT in the clean layer.
The clean layer keeps the honest superset; this module decides what a model sees, and
it is the surface the S2 experimenter agent proposes changes to.

Three rules it exists to enforce:

  1. **Never window across a gap.** A window is drawn inside one segment. Spanning a
     gap would invent continuity that was never measured (§3.1).
  2. **A mixed window is not a training example.** If a window spans a stand->walk
     transition its label is genuinely ambiguous, mirroring the human `-1` (§5.2).
     Such windows are kept and marked `transition`, but excluded from training targets:
     abstain rather than force (§7).
  3. **Window length is an OPEN TRADEOFF, not a constant** (§9). 2 s gives transition
     precision; 4 s is needed for slow gait — this population runs to 0.13 Hz, far below
     the healthy-adult band. Hence `window_s` is a parameter, and the frequency features
     below are only meaningful when the window actually spans a few strides.
"""

from __future__ import annotations

from collections.abc import Iterator
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

# Defaults. Non-overlapping by default: overlapping windows manufacture near-duplicate
# rows, which inflates apparent sample count and flatters any metric computed on them.
DEFAULT_WINDOW_S = 2.0
DEFAULT_STRIDE_S = 2.0

# A training window must be label-pure. Anything less is a transition (rule 2).
PURITY_MIN = 1.0

# The band a stride can plausibly occupy for this population (§9): cadence 16-102
# steps/min. Deliberately NOT the healthy-adult (0.5, 3.0) Hz band.
GAIT_BAND_HZ = (0.13, 3.0)

TRANSITION = "transition"


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


def _spectral(x: np.ndarray, fs: float) -> tuple[float, float]:
    """(dominant frequency in the gait band, fraction of power inside the band).

    Resolution is 1/window_s, so at 2 s the low edge of the band is unresolvable — the
    number is still computed, but it is near-meaningless for slow gait. That is the §9
    tradeoff made visible rather than hidden.
    """
    x = x - x.mean()
    if x.size < 4 or not np.any(x):
        return 0.0, 0.0
    power = np.abs(np.fft.rfft(x * np.hanning(x.size))) ** 2
    freq = np.fft.rfftfreq(x.size, 1.0 / fs)
    band = (freq >= GAIT_BAND_HZ[0]) & (freq <= GAIT_BAND_HZ[1])
    total = power[1:].sum()  # drop DC
    if not band.any() or total <= 0:
        return 0.0, 0.0
    return float(freq[band][np.argmax(power[band])]), float(power[band].sum() / total)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def window_features(win: pd.DataFrame, fs: float) -> dict[str, float]:
    """Features for one window. Time-domain per channel + cross-leg + spectral.

    Cross-leg correlation earns its place on physical grounds: walking swings the legs
    in antiphase, standing does not — it separates the two classes by mechanism rather
    than by amplitude, so it should survive a change of subject or walking speed.
    """
    feats: dict[str, float] = {}
    arrays = {c: win[c].to_numpy(float) for c in FEATURES}

    for name, v in arrays.items():
        feats[f"{name}_mean"] = float(v.mean())
        feats[f"{name}_std"] = float(v.std())
        feats[f"{name}_ptp"] = float(np.ptp(v))       # np.ptp: ndarray.ptp() gone in NumPy 2 (§8)
        feats[f"{name}_absmean"] = float(np.abs(v).mean())

    feats["ang_LR_corr"] = _corr(arrays["L_ang_LPF"], arrays["R_ang_LPF"])
    feats["angvel_LR_corr"] = _corr(arrays["L_angvel_LPF"], arrays["R_angvel_LPF"])
    feats["ang_LR_offset"] = float(np.median(arrays["L_ang_LPF"]) - np.median(arrays["R_ang_LPF"]))

    for side in ("L", "R"):
        dom, frac = _spectral(arrays[f"{side}_angvel_LPF"], fs)
        feats[f"{side}_angvel_dom_hz"] = dom
        feats[f"{side}_angvel_band_frac"] = frac

    return feats


def label_window(labels: np.ndarray) -> tuple[object, float, float]:
    """(label, purity, unknown_fraction) for a window.

    `-1` is excluded before voting (§5.2: training-poison, evaluation-gold) but its share
    is reported so a confidence signal can be evaluated against it later. A window that
    is not pure over {stand, walk} is TRANSITION, never a coin-flip majority vote.

    KNOWN AND ACCEPTED (Lu, 2026-08-03 — §5.2): exclusion is per ROW, so `purity` is
    computed over the survivors and a window can be "pure" on a minority of its rows.
    Measured at the 2 s default: 92 of 4,812 trainable windows contain `-1`, 20 are
    over half `-1`, worst is 93.5%. Left in deliberately — 1.9% label noise is inside
    what an RF tolerates, and no threshold on `unknown_frac` has evidence behind it.
    `unknown_frac` rides along on every window so the decision stays checkable.
    """
    unknown_frac = float(np.mean(labels == HUMAN_UNKNOWN))
    valid = labels[np.isin(labels, TRAIN_CLASSES)]
    if valid.size == 0:
        return None, 0.0, unknown_frac
    values, counts = np.unique(valid, return_counts=True)
    purity = float(counts.max() / valid.size)
    winner = int(values[counts.argmax()])
    return (winner if purity >= PURITY_MIN else TRANSITION), purity, unknown_frac


def iter_windows(trial: Trial, spec: WindowSpec) -> Iterator[tuple[dict, pd.DataFrame]]:
    """Yield (metadata, window) for every window of one trial, never spanning a gap.

    THE windowing primitive — S2 features and S3 anchors both consume it, so the two
    stages cannot drift into windowing the same trial differently. Anything keyed on
    `(segment, t_start_ms)` from one stage joins to the other exactly.

    Yields the window unlabeled: what a window *is* does not depend on ground truth,
    and S3 must be able to run on raw recordings that carry none.
    """
    frame = trial.frame
    if frame.empty:
        return
    for seg_id, seg in frame.groupby("segment", sort=True):
        seg = seg.reset_index(drop=True)
        for start in range(0, len(seg) - spec.n + 1, spec.step):
            win = seg.iloc[start:start + spec.n]
            yield ({"rev": trial.rev, "trial": trial.trial, "split": trial.split,
                    "segment": int(seg_id), "t_start_ms": float(win[TIME_COL].iloc[0])},
                   win)


def windows_of_trial(trial: Trial, spec: WindowSpec) -> pd.DataFrame:
    """Every window of one trial, labeled and featurized."""
    rows = []
    for meta, win in iter_windows(trial, spec):
        label, purity, unk = label_window(win[LABEL_COL].to_numpy())
        if label is None:
            continue
        rows.append({**meta, "label": label, "purity": purity, "unknown_frac": unk,
                     **window_features(win, spec.fs_hz)})
    return pd.DataFrame(rows)


def build_windows(trials: list[Trial], spec: WindowSpec | None = None) -> pd.DataFrame:
    spec = spec or WindowSpec()
    frames = [w for t in trials if not (w := windows_of_trial(t, spec)).empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Feature columns only — never the metadata or the target."""
    meta = {"rev", "trial", "split", "segment", "t_start_ms", "label", "purity", "unknown_frac"}
    return [c for c in df.columns if c not in meta]


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
