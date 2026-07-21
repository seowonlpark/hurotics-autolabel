# per-file calibration: the raw->rev2 signals recentred/rescaled onto the corpus-GLOBAL
# reference, harvested LABEL-FREE from each file's own signal. §7 "per-file, never
# corpus-wide" (a fixed threshold scores walk_rec 0.000 on rev8, per-file scores 1.000);
# §10.2 files begin at rest, giving a same-person/sensor standing reference minutes earlier.
#
# two corrections, both a delta FROM global (add/multiply the global anchor back, so the
# current global features stay the target distribution -- for a tree the anchor is a uniform
# shift/scale = cosmetic; the PER-FILE part is what moves the metric):
#   posture  -- subtract each file's resting `ang` median (from the at-rest opening). fixes
#               mounting/posture offset (§10.1 baseline -3.3 deg). needs the opening at rest.
#   amplitude-- divide each file's `angvel` by its own robust spread over the WHOLE file.
#               the at-rest opening is ~0, so the scale can't come from rest; whole-file MAD
#               is still label-free. fixes the gait-amplitude spread (gait-RMS 1080 vs 298).
#
# LORO/lockbox safe: the per-file part reads only that file's own signal, so it cannot cross
# a fold; the global anchor is built from TRAIN-split files only, so lockbox frames are never
# read here. NEVER reads the Label column (asserted). no silent mutation: every file's path
# is returned as a record for calibration.jsonl. default-off in ExperimentSpec.

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from stages.s1_clean.config import CANONICAL_HZ
from stages.s2_ml.dataset import LABEL_COL, TIME_COL, Trial

ANG_CHANNELS = ("L_ang_LPF", "R_ang_LPF")       # posture: recentred
ANGVEL_CHANNELS = ("L_angvel_LPF", "R_angvel_LPF")  # amplitude: rescaled

MODES = ("posture", "amplitude", "posture_amplitude")

DEFAULT_BASELINE_S = 2.0     # opening rest window = one model window (features DEFAULT_WINDOW_S)
DEFAULT_QUIET_RATIO = 0.35   # opening is "at rest" if its angvel spread <= this * whole-file spread
DEFAULT_SCALE_FLOOR = 0.20   # a file's angvel scale must be >= this * global scale, else amplitude falls back


# a challenger's calibration knobs; default-off is expressed by ExperimentSpec.calibrate=None,
# not here -- if this config exists, calibration runs
@dataclass(frozen=True)
class CalibrationConfig:
    mode: str = "posture_amplitude"
    baseline_s: float = DEFAULT_BASELINE_S
    quiet_ratio: float = DEFAULT_QUIET_RATIO
    scale_floor: float = DEFAULT_SCALE_FLOOR
    fs_hz: float = CANONICAL_HZ

    @property
    def do_posture(self) -> bool:
        return "posture" in self.mode

    @property
    def do_amplitude(self) -> bool:
        return "amplitude" in self.mode

    @property
    def n_rest(self) -> int:
        return int(round(self.baseline_s * self.fs_hz))


# one file's own reference, all label-free
@dataclass
class FileReference:
    posture: dict   # ang channel -> resting median over the at-rest opening
    scale: dict     # angvel channel -> robust spread (MAD*1.4826) over the whole file
    n_rest: int     # samples in the opening window
    quiet: bool     # the opening is genuinely at rest (posture is trustworthy)


# the corpus anchor, built from train-split files only
@dataclass
class GlobalReference:
    posture: dict
    scale: dict

    def to_dict(self) -> dict:
        return {"posture": self.posture, "scale": self.scale}


# median absolute deviation scaled to a std-equivalent; 0 on empty
def _mad(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return 0.0
    return float(np.median(np.abs(x - np.median(x))) * 1.4826)


# mean per-channel std of the angvel channels over `frame` -- the amplitude of motion
def _angvel_spread(frame: pd.DataFrame) -> float:
    stds = [float(frame[ch].to_numpy(float).std()) for ch in ANGVEL_CHANNELS if ch in frame]
    return float(np.mean(stds)) if stds else 0.0


# the opening of the first usable segment -- the at-rest reference (§10.2), never labels
def _opening(frame: pd.DataFrame, cfg: CalibrationConfig) -> pd.DataFrame:
    first_seg = frame["segment"].iloc[0]  # frame is time-ordered, segments ascending
    seg = frame[frame["segment"] == first_seg]
    return seg.iloc[: min(cfg.n_rest, len(seg))]


# one file's label-free reference: resting posture from the opening, amplitude from the whole file
def file_reference(frame: pd.DataFrame, cfg: CalibrationConfig) -> FileReference:
    opening = _opening(frame, cfg)
    posture = {ch: float(np.median(opening[ch].to_numpy(float))) for ch in ANG_CHANNELS}
    scale = {ch: _mad(frame[ch].to_numpy(float)) for ch in ANGVEL_CHANNELS}

    file_spread = _angvel_spread(frame)
    open_spread = _angvel_spread(opening)
    quiet = file_spread > 0 and open_spread <= cfg.quiet_ratio * file_spread
    return FileReference(posture, scale, len(opening), quiet)


# corpus anchor from a set of per-file references (train-split only). posture from the files
# that were genuinely at rest (an unreliable posture must not pollute the anchor); scale from
# the non-degenerate files
def global_reference(refs: list[FileReference]) -> GlobalReference:
    posture = {}
    for ch in ANG_CHANNELS:
        vals = [r.posture[ch] for r in refs if r.quiet] or [r.posture[ch] for r in refs]
        posture[ch] = float(np.median(vals)) if vals else 0.0
    scale = {}
    for ch in ANGVEL_CHANNELS:
        vals = [r.scale[ch] for r in refs if r.scale[ch] > 0]
        scale[ch] = float(np.median(vals)) if vals else 1.0
    return GlobalReference(posture, scale)


# apply the calibration to one file's frame, returning (new_frame, record). the record names
# every path taken so nothing is silent. asserts the label column is untouched
def apply_calibration(frame: pd.DataFrame, ref: FileReference, gref: GlobalReference,
                      cfg: CalibrationConfig) -> tuple[pd.DataFrame, dict]:
    out = frame.copy()

    posture_source = "off"
    if cfg.do_posture:
        if ref.quiet:
            for ch in ANG_CHANNELS:
                out[ch] = out[ch] - ref.posture[ch] + gref.posture[ch]
            posture_source = "calibrated"
        else:
            posture_source = "fallback_not_quiet"  # opening not at rest -> leave posture as-is

    amplitude_source = "off"
    scale_ratio = {}
    if cfg.do_amplitude:
        applied = 0
        for ch in ANGVEL_CHANNELS:
            fs, gs = ref.scale[ch], gref.scale[ch]
            if fs > 0 and fs >= cfg.scale_floor * gs:
                out[ch] = out[ch] * (gs / fs)
                scale_ratio[ch] = gs / fs
                applied += 1
            else:
                scale_ratio[ch] = None  # degenerate (near-static file) -> leave as-is
        amplitude_source = ("calibrated" if applied == len(ANGVEL_CHANNELS)
                            else "partial" if applied else "fallback_degenerate")

    assert out[LABEL_COL].equals(frame[LABEL_COL]), "calibration mutated the label column"
    assert out[TIME_COL].equals(frame[TIME_COL]), "calibration mutated the time column"

    record = {
        "posture_source": posture_source, "amplitude_source": amplitude_source,
        "n_rest": ref.n_rest, "quiet": ref.quiet,
        "file_posture": ref.posture, "file_scale": ref.scale, "scale_ratio": scale_ratio,
    }
    return out, record


# calibrate a whole trial list. lockbox trials are left untouched (they are filtered out of
# training anyway, and this keeps their frames physically unread until the lockbox opens).
# returns (new_trials, global_ref, per-file records)
def calibrate_trials(trials: list[Trial],
                     cfg: CalibrationConfig) -> tuple[list[Trial], GlobalReference, list[dict]]:
    refs = {(t.rev, t.trial): file_reference(t.frame, cfg)
            for t in trials if t.split != "lockbox"}
    gref = global_reference(list(refs.values()))

    new_trials, records = [], []
    for t in trials:
        if t.split == "lockbox":
            new_trials.append(t)  # sealed: not calibrated, not even read
            continue
        ref = refs[(t.rev, t.trial)]
        new_frame, rec = apply_calibration(t.frame, ref, gref, cfg)
        new_trials.append(replace(t, frame=new_frame))
        records.append({"rev": t.rev, "trial": t.trial, "path": t.path,
                        "split": t.split, **rec})
    return new_trials, gref, records


# a compact, ledger-friendly summary of a calibration pass
def summarize(mode: str, gref: GlobalReference, records: list[dict]) -> dict:
    def count(key, value):
        return sum(1 for r in records if r[key] == value)
    return {
        "mode": mode,
        "global_reference": gref.to_dict(),
        "n_files": len(records),
        "posture": {"calibrated": count("posture_source", "calibrated"),
                    "fallback_not_quiet": count("posture_source", "fallback_not_quiet")},
        "amplitude": {"calibrated": count("amplitude_source", "calibrated"),
                      "partial": count("amplitude_source", "partial"),
                      "fallback_degenerate": count("amplitude_source", "fallback_degenerate")},
        "files": records,
    }
