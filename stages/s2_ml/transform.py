# the raw -> lpf_view bridge: rebuild the labeled columns from a raw CSV
# reproduces their MATLAB to ~1e-13; any drift is silent train/serve skew
# the filter is CAUSAL- filtfilt is better signal processing and WRONG here
# the sagittal axis is a DEVICE property, resolved by variant, never guessed
# sagittality is not the permutation; unknown variant ABSTAINS

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from stages.s1_clean.census import family_of, fingerprint, read_header, strip_prefix
from stages.s1_clean.config import (
    CANONICAL_GYRO_UNIT,
    DOCUMENTED_GYRO_PERMUTATION,
    GAP_FACTOR,
)

# csv2mat.m: f_ang = 1; f_angvel = 1;  ("for locomotion classification")
# f_angvel = 10 is the GCP variant and must NOT be used for this task
FC_ANG_HZ = 1.0
FC_ANGVEL_HZ = 1.0

# variant_id -> the Deg axis that revision's exporter treated as SAGITTAL
# the gyro axis is NOT stored, it follows from DOCUMENTED_GYRO_PERMUTATION
# extend only with new evidence: one paired raw+annotated file, never a guess
#
# re-measured 2026-08-03 on all 18 pairs; every one reproduces from Deg_Y to <=7.3e-13,
# and the alternatives are nowhere near (Deg_X lands 177-381 deg away, Deg_Z 99-460)
#
# fb5ea2c2 read "X" here until 2026-08-03 and it was WRONG; it is 62 of 91 raw files, so
# every serve read fed the classifier the frontal plane instead of the sagittal one, with
# nothing downstream able to notice; it survived because nothing ever ran this module
# against a raw file, and it was found the day the serve path was built and read one
#
# all three measured variants read Deg_Y, but the per-variant shape stays: three
# revisions agreeing is not a licence to default a fourth; unknown variant ABSTAINS
SAGITTAL_DEG_AXIS_BY_VARIANT = {
    "fb5ea2c2": "Y",   # rev13 / rev14  (majority variant), 7 pairs, all rev13
    "0fda484e": "Y",   # rev7  / rev8, 10 pairs
    "4bfd6ab2": "Y",   # rev4, 1 pair
}

FEATURE_COLUMNS = ("L_ang_LPF", "R_ang_LPF", "L_angvel_LPF", "R_angvel_LPF")
TIME_COL = "Time"


# no measured axis mapping for this variant; fatal rather than defaulted, since a guessed
# axis feeds the classifier a channel it was not trained on and nothing would notice
class UnknownVariantError(Exception):
    pass


# an lpf_view file came down the raw path; the families share only `Time`, so reading it
# here resolves columns that do not exist; separate from KeyError because the remedy is
# different- this file wants the other path
class NotRawDeviceError(Exception):
    pass


# gyro is not natively deg/s; converting would be defensible physics and is still refused,
# because all 91 trust records measure 1.0 so a 57.3x rescale has no ground truth behind it
class GyroUnitError(Exception):
    pass


# the FINAL interval cannot carry the filter, and matlab_dt sets alpha for the whole
# recording from it- so a bad last tick is a bad filter everywhere, not a bad last sample
# (tie the last two stamps and the output freezes flat with nothing raised)
class DegenerateClockError(Exception):
    pass


# this file's MEASURED permutation contradicts the documented one on the axis about to be
# read; the variant lookup cannot answer whether THIS file obeys the permutation, S1 can
class AxisConflictError(Exception):
    pass


# a sentinel, not `None`, so skipping the check is a decision at the call site rather than
# the silent default that let this gap sit open
TRUST_UNCHECKED = "trust_unchecked"


# the MATLAB coefficient: a = 2*pi*dt*fc / (2*pi*dt*fc + 1)
def alpha(dt_s: float, fc_hz: float) -> float:
    w = 2.0 * math.pi * dt_s * fc_hz
    return w / (w + 1.0)


# y[n] = a*x[n] + (1-a)*y[n-1], seeded y[0] = x[0]; exact transcription of LPF.m
def lpf(x: np.ndarray, dt_s: float, fc_hz: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x.copy()
    a = alpha(dt_s, fc_hz)
    # lfilter state form: zi = (1-a)*y[-1]; seeding with x[0] gives y[0] = x[0]
    y, _ = lfilter([a], [1.0, -(1.0 - a)], x, zi=[(1.0 - a) * x[0]])
    return y


# the dt their pipeline actually uses: the LAST interval, in seconds; timestamp.m
# overwrites del_t each pass so only the final one survives, and it filters the whole
# trial; reproduced because matching the training features beats being correct
def matlab_dt(time_ms: np.ndarray) -> float:
    t = np.asarray(time_ms, dtype=float)
    return float(t[-1] - t[-2]) / 1000.0


# median interval- what timestamp.m should have used; robust to one bad final tick,
# which in the MATLAB can be dt=0 and freeze the output flat for a whole trial
def safe_dt(time_ms: np.ndarray) -> float:
    t = np.asarray(time_ms, dtype=float)
    return float(np.median(np.diff(t))) / 1000.0


# (deg_axis, gyro_axis) or raise; never guesses; the gyro axis is derived through the
# permutation, so the two cannot fall out of sync
def resolve_axes(variant_id: str) -> tuple[str, str]:
    deg_axis = SAGITTAL_DEG_AXIS_BY_VARIANT.get(variant_id)
    if deg_axis is None:
        raise UnknownVariantError(
            f"no measured sagittal-axis mapping for variant {variant_id!r}. "
            f"Known: {sorted(SAGITTAL_DEG_AXIS_BY_VARIANT)}. Signal-based detection "
            f"scores below chance (DOMAIN_NOTES 6.2) — add a mapping from one paired "
            f"raw+annotated recording instead of guessing."
        )
    return deg_axis, DOCUMENTED_GYRO_PERMUTATION[deg_axis]


# one definition, so the loader and the provenance line cannot disagree about which
# file was consulted
def trust_path(raw_path: Path, repo_root: Path | None = None) -> Path:
    root = repo_root or Path(__file__).resolve().parents[2]
    return (root / "data" / "clean" / raw_path.parent.name /
            f"{raw_path.stem}.channel_trust.json")


# missing is an ERROR, not an empty record: no trust record means the file was never
# cleaned, and a permissive default here is the exact hole this check exists to close
def load_trust(raw_path: Path, repo_root: Path | None = None) -> dict:
    p = trust_path(raw_path, repo_root)
    if not p.exists():
        raise FileNotFoundError(
            f"no channel_trust record at {p} for {raw_path.name}. The file was never "
            f"cleaned (quarantined, or S1 has not run). Run S1, or pass "
            f"TRUST_UNCHECKED if skipping the axis check is genuinely intended."
        )
    return json.loads(p.read_text(encoding="utf-8"))


# refuse a file whose measured permutation breaks on the axis about to be read; only L/R
# and only deg_axis matter, a conflict elsewhere is real but inert here
def check_axis_trust(trust: dict | str, deg_axis: str) -> None:
    if trust is TRUST_UNCHECKED:
        return
    if not isinstance(trust, dict):
        raise TypeError(
            f"trust must be a channel_trust record or TRUST_UNCHECKED, got {type(trust)}"
        )
    broken = {}
    for side in ("L", "R"):
        rec = trust.get("sides", {}).get(side)
        if rec and deg_axis in rec.get("conflicts_with_documented", []):
            broken[side] = rec.get("gyro_axis_by_deg_axis", {}).get(deg_axis)
    if broken:
        detail = ", ".join(f"{s}_Deg_{deg_axis} -> {s}_Gyro_{g}" for s, g in broken.items())
        raise AxisConflictError(
            f"this file's measured permutation breaks on the sagittal axis: {detail}, "
            f"but the documented map says Deg_{deg_axis} -> "
            f"Gyro_{DOCUMENTED_GYRO_PERMUTATION[deg_axis]}. Reading the documented "
            f"column would feed the classifier a channel it was not trained on, and "
            f"nothing downstream would notice. Resolve the file by hand (DOMAIN_NOTES "
            f"§4.1b) — do not suppress this."
        )


# a rad/s file read through here hands the classifier velocities 57.3x too small- well
# inside the range the trees split on, and invisible downstream
def check_gyro_unit(trust: dict | str) -> None:
    if trust is TRUST_UNCHECKED:
        return
    if not isinstance(trust, dict):
        raise TypeError(
            f"trust must be a channel_trust record or TRUST_UNCHECKED, got {type(trust)}"
        )
    off = {}
    for side in ("L", "R"):
        rec = trust.get("sides", {}).get(side)
        if rec and float(rec.get("scale_to_degps", 1.0)) != 1.0:
            off[side] = (rec.get("unit"), float(rec["scale_to_degps"]), rec.get("method"))
    if off:
        detail = ", ".join(f"{s}_Gyro measured {u} (x{sc}, {m})" for s, (u, sc, m) in off.items())
        raise GyroUnitError(
            f"gyro is not natively {CANONICAL_GYRO_UNIT} on this file: {detail}. The four "
            f"lpf_view features the model was trained on come from raw channels that were "
            f"already {CANONICAL_GYRO_UNIT} on every paired recording, so the conversion "
            f"has no ground truth behind it here. Verify it against one paired "
            f"raw+annotated rad/s recording and apply the scale explicitly — do not "
            f"suppress this."
        )


# (frame, variant_id, family), columns resolved BY NAME; the single reader for both the
# verification and the serve path- two copies drifting apart would mean "verified to
# 1e-13" describes a read production never performs
def load_raw_frame(path: Path) -> tuple[pd.DataFrame, str, str]:
    header = read_header(path)
    names = [strip_prefix(c) for c in header]
    # index_col=False: without it a trailing comma (data one field wider than the header) makes
    # pandas promote column 0 to the index, shifting every column left by one; the count check
    # below cannot see that- the shift leaves the count correct- so it must not happen at all
    df = pd.read_csv(path, encoding="utf-8-sig", index_col=False)
    df = df.loc[:, [c for c in df.columns if not c.startswith("Unnamed")]]
    if len(df.columns) > len(names):
        raise NotRawDeviceError(
            f"{path.name}: {len(df.columns)} data columns but only {len(names)} header "
            f"names — ragged file, resolve it by hand rather than by truncation (§1.3)."
        )
    df.columns = names[: len(df.columns)]
    dupes = sorted({n for n in df.columns if list(df.columns).count(n) > 1})
    if dupes:
        raise NotRawDeviceError(
            f"{path.name}: duplicated column names after prefix stripping: {dupes}. "
            f"`df[name]` would return a frame, not a column, so every read below is "
            f"ambiguous. Resolve by hand."
        )
    return df, fingerprint(header), family_of(list(df.columns))


# (dt seconds, last/median ratio) or raise; matlab_dt NOT safe_dt, and that is measured
# rather than stylistic- the median moves the features up to 20.1 deg on the paired
# recordings, which is the train/serve skew this module exists to prevent
def serve_dt(time_ms: np.ndarray) -> tuple[float, float]:
    t = np.asarray(time_ms, dtype=float)
    if t.size < 2:
        raise DegenerateClockError("fewer than 2 samples: no inter-sample interval exists")
    d = np.diff(t)
    last, med = float(d[-1]), float(np.median(d))
    if med <= 0:
        raise DegenerateClockError(
            f"median interval {med} ms <= 0 — duplicate or backward timestamps (§2.6). "
            f"S1 rejects this file wholesale; so does this path."
        )
    if last <= 0:
        raise DegenerateClockError(
            f"final interval {last} ms <= 0. `matlab_dt` reads that one interval and it "
            f"sets alpha for the WHOLE recording, so the filter would freeze flat and "
            f"return a constant with nothing raised (§6.2)."
        )
    if last > GAP_FACTOR * med:
        raise DegenerateClockError(
            f"final interval {last:.4f} ms is a gap by S1's own rule "
            f"(> {GAP_FACTOR}x the {med:.4f} ms median). One gap-sized dt would filter the "
            f"entire recording at the wrong cutoff."
        )
    return matlab_dt(t), last / med


# the four lpf_view features from a name-resolved raw frame; `trust` is required, not
# optional- the variant lookup cannot tell whether THIS file obeys the permutation
def raw_to_features(df: pd.DataFrame, variant_id: str, *, trust: dict | str,
                    dt_s: float | None = None,
                    time_col: str = TIME_COL) -> pd.DataFrame:
    # dt_s defaults to the median; pass matlab_dt() to reproduce training bit-for-bit
    deg_axis, gyro_axis = resolve_axes(variant_id)
    check_axis_trust(trust, deg_axis)
    if dt_s is None:
        dt_s = safe_dt(df[time_col].to_numpy(float))

    out = {}
    for side in ("L", "R"):
        deg_col, gyro_col = f"{side}_Deg_{deg_axis}", f"{side}_Gyro_{gyro_axis}"
        for col in (deg_col, gyro_col):
            if col not in df.columns:
                raise KeyError(f"raw frame is missing {col!r} (variant {variant_id})")
        out[f"{side}_ang_LPF"] = lpf(df[deg_col].to_numpy(float), dt_s, FC_ANG_HZ)
        out[f"{side}_angvel_LPF"] = lpf(df[gyro_col].to_numpy(float), dt_s, FC_ANGVEL_HZ)

    return pd.DataFrame({time_col: df[time_col].to_numpy(float), **out})


# the serve entry point: (raw rows, features, provenance) or refuse with a reason
# every guard in this module runs HERE against a real record, which is the whole point;
# a file that fails any of them raises, it does not quietly get a frame
# both frames come back because the deliverable is rows-in/rows-out
def raw_csv_to_features(path: Path, *, repo_root: Path | None = None,
                        trust: dict | str | None = None
                        ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    df, variant_id, family = load_raw_frame(path)
    if family != "raw_device":
        raise NotRawDeviceError(
            f"{path.name} is family {family!r}, not 'raw_device'. An lpf_view file is already "
            f"the model's input and needs no bridge; anything else is unrecognized."
        )
    if trust is None:
        trust = load_trust(path, repo_root)
    check_gyro_unit(trust)
    deg_axis, gyro_axis = resolve_axes(variant_id)   # raises on an unknown variant
    dt_s, dt_ratio = serve_dt(df[TIME_COL].to_numpy(float))

    feat = raw_to_features(df, variant_id, trust=trust, dt_s=dt_s)  # runs check_axis_trust

    sides = {} if trust is TRUST_UNCHECKED else trust.get("sides", {})
    provenance = {
        "source": str(path),
        "family": family,
        "variant_id": variant_id,
        "sagittal_deg_axis": deg_axis,
        "gyro_axis": gyro_axis,
        "columns_read": [f"{s}_{k}_{a}" for s in ("L", "R")
                         for k, a in (("Deg", deg_axis), ("Gyro", gyro_axis))],
        "n_rows": int(len(df)),
        "dt_ms": round(dt_s * 1000.0, 6),
        "dt_source": "matlab_dt (final interval, as the training features were built)",
        "dt_last_over_median": round(dt_ratio, 6),
        "trust_record": (TRUST_UNCHECKED if trust is TRUST_UNCHECKED
                         else str(trust_path(path, repo_root))),
        "axis_check": {s: sides.get(s, {}).get("method") for s in ("L", "R")},
        "gyro_unit": {s: sides.get(s, {}).get("unit") for s in ("L", "R")},
    }
    return df, feat, provenance
