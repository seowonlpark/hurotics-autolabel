"""The raw -> rev2 feature bridge: reproduce the labeled columns from a raw CSV.

The model trains on the labeled rev2 view (four rotational features) but must RUN on
raw device CSVs. This module is the bridge, and it reproduces HUROTICS' MATLAB
(`LPF.m` / `csv2mat.m`) exactly — verified to ~1e-13 against 19 paired recordings
(DOMAIN_NOTES §6.2). Any drift between this and the training features is silent
train/serve skew, so it is regression-tested against those pairs.

Two things here are easy to get wrong:

  1. **The filter is CAUSAL.** First-order IIR, one pole, fc = 1 Hz. It has phase lag
     by construction. `filtfilt` would be "better" signal processing and WRONG here —
     it would shift the features relative to the labels the model learned.
  2. **The sagittal axis is a DEVICE property, not a signal property.** It is fixed by
     how the IMU sat in that hardware revision, so it is resolved by schema variant,
     never guessed from the waveform. Signal-only detection was measured and scored
     BELOW CHANCE (§6.2) — do not reintroduce it. Unknown variant => abstain.

Sagittality vs the permutation — keep them apart (§4.1b/§6.2). S1 measures the
**permutation** (which Gyro axis is the rate of which Deg axis: X->X, Y->Z, Z->Y); that
is universal across variants and knowable from inside one file. It does NOT say which
plane is sagittal. Sagittality is per hardware revision, unanswerable from inside a
file, and lives here — resolved against paired ground truth. Because the permutation is
universal, the only independent per-variant fact is **which Deg axis is sagittal**; the
gyro axis follows from it, so storing both would invite the two to drift apart.

But "universal" is a corpus finding, not a guarantee, and §4.1b names two files that
break it. So the variant lookup is checked against the file's own measured record
(`check_axis_trust`) before any column is read: the lookup says which axis SHOULD be
sagittal, S1 says whether THIS file obeys the permutation, and a disagreement is fatal.
Until 2026-07-20 this path resolved axes from the documented map alone and never opened
`channel_trust.json` — S1 measured the answer and nothing read it. That is why `trust`
is a required argument with an explicit opt-out rather than an optional one.

`raw_csv_to_features` is the entry point a caller actually runs a file through: it resolves
the variant from the header, opens that file's trust record, runs both guards, and refuses
with a stated reason rather than returning a frame it cannot defend. Until 2026-08-03 no
such entry point existed — the only caller was `verify_transform`, which passes
TRUST_UNCHECKED, so `load_trust` had zero callers and `check_axis_trust` had never once run
against a real record (caveats §3.1). The cost of that gap was not hypothetical: the
sagittal axis for the majority variant was wrong the whole time (see the table below).
"""

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

# csv2mat.m: f_ang = 1; f_angvel = 1;  ("for locomotion classification").
# f_angvel = 10 is the GCP variant and must NOT be used for this task.
FC_ANG_HZ = 1.0
FC_ANGVEL_HZ = 1.0

# variant_id -> the Deg axis that revision's exporter treated as SAGITTAL.
# The matching Gyro axis is NOT stored: it follows from DOCUMENTED_GYRO_PERMUTATION,
# which every variant obeys. Extend only with new evidence: one paired raw+annotated
# file, never a guess.
#
# Re-measured 2026-08-03 against every pair the corpus holds — 18 annotated trials whose
# raw source is present with a bit-identical `Time` vector. All 18, on all three variants,
# reproduce the labeled columns from `Deg_Y` (+ `Gyro_Z`) to <=7.3e-13, float roundoff on a
# ~1e5-sample IIR. The alternatives are nowhere near: `Deg_X` lands 177-381 deg away and
# `Deg_Z` 99-460. Per variant: fb5ea2c2 7 pairs (all rev13), 0fda484e 10 (rev7 + rev8),
# 4bfd6ab2 1 (rev4).
#
# **`fb5ea2c2` read "X" here until 2026-08-03, and it was wrong.** It is the majority
# variant — 62 of 91 raw files — so every serve-path read of those files would have fed the
# classifier the frontal plane instead of the sagittal one, silently, with no downstream
# check able to notice. It survived because nothing ever ran this module against a raw file
# in anger (caveats §3.1) and `verify_transform` was not re-run after the rev13 raw files
# landed; the moment it is run, it fails on exactly these six pairs. See caveats §5.
#
# Note what the corrected table no longer says: with every measured variant reading `Deg_Y`,
# this corpus contains **no** variant whose sagittal plane differs. The per-variant shape is
# kept anyway — sagittality is a hardware-revision property with no in-file signature (§6.2),
# so "all three agree" is a fact about three revisions, not a licence to default an unknown
# one to Y. Unknown variant still abstains.
SAGITTAL_DEG_AXIS_BY_VARIANT = {
    "fb5ea2c2": "Y",   # rev13 / rev14  (majority variant) — 7 pairs, all rev13
    "0fda484e": "Y",   # rev7  / rev8                      — 10 pairs
    "4bfd6ab2": "Y",   # rev4                              — 1 pair
}

FEATURE_COLUMNS = ("L_ang_LPF", "R_ang_LPF", "L_angvel_LPF", "R_angvel_LPF")
TIME_COL = "Time"


class UnknownVariantError(Exception):
    """Raised when a file's schema variant has no measured axis mapping.

    Deliberately fatal rather than defaulted: guessing the axis feeds the classifier a
    channel that is not the one it was trained on, and nothing downstream would notice.
    """


class NotRawDeviceError(Exception):
    """Raised when the file handed to the raw path is not a raw device log.

    The two families share only `Time` (§1.3/config.FAMILY_MARKERS), so reading a rev2
    view through here would resolve columns that do not exist. Named separately from a
    bare KeyError because the remedy is different: this file wants the other path.
    """


class GyroUnitError(Exception):
    """Raised when a file's measured gyro unit is not the one the model was trained on.

    S1 records `scale_to_degps` per side and normalizes the CLEAN layer with it, but this
    path reads the raw CSV, which is un-normalized by construction. Converting here would
    be defensible physics and is still refused: every one of the 90 trust records in this
    corpus measures L and R at exactly 1.0, so a 57.3x rescale has never been checked
    against paired ground truth, and an unverified transform of the classifier's input is
    the same class of silent skew this module exists to prevent. Refuse, state it, and let
    a human extend the verification with the first rad/s pair that appears.
    """


class DegenerateClockError(Exception):
    """Raised when the file's FINAL inter-sample interval cannot carry the filter.

    `matlab_dt` reads that one interval and it sets `alpha` for the whole recording, so a
    bad final tick is not a bad last sample — it is a bad filter everywhere. §6.2 records
    the failure mode: if the last two timestamps tie, `alpha = 0`, the recursion collapses
    to `y[n] = y[n-1]`, and the output freezes flat for the entire trial with nothing
    raised. Measured over all 91 raw files, the final interval is sane on every one
    (0 degenerate, 0 gap-sized), so this guard is a tripwire, not a routine branch.
    """


class AxisConflictError(Exception):
    """Raised when a file's MEASURED permutation contradicts the documented one on the
    axis this path is about to read.

    Same failure as `UnknownVariantError` and the same remedy — refuse rather than
    guess. The variant lookup answers "which axis is sagittal for this hardware"; it
    cannot answer "does this particular file obey the permutation." S1 measures that
    per file, and until now nothing read the answer.
    """


# Explicit opt-out for callers with no per-file trust record to offer. A sentinel
# rather than `None`, so skipping the check is a decision in the call site instead of
# the silent default that let this gap sit open.
TRUST_UNCHECKED = "trust_unchecked"


def alpha(dt_s: float, fc_hz: float) -> float:
    """The MATLAB coefficient: a = 2*pi*dt*fc / (2*pi*dt*fc + 1)."""
    w = 2.0 * math.pi * dt_s * fc_hz
    return w / (w + 1.0)


def lpf(x: np.ndarray, dt_s: float, fc_hz: float) -> np.ndarray:
    """First-order causal IIR, seeded y[0] = x[0]. Exact transcription of LPF.m.

    y[n] = a*x[n] + (1-a)*y[n-1]
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x.copy()
    a = alpha(dt_s, fc_hz)
    # lfilter state form: zi = (1-a)*y[-1]; seeding with x[0] gives y[0] = x[0].
    y, _ = lfilter([a], [1.0, -(1.0 - a)], x, zi=[(1.0 - a) * x[0]])
    return y


def matlab_dt(time_ms: np.ndarray) -> float:
    """The dt their pipeline actually uses: the LAST inter-sample interval, in seconds.

    `timestamp.m` loops the whole time vector but overwrites `del_t` each pass, so only
    the final interval survives — and `csv2mat.m` passes the full vector once, so one dt
    filters the entire trial (§6.2). Reproduced here because matching the training
    features matters more than being correct; see `safe_dt` for the honest version.
    """
    t = np.asarray(time_ms, dtype=float)
    return float(t[-1] - t[-2]) / 1000.0


def safe_dt(time_ms: np.ndarray) -> float:
    """Median interval — what `timestamp.m` should have used.

    Robust to a single bad final interval, which in the MATLAB can be dt=0 and freeze
    the filter output flat for a whole trial. Use this on the canonical grid, where it
    equals the nominal dt anyway.
    """
    t = np.asarray(time_ms, dtype=float)
    return float(np.median(np.diff(t))) / 1000.0


def resolve_axes(variant_id: str) -> tuple[str, str]:
    """(deg_axis, gyro_axis) for a schema variant, or raise. Never guesses.

    Only the Deg axis is stored; the Gyro axis is derived through the permutation S1
    measures, so the two cannot fall out of sync.
    """
    deg_axis = SAGITTAL_DEG_AXIS_BY_VARIANT.get(variant_id)
    if deg_axis is None:
        raise UnknownVariantError(
            f"no measured sagittal-axis mapping for variant {variant_id!r}. "
            f"Known: {sorted(SAGITTAL_DEG_AXIS_BY_VARIANT)}. Signal-based detection "
            f"scores below chance (DOMAIN_NOTES 6.2) — add a mapping from one paired "
            f"raw+annotated recording instead of guessing."
        )
    return deg_axis, DOCUMENTED_GYRO_PERMUTATION[deg_axis]


def trust_path(raw_path: Path, repo_root: Path | None = None) -> Path:
    """Where S1 wrote this raw file's trust record. One definition, so the loader and the
    provenance line can never disagree about which file was consulted."""
    root = repo_root or Path(__file__).resolve().parents[2]
    return (root / "data" / "clean" / raw_path.parent.name /
            f"{raw_path.stem}.channel_trust.json")


def load_trust(raw_path: Path, repo_root: Path | None = None) -> dict:
    """The `channel_trust.json` S1 wrote beside a raw file's clean parquet.

    Missing is an error, not an empty record: a raw file with no trust record was never
    cleaned (quarantined, or S1 has not run), and inventing a permissive default here is
    exactly the "recorded but never read" hole this check exists to close.
    """
    p = trust_path(raw_path, repo_root)
    if not p.exists():
        raise FileNotFoundError(
            f"no channel_trust record at {p} for {raw_path.name}. The file was never "
            f"cleaned (quarantined, or S1 has not run). Run S1, or pass "
            f"TRUST_UNCHECKED if skipping the axis check is genuinely intended."
        )
    return json.loads(p.read_text(encoding="utf-8"))


def check_axis_trust(trust: dict | str, deg_axis: str) -> None:
    """Refuse a file whose measured permutation breaks on the axis we are about to read.

    Only L and R matter — those are the sides `raw_to_features` reads — and only
    `deg_axis`, the one this revision treats as sagittal. A conflict on any other axis
    is real but inert here (`Deg_Z` in particular is the yaw-like axis whose derivative
    is often noise, §4.1b), and an abstaining axis is silent rather than dissenting, so
    it never reaches `conflicts_with_documented` in the first place.

    `trust` is the file's `channel_trust.json`, written beside its clean parquet.
    """
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


def check_gyro_unit(trust: dict | str) -> None:
    """Refuse a file whose gyro is not natively deg/s on the sides this path reads.

    The angle channels are degrees, so the model's `*_angvel_LPF` features are deg/s. A
    rad/s file read through here would hand the classifier angular velocities 57.3x too
    small — well inside the range the trees split on, and invisible downstream. See
    `GyroUnitError` for why this refuses rather than converting.
    """
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
            f"rev2 features the model was trained on come from raw channels that were "
            f"already {CANONICAL_GYRO_UNIT} on every paired recording, so the conversion "
            f"has no ground truth behind it here. Verify it against one paired "
            f"raw+annotated rad/s recording and apply the scale explicitly — do not "
            f"suppress this."
        )


def load_raw_frame(path: Path) -> tuple[pd.DataFrame, str, str]:
    """(frame, variant_id, family) for a raw device CSV, columns resolved BY NAME (§1.3).

    The single reader for both the verification and the serve path. Two copies of this
    drifting apart would mean "verified to 1e-13" describes a read that production never
    performs, which is the same train/serve skew one level up.
    """
    header = read_header(path)
    names = [strip_prefix(c) for c in header]
    df = pd.read_csv(path, encoding="utf-8-sig")
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


def serve_dt(time_ms: np.ndarray) -> tuple[float, float]:
    """(dt in seconds, last/median ratio) for a raw file, or raise.

    `matlab_dt`, not `safe_dt`, and the choice is measured rather than stylistic: the
    labeled features this model trained on were produced by the MATLAB, which filters a
    whole trial with its final interval. Substituting the median moves the features by up
    to **20.1 deg** on the 17 verifiable pairs (worst `rev7_trial_1`, whose last interval
    is 12.0 ms against a 10.0 ms median). That is train/serve skew of the exact kind this
    module exists to prevent, so serve reproduces the upstream quirk deliberately — and
    guards the one case where the quirk is not merely odd but fatal.
    """
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


def raw_to_features(df: pd.DataFrame, variant_id: str, *, trust: dict | str,
                    dt_s: float | None = None,
                    time_col: str = TIME_COL) -> pd.DataFrame:
    """Build the four rev2 features from a name-resolved raw device frame.

    `dt_s` defaults to the median interval; pass `matlab_dt(...)` to reproduce the
    training pipeline bit-for-bit on an un-resampled raw file.

    `trust` is required, not optional: the variant lookup alone cannot tell whether THIS
    file obeys the permutation, and the honest answer S1 already measured is worthless
    if the read path defaults to ignoring it. Pass `TRUST_UNCHECKED` to skip the check
    deliberately.
    """
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


def raw_csv_to_features(path: Path, *, repo_root: Path | None = None,
                        trust: dict | str | None = None
                        ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """(raw rows as read, the four rev2 features, provenance) — or refuse with a reason.

    The serve entry point. Every guard in this module runs here, against a real record,
    which is the whole point: the variant lookup says which axis SHOULD be sagittal, S1's
    per-file record says whether this file obeys the permutation and what unit its gyro is
    in, and the clock check says whether the filter coefficient is meaningful at all. A
    file that fails any of them raises — the caller reports an abstention, it does not
    quietly get a frame.

    Both frames come back because the deliverable is rows-in/rows-out: the caller's own
    rows are returned with columns appended, and re-reading a 132k-row CSV to recover them
    would be the only reason to open the file twice.
    """
    df, variant_id, family = load_raw_frame(path)
    if family != "raw_device":
        raise NotRawDeviceError(
            f"{path.name} is family {family!r}, not 'raw_device'. A rev2 view is already "
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
