"""S1 constants. Everything tunable lives here, nothing is buried in logic."""

# --- Column naming -----------------------------------------------------------
# Raw headers carry a positional prefix ("47_loco"). The prefix is a per-file
# position, NOT a stable id: at the same index one variant reads `loco`, another
# reads `Step`. Resolution is always by name. This regex strips the prefix, nothing else.
COLUMN_PREFIX_PATTERN = r"^\d{2,}_"

# --- Channel roles -----------------------------------------------------------
# Name -> role. Absence is recorded as a fact, never an error: presence is
# per-file, not per-corpus.
ROLE_BY_NAME = {
    "Time": "time",
    # Thigh/trunk IMUs (L/R/B) — the trusted angle source per DOMAIN_NOTES 4 / 6.2
    **{f"{s}_Deg_{a}": "imu_deg" for s in ("L", "R", "B") for a in "XYZ"},
    # Gyro = angular velocity, and it is reliable: Gyro == d(Deg)/dt (DOMAIN_NOTES
    # 4.1). But units differ per side and axes are swapped — normalized in the clean
    # layer, detected per file. See the "Gyro trust / normalization" section below.
    **{f"{s}_Gyro_{a}": "imu_gyro" for s in ("L", "R", "B") for a in "XYZ"},
    # Accelerometer = gravity reference. Required for axis/calibration checks.
    **{f"{s}_Acc_{a}": "imu_acc" for s in ("L", "R", "B") for a in "XYZ"},
    "L LC": "load_cell",
    "R LC": "load_cell",
    "L_GCP": "gait_cycle_pct",
    "R_GCP": "gait_cycle_pct",
    "Hip_Deg_L": "hip_angle",
    "Hip_Deg_R": "hip_angle",
    "Label": "label",
    # rev2 derived view
    "L_ang_LPF": "rev2_angle",
    "R_ang_LPF": "rev2_angle",
    "L_angvel_LPF": "rev2_angvel",
    "R_angvel_LPF": "rev2_angvel",
}

# Human ground-truth annotation.
LABEL_COLUMNS = ("Label",)

# The corpus holds two different products. "Stable prefix across everything" is a
# meaningless question: raw device logs and the rev2 derived view share only Time.
# Family is decided by a marker column, and contracts are per family.
FAMILY_MARKERS = {
    "raw_device": "L_Deg_X",
    "rev2_view": "L_ang_LPF",
}
FAMILY_UNKNOWN = "unknown"

# Outdated rule-based algorithm output. Recorded for provenance, never a feature,
# never ground truth.
LEGACY_ALGO_COLUMNS = ("loco",)

# --- Label encoding ----------------------------------------------------------
# Two unknowns, opposite in kind. Never merge them.
LABEL_UNKNOWN_MACHINE = 255  # data error: nothing was measured properly
LABEL_UNKNOWN_HUMAN = -1     # valid annotation: a human looked and could not call it

# --- Time / rate -------------------------------------------------------------
TIME_UNIT_MS = 1.0  # Time column is a device uptime counter in milliseconds
GAP_FACTOR = 5.0    # dt > GAP_FACTOR * median(dt) counts as a gap
PLAUSIBLE_HZ = (1.0, 2000.0)  # outside this, suspect the unit, not the device

# --- Resampling --------------------------------------------------------------
# Golden data arrives at 100 Hz, so 100 Hz is the canonical grid: ~100 Hz files
# already sit there, 500 Hz files are decimated down, quantized clocks are
# grid-corrected. Nothing is upsampled and no bandwidth is invented.
CANONICAL_HZ = 100.0
CANONICAL_DT_MS = 1000.0 / CANONICAL_HZ

# Measured rates are grouped into families within this relative tolerance. The
# ~100 Hz family spans 99.38-100.0 because the device clock counts in binary
# sub-ms ticks: dt lands on 10 + 2^-k ms (1/16, 1/32, 1/256). Those are the SAME
# 100 Hz device, quantized differently — not different acquisition rates.
RATE_TOLERANCE = 0.05

# Gaps fall anywhere, unpredictably. The segment — a continuous run between gaps —
# is therefore the unit of analysis, not the file. Nothing is ever resampled
# across a gap: that would invent data that was never measured.
#
# The floor is a DURATION, not a sample count. It was 100 source rows, described as
# "1 s at canonical rate" — true only in the ~100 Hz era. At 500 Hz the same constant
# meant 0.2 s, so the rule silently loosened by 5x exactly where the data is densest:
# `annotated_loco_rev6_trial_3` carried a 238-row / 500 Hz segment (0.48 s) through as
# usable, into data/clean and into clean_report.md's usable minutes, despite being too
# short to yield even one analysis window. A threshold whose meaning depends on the
# rate era is a rate confound in the segment filter (§2.1).
MIN_SEGMENT_S = 1.0  # shorter runs are recorded, not used

# Anti-aliasing is not optional when downsampling. Dropping every 5th sample folds
# everything above 50 Hz into the passband — heel-strike transients and rig
# vibration would masquerade as low-frequency gait signal.
DECIMATE_FILTER = "fir"

# Interpolation policy by role. Categorical channels must never be averaged. Only
# Label reaches the resampler — legacy/computed columns are pruned before it runs —
# so it is the only nearest-interpolated role in practice.
NEAREST_ROLES = ("label",)

# --- Gyro trust / normalization ----------------------------------------------
# DOMAIN_NOTES 4.1b: gyro is reliable (Gyro == d(Deg)/dt) but its UNITS and AXES
# are inconsistent within one file — B_Gyro is rad/s, L/R_Gyro is deg/s (a silent
# 57.3x), and d(Deg_Y)/dt tracks Gyro_Z, not Gyro_Y. That is a measurement
# property, so it is corrected here, in the clean layer, before any feature runs.
#
# It is DETECTED per file, never asserted from the table below: for each side we
# regress d(Deg_A)/dt (deg/s) against every Gyro axis, for every Deg axis A. The
# strongest-correlated gyro axis is A's counterpart (recovering the full
# permutation); the regression slope reveals the unit (~1 -> already deg/s,
# ~1/57.3 -> rad/s). This doubles as the channel-trust check PLAN S1 requires, and
# it catches the files that break the corpus-wide rule (e.g. some B/trunk channels
# map Y->Y, which is not a permutation at all).
SIDES = ("L", "R", "B")
GYRO_AXES = "XYZ"
RAD2DEG = 57.29577951308232

CANONICAL_GYRO_UNIT = "deg/s"  # angle channels are degrees, so deg/s keeps slope~1
UNIT_TO_DEGPS_SCALE = {"deg/s": 1.0, "rad/s": RAD2DEG}

# The Deg->Gyro axis correspondence is a FIXED PERMUTATION of the device
# (DOMAIN_NOTES 4.1b): the gyro labels carry a Y<->Z transposition relative to the
# angle labels, X maps to itself. Measured across every variant that answers --
# fb5ea2c2, 0fda484e, 4bfd6ab2 all identical, no firmware dependence.
#
# NOTE THIS IS NOT SAGITTALITY. This map says which Gyro axis measures the rate of
# which Deg axis. It does NOT say which axis is the sagittal (flexion) plane -- that
# is a per-hardware-revision fact, unanswerable from inside a file, and it lives in
# SAGITTAL_AXIS_BY_VARIANT (stages/s2_ml/transform.py), resolved against paired
# ground truth. Keep the two questions apart: this layer can measure the permutation
# and cannot measure sagittality, so it asserts only the former.
DOCUMENTED_GYRO_PERMUTATION = {"X": "X", "Y": "Z", "Z": "Y"}

# Below this |r| the axis is too static for d(Deg_A)/dt to carry signal — the fit is
# noise (DOMAIN_NOTES 11.1, "density needs mass"). We do NOT assert from a motionless
# fit: that axis abstains and is recorded as unresolved.
#
# Applied PER DEG AXIS, not once per side. Deg_Z is the yaw-like axis, which drifts
# rather than oscillates (4.2), so its derivative is often noise even mid-walk;
# scoring the side as a whole would launder that noise into a confident anomaly.
TRUST_R_FLOOR = 0.9

# The documented convention, used only as the abstention fallback — never as the
# first answer. Measured across the corpus (DOMAIN_NOTES 4.1b).
DOCUMENTED_GYRO_UNIT = {"L": "deg/s", "R": "deg/s", "B": "rad/s"}

# --- Yaw / drift trust -------------------------------------------------------
# DOMAIN_NOTES 4.2: a channel whose value tracks session TIME is measuring elapsed
# time (integration drift), not orientation. Measured on the raw corpus, Deg_Z is
# the yaw-like axis (median |corr(Deg,Time)| ~0.33 vs sagittal Y ~0.08), but drift
# is FILE-SPECIFIC — strong on only a few files — so it is flagged per channel, per
# file, never dropped wholesale. This is a feature-time exclusion signal, not a
# quarantine trigger: the raw superset is kept.
YAW_DRIFT_R_FLOOR = 0.9        # |corr(Deg, Time)| at/above this => drift-contaminated
DRIFT_MIN_SEGMENT_S = 5.0      # a segment must span this long for its drift to mean anything

# --- What the canonical file keeps -------------------------------------------
# The line is MEASURED vs COMPUTED, not useful vs useless.
#
# The device *measures* IMU channels and load cells. It *computes* Cadence,
# Stride Length, Hip_ROM, GCP, Adaptability, admittance, PID state — those are
# the firmware's opinion, not observation, and belong in the same bucket as
# `loco`. There is nothing to trust-check in a number the firmware derived.
#
# Two consequences beyond storage:
#   1. Every column that churns position between variants (Step 46/47, Cadence
#      45/46, the whole 83/79/91 tail) is a COMPUTED one. Dropping them collapses
#      the schema variants into a single canonical shape.
#   2. Computed values depend on firmware version, so training on them partly
#      learns which firmware produced the file. That is the era confound baked
#      straight into the feature set.
KEEP_MEASURED = (
    "Time",
    *[f"{s}_{k}_{a}" for s in ("L", "R", "B") for k in ("Deg", "Gyro", "Acc") for a in "XYZ"],
    "L LC",
    "R LC",
)

# Documented exceptions to the measured-only rule. Each needs a reason.
# Currently EMPTY, and that is the finding, not an oversight (DOMAIN_NOTES 9).
#
# `Hip_Deg_L`/`Hip_Deg_R` lived here as a "bridge to the open-source gait dataset's
# Hip_Flex_L/R". Removed 2026-07-20 — the exception failed every test it implied:
#   - REDUNDANT: corr with same-side Deg_Y is 0.991 (vs ~0.02-0.21 on X/Z). It is
#     the sagittal angle already kept, re-zeroed.
#   - NOT a clean function of it: slope ~0.96-0.99 with a per-file offset of -75
#     to -88 deg and max residual 4-152 deg. The unmodeled remainder IS the
#     firmware's zeroing convention -- precisely the firmware-version signal the
#     measured-only rule exists to strip.
#   - DEAD on 12 of 180 (file, side) pairs: zero-variance, and twice frozen at a
#     NONZERO constant, which no `!= 0` guard would catch.
#   - The bridge had no far side: the open dataset is not in the repo (5.7), and
#     nothing downstream ever read the column.
# Recoverable by name from data/raw if that dataset ever lands -- same standing as
# `loco` and `L/R_Ref_Force` below. The mechanism stays; the entry is gone.
KEEP_EXCEPTIONS: dict[str, str] = {}

# Kept when present: human ground truth travels with the data.
KEEP_IF_PRESENT = ("Label",)

# `loco` (outdated algorithm output) and `L/R_Ref_Force` (controller setpoints,
# commanded not measured) are deliberately excluded. Both remain recoverable from
# data/raw by name if a benchmark against the old algorithm is ever wanted.

# Parquet, not CSV. Storage cost is a file-format problem, not a column-count
# problem: ~5-10x smaller, ~10x faster to read, dtypes preserved.
CLEAN_FORMAT = "parquet"

# --- Session ---------------------------------------------------------------
# data/raw/<YYYYMMDD[_n]>/<file>.csv — session date comes from the folder.
SESSION_DIR_PATTERN = r"^(\d{8})(?:_(\d+))?$"
