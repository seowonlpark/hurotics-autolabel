# S1 constants -- everything tunable lives here, nothing buried in logic

# the label codes are defined once in the top-level dataset_profile (the data contract) and
# aliased below under S1's names, so the raw layer and the labeled layer share one definition.
from dataset_profile import HUMAN_UNKNOWN as LABEL_UNKNOWN_HUMAN
from dataset_profile import MACHINE_UNKNOWN as LABEL_UNKNOWN_MACHINE

# column naming
# raw headers carry a positional prefix ("47_loco"), NOT a stable id -- same index is
# `loco` at that index in some files, `Step` in others. resolve by name. this regex strips it
COLUMN_PREFIX_PATTERN = r"^\d{2,}_"

# channel roles
# name -> role; absence is a fact, not an error (presence is per-file, not per-corpus)
ROLE_BY_NAME = {
    "Time": "time",
    # thigh/trunk IMUs (L/R/B) -- the trusted angle source (Section 4/6.2)
    **{f"{s}_Deg_{a}": "imu_deg" for s in ("L", "R", "B") for a in "XYZ"},
    # gyro = angular velocity, reliable (Gyro == d(Deg)/dt, Section 4.1); units/axes differ
    # per file, normalized in the clean layer (see "gyro trust" below)
    **{f"{s}_Gyro_{a}": "imu_gyro" for s in ("L", "R", "B") for a in "XYZ"},
    # accelerometer = gravity reference, needed for axis/calibration checks
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

# human ground-truth annotation
LABEL_COLUMNS = ("Label",)

# the corpus holds two products; contracts are per family, decided by a marker column
# (raw device logs and the rev2 view share only Time)
FAMILY_MARKERS = {
    "raw_device": "L_Deg_X",
    "rev2_view": "L_ang_LPF",
}
FAMILY_UNKNOWN = "unknown"

# outdated rule-based algorithm output; provenance only, never a feature or ground truth
LEGACY_ALGO_COLUMNS = ("loco",)

# label encoding
# two unknowns, opposite in kind -- never merge them. LABEL_UNKNOWN_HUMAN (-1) and
# LABEL_UNKNOWN_MACHINE (255) are imported from dataset_profile at the top of this file.

# time / rate
TIME_UNIT_MS = 1.0 # Time column is a device uptime counter in ms
GAP_FACTOR = 5.0 # dt > GAP_FACTOR * median(dt) counts as a gap
PLAUSIBLE_HZ = (1.0, 2000.0) # outside this, suspect the unit, not the device

# resampling
# golden data is 100 Hz, so that's the canonical grid: ~100 Hz files sit there, 500 Hz
# is decimated down, quantized clocks are grid-corrected; nothing is upsampled
CANONICAL_HZ = 100.0
CANONICAL_DT_MS = 1000.0 / CANONICAL_HZ

# measured rates grouped into families within this tolerance; the ~100 Hz family spans
# 99.38-100.0 because the clock quantizes dt to 10 + 2^-k ms -- same device (Section 2.5)
RATE_TOLERANCE = 0.05

# gaps fall anywhere, so the segment (a gap-free run) is the unit, not the file;
# nothing is ever resampled across a gap
MIN_SEGMENT_SAMPLES = 100 # 1 s at canonical rate; shorter runs recorded, not used

# anti-aliasing is not optional: naive [::5] folds >50 Hz into the gait band (Section 2.5)
DECIMATE_FILTER = "fir"

# interpolation by role; only Label reaches the resampler (computed cols are pruned first),
# so it is the only nearest-interpolated role in practice
NEAREST_ROLES = ("label",)

# gyro trust / normalization
# gyro is reliable but its units/axes are inconsistent within a file (Section 4.1b): B is
# rad/s, L/R is deg/s, and d(Deg_Y)/dt tracks Gyro_Z not Gyro_Y. detected per file
# in channel_trust.py, never asserted from the tables below. see DOMAIN_NOTES.
SIDES = ("L", "R", "B")
GYRO_AXES = "XYZ"
RAD2DEG = 57.29577951308232

CANONICAL_GYRO_UNIT = "deg/s" # angle channels are degrees, so deg/s keeps slope~1
UNIT_TO_DEGPS_SCALE = {"deg/s": 1.0, "rad/s": RAD2DEG}

# the Deg->Gyro axis map is a fixed device permutation (Y<->Z, X to itself, Section 4.1b).
# NOT sagittality -- which axis is sagittal has no in-file signature; transform.py fixes
# it to the Y plane for every file (Section 6.3). used only as the abstention fallback.
DOCUMENTED_GYRO_PERMUTATION = {"X": "X", "Y": "Z", "Z": "Y"}

# below this |r| the axis is too static to read; it abstains (Section 11.1). applied per Deg
# axis, not once per side -- Deg_Z drifts and its derivative is often noise (Section 4.2)
TRUST_R_FLOOR = 0.9

# documented convention, used only as the abstention fallback, never the first answer (Section 4.1b)
DOCUMENTED_GYRO_UNIT = {"L": "deg/s", "R": "deg/s", "B": "rad/s"}

# yaw / drift trust
# a Deg channel whose value tracks session time is measuring integration drift, not
# orientation (Section 4.2); flagged per channel per file, never dropped -- the raw superset stays
YAW_DRIFT_R_FLOOR = 0.9 # |corr(Deg, Time)| at/above this => drift-contaminated
DRIFT_MIN_SEGMENT_S = 5.0 # a segment must span this long for its drift to mean anything

# what the canonical file keeps
# the line is MEASURED vs COMPUTED. the device measures IMU + load cells; it computes
# Cadence, GCP, admittance, PID state etc -- firmware opinion, same bucket as `loco`.
# dropping computed cols also collapses every header shape into one canonical form and strips
# the firmware-era confound (every position-churning column is a computed one). see Section 9.
KEEP_MEASURED = (
    "Time",
    *[f"{s}_{k}_{a}" for s in ("L", "R", "B") for k in ("Deg", "Gyro", "Acc") for a in "XYZ"],
    "L LC",
    "R LC",
)

# documented exceptions to the measured-only rule; currently EMPTY, and that's the
# finding, not an oversight (Section 9). Hip_Deg_L/R was cut 2026-07-20 -- 0.991 redundant with
# Deg_Y, its residual only the firmware's zeroing convention. see README / DOMAIN_NOTES Section 9.
KEEP_EXCEPTIONS: dict[str, str] = {}

# kept when present, but ONLY for files under data/labeled: human ground truth travels with
# the data. a Label column in a raw device log is not trusted ground truth -- it is dropped
# there. keeping is by provenance, not by presence (the raw corpus carries no Label today).
KEEP_IF_PRESENT = ("Label",)

# path segment that marks a labeled-source file; KEEP_IF_PRESENT is honored only under it
LABELED_ROOT = "labeled"

# `loco` and `L/R_Ref_Force` (commanded, not measured) are deliberately excluded,
# recoverable by name from data/raw if ever needed.

# parquet, not CSV: ~5-10x smaller, ~10x faster to read, dtypes preserved
CLEAN_FORMAT = "parquet"

# session
# data/raw/<YYYYMMDD[_n]>/<file>.csv -- session date comes from the folder
SESSION_DIR_PATTERN = r"^(\d{8})(?:_(\d+))?$"
