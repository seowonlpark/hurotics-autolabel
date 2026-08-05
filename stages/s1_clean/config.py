# S1 constants; everything tunable lives here

COLUMN_PREFIX_PATTERN = r"^\d{2,}_"

ROLE_BY_NAME = {
    "Time": "time",
    **{f"{s}_Deg_{a}": "imu_deg" for s in ("L", "R", "B") for a in "XYZ"},
    **{f"{s}_Gyro_{a}": "imu_gyro" for s in ("L", "R", "B") for a in "XYZ"},
    **{f"{s}_Acc_{a}": "imu_acc" for s in ("L", "R", "B") for a in "XYZ"},
    "L LC": "load_cell",
    "R LC": "load_cell",
    "L_GCP": "gait_cycle_pct",
    "R_GCP": "gait_cycle_pct",
    "Hip_Deg_L": "hip_angle",
    "Hip_Deg_R": "hip_angle",
    "Label": "label",

    # the lpf_view four
    "L_ang_LPF": "lpf_angle",
    "R_ang_LPF": "lpf_angle",
    "L_angvel_LPF": "lpf_angvel",
    "R_angvel_LPF": "lpf_angvel",
}

# human ground truth
LABEL_COLUMNS = ("Label",)

# two shapes of file, sharing only Time; contracts are per family
FAMILY_MARKERS = {
    "raw_device": "L_Deg_X",
    "lpf_view": "L_ang_LPF",
}
FAMILY_UNKNOWN = "unknown"

LABEL_UNKNOWN_MACHINE = 255  # data error
LABEL_UNKNOWN_HUMAN = -1     # a human looked and couldn't call it

TIME_UNIT_MS = 1.0  # Time is device uptime, ms
GAP_FACTOR = 5.0    # dt > this * median => gap
PLAUSIBLE_HZ = (1.0, 2000.0)  # outside: suspect the unit, not the device

CANONICAL_HZ = 100.0
CANONICAL_DT_MS = 1000.0 / CANONICAL_HZ

RATE_TOLERANCE = 0.05
MIN_SEGMENT_S = 1.0  # shorter is recorded, not used

DECIMATE_FILTER = "fir"

NEAREST_ROLES = ("label",)

SIDES = ("L", "R")
GYRO_AXES = "XYZ"
RAD2DEG = 57.29577951308232

CANONICAL_GYRO_UNIT = "deg/s"  # angles are degrees, so slope~1
UNIT_TO_DEGPS_SCALE = {"deg/s": 1.0, "rad/s": RAD2DEG}

# fixed device permutation, Y<->Z; NOT sagittality- that is transform.SAGITTAL_DEG_AXIS_BY_VARIANT
DOCUMENTED_GYRO_PERMUTATION = {"X": "X", "Y": "Z", "Z": "Y"}

# below this the axis is too static to fit; applied per deg axis, not per side
TRUST_R_FLOOR = 0.9

# abstention fallback only, never the first answer
DOCUMENTED_GYRO_UNIT = {"L": "deg/s", "R": "deg/s"}

YAW_DRIFT_R_FLOOR = 0.9        # |corr(Deg, Time)| >= this => contaminated
DRIFT_MIN_SEGMENT_S = 5.0      # shorter and the drift means nothing

KEEP_MEASURED = (
    "Time",
    *[f"{s}_{k}_{a}" for s in ("L", "R") for k in ("Deg", "Gyro") for a in "XYZ"],
)

# empty IS the finding: canonical == measured, no caveat
KEEP_EXCEPTIONS: dict[str, str] = {}

# ground truth travels with the data
KEEP_IF_PRESENT = ("Label",)

# `loco` and `L/R_Ref_Force` excluded on purpose; recoverable by name

# data/raw/<YYYYMMDD[_n]>/- session date comes from the folder
SESSION_DIR_PATTERN = r"^(\d{8})(?:_(\d+))?$"
