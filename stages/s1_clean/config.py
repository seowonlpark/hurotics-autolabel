"""S1 constants. Everything tunable lives here, nothing is buried in logic."""

# --- Column naming -----------------------------------------------------------
# Raw headers carry a positional prefix ("47_loco"). The prefix is a per-file
# position, NOT a stable id: column 47 is `loco` in 62 files and `Step` in 11.
# Resolution is always by name. This regex strips the prefix and nothing else.
COLUMN_PREFIX_PATTERN = r"^\d{2,}_"

# --- Channel roles -----------------------------------------------------------
# Name -> role. Absence is recorded as a fact, never an error: presence is
# per-file, not per-corpus.
ROLE_BY_NAME = {
    "Time": "time",
    # Thigh IMUs (L/R) — the trusted angle source per DOMAIN_NOTES 1.2
    **{f"{s}_Deg_{a}": "imu_deg" for s in ("L", "R", "B") for a in "XYZ"},
    # Gyro = angular velocity. DOMAIN_NOTES 1.1: untrusted as provided.
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

# --- Session ---------------------------------------------------------------
# data/raw/<YYYYMMDD[_n]>/<file>.csv — session date comes from the folder.
SESSION_DIR_PATTERN = r"^(\d{8})(?:_(\d+))?$"
