# the ingest adapter maps an arbitrary sheet into the labeled-trial contract. these lock the two
# guarantees that matter: a correct mapping produces a file the pipeline's own loader accepts,
# and a wrong mapping fails loudly (never a silently mislabeled trial) rather than writing garbage.

import pandas as pd
import pytest

from dataset_profile import FEATURES, HUMAN_UNKNOWN, LABEL_COL, STAND, TIME_COL, WALK
from ingest.adapt import (
    RAW, AdaptError, apply_mapping, infer_rev, infer_session, infer_trial, raw_path,
    scaffold, write_output,
)

CONTRACT = (TIME_COL, *FEATURES, LABEL_COL)


# a sheet with foreign column names and word labels, like a real one someone would bring in
def _foreign_sheet() -> pd.DataFrame:
    return pd.DataFrame({
        "timestamp_ms": [0.0, 10.0, 20.0, 30.0],
        "left_knee_angle": [1.0, 2.0, 3.0, 4.0],
        "right_knee_angle": [1.0, 2.0, 3.0, 4.0],
        "left_knee_vel": [0.1, 0.2, 0.3, 0.4],
        "right_knee_vel": [0.1, 0.2, 0.3, 0.4],
        "activity": ["standing", "walking", "walking", "?"],
    })


_MAP = {
    "columns": {
        TIME_COL: "timestamp_ms",
        "L_ang_LPF": "left_knee_angle",
        "R_ang_LPF": "right_knee_angle",
        "L_angvel_LPF": "left_knee_vel",
        "R_angvel_LPF": "right_knee_vel",
        LABEL_COL: "activity",
    },
    "label_values": {"standing": STAND, "walking": WALK, "?": HUMAN_UNKNOWN},
}


def test_mapping_yields_the_contract_columns_and_codes():
    out = apply_mapping(_foreign_sheet(), _MAP)
    assert list(out.columns) == list(CONTRACT)
    assert list(out[LABEL_COL]) == [STAND, WALK, WALK, HUMAN_UNKNOWN]


def test_written_trial_passes_the_pipelines_own_loader(tmp_path):
    from ingest.adapt import write_trial
    out = apply_mapping(_foreign_sheet(), _MAP)
    dest = write_trial(out, "rev3", 2, tmp_path)  # write_trial re-reads via dataset._read_raw
    assert dest.name == "annotated_loco_rev3_trial_2.csv"
    assert dest.parent.name == "rev3"


def test_unmapped_label_value_fails_loud():
    partial = {**_MAP, "label_values": {"standing": STAND, "walking": WALK}}  # no "?"
    with pytest.raises(AdaptError, match=r"\?"):
        apply_mapping(_foreign_sheet(), partial)


def test_label_code_outside_the_allowed_set_fails_loud():
    bad = {**_MAP, "label_values": {"standing": STAND, "walking": WALK, "?": 7}}
    with pytest.raises(AdaptError, match="allowed set"):
        apply_mapping(_foreign_sheet(), bad)


def test_mispointed_column_names_available_columns():
    with pytest.raises(AdaptError, match="not in the sheet"):
        apply_mapping(_foreign_sheet(), {"columns": {TIME_COL: "nope"}})


def test_absent_mapping_assumes_canonical_names_already_present():
    # a sheet that already uses the canonical names needs no column mapping at all
    canonical = _foreign_sheet().rename(columns={
        "timestamp_ms": TIME_COL, "left_knee_angle": "L_ang_LPF",
        "right_knee_angle": "R_ang_LPF", "left_knee_vel": "L_angvel_LPF",
        "right_knee_vel": "R_angvel_LPF", "activity": LABEL_COL})
    canonical[LABEL_COL] = [STAND, WALK, WALK, HUMAN_UNKNOWN]
    out = apply_mapping(canonical, {})  # no columns, no label_values
    assert list(out.columns) == list(CONTRACT)


def test_rev_and_trial_read_from_filename():
    assert infer_rev("annotated_loco_rev7_trial_3.csv") == "rev7"
    assert infer_trial("annotated_loco_rev7_trial_3.csv") == 3


def test_scaffold_lists_available_columns_and_label_values():
    sc = scaffold(_foreign_sheet())  # default target is labeled
    assert "timestamp_ms" in sc["_available_columns"]
    assert set(sc["columns"]) == set(CONTRACT)
    assert "label_values" in sc  # labeled target draws out the distinct label words to encode


# raw target: an unlabeled recording for scoring, validated through the feature bridge

def _foreign_raw_sheet() -> pd.DataFrame:
    n = 8
    return pd.DataFrame({
        "t_ms": [i * 10.0 for i in range(n)],
        "left_angle_deg": [float(i) for i in range(n)],
        "right_angle_deg": [float(n - i) for i in range(n)],
        "left_rate_dps": [0.5] * n,
        "right_rate_dps": [-0.5] * n,
    })


_RAW_MAP = {"columns": {
    TIME_COL: "t_ms",
    "L_Deg_Y": "left_angle_deg", "R_Deg_Y": "right_angle_deg",
    "L_Gyro_Z": "left_rate_dps", "R_Gyro_Z": "right_rate_dps",
}}


def test_raw_mapping_passes_the_feature_bridge():
    # apply_mapping runs transform.raw_to_features as its contract check, so a passing result
    # means the pipeline's verified bridge can turn this file into model features
    out = apply_mapping(_foreign_raw_sheet(), _RAW_MAP, RAW)
    assert list(out.columns) == list(RAW.columns)


def test_raw_write_lands_in_session_folder(tmp_path):
    out = apply_mapping(_foreign_raw_sheet(), _RAW_MAP, RAW)
    dest = write_output(out, raw_path(tmp_path, "20260114", "rec1"), RAW)
    assert dest.parent.name == "20260114" and dest.name == "rec1.csv"


def test_raw_missing_angle_column_fails_loud():
    with pytest.raises(AdaptError, match="not in the sheet"):
        apply_mapping(_foreign_raw_sheet(), {"columns": {TIME_COL: "t_ms"}}, RAW)


def test_infer_session_from_filename_or_folder():
    assert infer_session("walk_20260114.csv") == "20260114"
    assert infer_session("rec.csv", "20260114") == "20260114"
    assert infer_session("rec.csv", "notadate") is None
