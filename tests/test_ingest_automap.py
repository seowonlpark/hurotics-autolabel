# the agent auto-mapper PROPOSES a column mapping; code VALIDATES it through the same gate a
# hand-written mapping passes. these stub the agent call (no API key, no network) and lock the
# contract that matters: a good proposal is accepted and persisted, a wrong one is rejected with
# the same loud error a bad hand-mapping raises -- the agent can never push through a bad mapping.

import pandas as pd
import pytest

from dataset_profile import HUMAN_UNKNOWN, LABEL_COL, STAND, TIME_COL, WALK
from ingest.adapt import LABELED, RAW, AdaptError
from ingest import automap as A


def _sheet() -> pd.DataFrame:
    return pd.DataFrame({
        "timestamp_ms": [0.0, 10.0, 20.0, 30.0],
        "left_knee_angle": [1.0, 2.0, 3.0, 4.0],
        "right_knee_angle": [1.0, 2.0, 3.0, 4.0],
        "left_knee_vel": [0.1, 0.2, 0.3, 0.4],
        "right_knee_vel": [0.1, 0.2, 0.3, 0.4],
        "activity": ["standing", "walking", "walking", "?"],
    })


_GOOD = {
    "columns": {
        TIME_COL: "timestamp_ms", "L_ang_LPF": "left_knee_angle", "R_ang_LPF": "right_knee_angle",
        "L_angvel_LPF": "left_knee_vel", "R_angvel_LPF": "right_knee_vel", LABEL_COL: "activity",
    },
    "label_values": {"standing": STAND, "walking": WALK, "?": HUMAN_UNKNOWN},
}


# replace the agent call with a canned reply so the test is deterministic and offline
def _stub_propose(mapping, raw="(stub reply)"):
    async def fake(df, target, run_dir, model=A.MODEL_SMART):
        return mapping, raw
    return fake


def test_good_proposal_is_accepted_and_persisted(tmp_path, monkeypatch):
    monkeypatch.setattr(A, "propose", _stub_propose(_GOOD))
    run_dir = tmp_path / "run"
    mapping = A.automap(_sheet(), LABELED, run_dir)
    assert mapping == _GOOD
    assert (run_dir / "proposed_mapping.json").exists()      # the validated mapping is saved
    assert (run_dir / "proposed_mapping_raw.txt").exists()   # and the raw reply, for audit


def test_wrong_proposal_is_rejected_by_the_gate(tmp_path, monkeypatch):
    bad = {**_GOOD, "label_values": {"standing": STAND, "walking": WALK}}  # forgot "?"
    monkeypatch.setattr(A, "propose", _stub_propose(bad))
    with pytest.raises(AdaptError, match=r"\?"):
        A.automap(_sheet(), LABELED, tmp_path / "run")


def test_mispointed_proposal_is_rejected_by_the_gate(tmp_path, monkeypatch):
    bad = {"columns": {TIME_COL: "does_not_exist"}}
    monkeypatch.setattr(A, "propose", _stub_propose(bad))
    with pytest.raises(AdaptError, match="not in the sheet"):
        A.automap(_sheet(), LABELED, tmp_path / "run")


def test_unparseable_reply_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(A, "propose", _stub_propose(None, raw="sorry, I could not decide"))
    run_dir = tmp_path / "run"
    with pytest.raises(AdaptError, match="did not contain a JSON mapping"):
        A.automap(_sheet(), LABELED, run_dir)
    assert (run_dir / "proposed_mapping_raw.txt").read_text() == "sorry, I could not decide"


def test_prompt_reflects_target_label_presence():
    assert "HAS a Label" in A.build_prompt(_sheet(), LABELED)
    assert "NO Label" in A.build_prompt(_sheet(), RAW)


def test_agent_uses_no_tools_and_no_domain_notes():
    spec = A._agent("haiku")
    assert spec.allowed_tools == [] and spec.domain_sections == ()
