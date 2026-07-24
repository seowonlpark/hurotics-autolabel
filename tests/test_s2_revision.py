# the S2 critic's 'revise' verdict is acted on: the orchestrator feeds the reasons back for one bounded
# retry, writing each attempt to its own file. these tests pin the pure pieces that make that safe (the
# attempt-filename scheme and the revision prompt block) without touching the API.

from agents import s2_critic, s2_experimenter
from orchestrator import MAX_PROPOSE_ATTEMPTS, _attempt_name


def test_first_attempt_keeps_canonical_name():
    # backward compatible: attempt 1 writes proposal.json / critic_review.json unchanged
    assert _attempt_name(s2_experimenter.PROPOSAL_FILENAME, 1) == "proposal.json"
    assert _attempt_name(s2_critic.REVIEW_FILENAME, 1) == "critic_review.json"


def test_revision_attempt_gets_distinct_suffix():
    # a revision writes beside the first attempt, never over it
    assert _attempt_name(s2_experimenter.PROPOSAL_FILENAME, 2) == "proposal_rev1.json"
    assert _attempt_name(s2_critic.REVIEW_FILENAME, 2) == "critic_review_rev1.json"


def test_attempt_names_are_unique_per_attempt():
    names = {_attempt_name("proposal.json", a) for a in range(1, MAX_PROPOSE_ATTEMPTS + 1)}
    assert len(names) == MAX_PROPOSE_ATTEMPTS


def test_revision_block_carries_prior_spec_and_reasons():
    prev = {"name": "drop_jerk", "rationale": "jerk features add noise"}
    reasons = ["window_s is out of range", "name collides with ledger entry drop_jerk_v1"]
    block = s2_experimenter.revision_block(prev, reasons)
    # the experimenter must see exactly what to fix: the prior spec and every reason
    assert "drop_jerk" in block
    assert "jerk features add noise" in block
    for r in reasons:
        assert r in block
    # it must be framed as a fix, not a fresh start, or the retry loses the idea
    assert "same core idea" in block.lower()


def test_write_proposal_honours_name(tmp_path):
    p = s2_experimenter.write_proposal(tmp_path, {"name": "x"}, "raw", name="proposal_rev1.json")
    assert p.name == "proposal_rev1.json" and p.exists()


def test_write_review_honours_name(tmp_path):
    p = s2_critic.write_review(tmp_path, {"verdict": "approve"}, "raw", name="critic_review_rev1.json")
    assert p.name == "critic_review_rev1.json" and p.exists()
