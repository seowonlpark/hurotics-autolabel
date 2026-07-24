# collapse() is the ONE place the S1 agent's two orthogonal judgements (explained x action) fold
# into a disposition -- so two runs can't disposition the same item differently. an unrecognized
# or missing verdict must escalate to needs_human, never silently pass. this is the whole matrix.

import pytest

from agents.s1_exception import collapse


@pytest.mark.parametrize("explained,action,expected", [
    ("yes", "none", ("known_expected", "none")),
    ("yes", "human", ("needs_human", "human")),
    ("no", "human", ("novel", "human")),
    ("no", "none", ("novel", "none")),
    ("no", None, ("novel", "human")),          # unusable action defaults to human
    ("contradicts", "none", ("novel", "human")),  # contradiction forces human regardless
    ("contradicts", "human", ("novel", "human")),
    ("garbage", "none", ("needs_human", "human")),  # unrecognized -> escalate
    (None, None, ("needs_human", "human")),
    ("yes", "garbage", ("needs_human", "human")),   # yes but unusable action -> escalate
])
def test_collapse_matrix(explained, action, expected):
    assert collapse(explained, action) == expected
