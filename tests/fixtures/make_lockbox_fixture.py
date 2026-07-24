# regenerate the frozen lockbox-window fixture (tests/fixtures/lockbox_windows.csv). freezes the spent
# revs' per-window fusion inputs so the deterministic fusion + scoring layers can be regression-tested
# without refitting the model.

# re-running this re-derives the spent revs, so it is a deliberate act: do it only when a model/feature
# change is meant to move the fixture, and update the asserted numbers in test_lockbox_regression.py.

#   python tests/fixtures/make_lockbox_fixture.py

import sys
from pathlib import Path

# runnable by hand without PYTHONPATH: conftest's path shim only applies under pytest, and this is
# a hand-run generator. put the repo root on the path the same way before importing the pipeline.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stages.s4_fusion.lockbox import lockbox_aligned  # noqa: E402

COLS = ["rev", "trial", "segment", "t_start_ms", "true", "s2_pred", "s2_proba", "s3"]
OUT = Path(__file__).resolve().parent / "lockbox_windows.csv"


def main() -> None:
    df = lockbox_aligned()[COLS].sort_values(["rev", "trial", "segment", "t_start_ms"])
    OUT.write_text(df.to_csv(index=False), encoding="utf-8")
    print(f"wrote {len(df)} windows -> {OUT}")
    print(df.groupby("rev").size().to_string())


if __name__ == "__main__":
    main()
