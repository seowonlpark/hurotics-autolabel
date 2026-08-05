# the `--report` convention: every stage's `.md` is opt-in, its `.json` is not

from __future__ import annotations

import argparse

# Stated once here rather than in nine argparse blocks, so the rule cannot drift between stages.
#
# Each stage writes two things: a `.json` that the next stage and `breakdown` parse, and a `.md`
# that renders the same values for a person. The `.md` is a PURE RENDER -- every render takes
# exactly the values its twin serializes, so a report holds no number its twin does not. That
# makes it the one output safe to not write by default:
#
#   - nothing reads it. No code anywhere parses a `.md` under `runs/`, and gates name the `.json`
#     (`run_pipeline.Step.gate`), so an absent report cannot make a stage look like it never ran.
#   - nothing is lost by it. The numbers are in the twin either way; `--report` re-renders them.
#
# `runs/breakdown.md` is deliberately NOT behind this flag. It is the one page written to be read,
# and it is assembled from every stage's `.json` rather than rendered from one of them -- so it is
# the deliverable, not a redundant second copy of a stage's own output.
REPORT_HELP = ("also write the human-readable .md render beside the .json "
               "(the .json is always written; nothing in the pipeline reads the .md)")


def add_report_flag(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--report", action="store_true", help=REPORT_HELP)
