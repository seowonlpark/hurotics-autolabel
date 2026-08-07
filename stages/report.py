# the `--report` convention: every stage's `.md` is opt-in, its `.json` is not

from __future__ import annotations

import argparse

# stated once, not in nine argparse blocks- the `.md` is a PURE RENDER, so nothing is lost by skipping it
REPORT_HELP = ("also write the human-readable .md render beside the .json "
               "(the .json is always written; nothing in the pipeline reads the .md)")


def add_report_flag(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--report", action="store_true", help=REPORT_HELP)
