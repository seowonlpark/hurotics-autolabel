# S4 results export: one CSV per labeled trial, written to results/ under the same name,
# carrying the original rev2 columns plus the fused per-window (label, confidence) mapped
# densely back onto every row. read-only over the labeled corpus and the S4 fused table.
# rows with no fused window -- transitions, dropped/short segments, and the lockbox +
# excluded revs (rev8/rev13/rev14) -- carry blank fused columns, not a guess. see PLAN S4.

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from stages.s2_ml.dataset import LABELED_DIR, TIME_COL, rev_of, trial_of
from stages.s2_ml.features import DEFAULT_WINDOW_S
from stages.s4_fusion.run import FUSED_CSV, S4_OUT_DIR, run as run_s4

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"

# the S2 OOF windows the fusion joins on are the default fixed length; a window owns the
# original rows whose Time falls in [t_start, t_start + WINDOW) on the same device clock
WINDOW_MS = DEFAULT_WINDOW_S * 1000.0
FUSED_COLS = ("fused_label", "confidence")


# every labeled trial csv on disk -- the full mirror, lockbox + excluded revs included
# (they simply carry no fused prediction). glob, not the S2 loader, so nothing is filtered
def all_labeled_trials(labeled_dir: Path = LABELED_DIR) -> list[Path]:
    return sorted(labeled_dir.rglob("annotated_loco_*_trial_*.csv"))


# attach fused_label + confidence to each original row via the window that covers its Time.
# windows = this trial's slice of the fused table (may be empty). backward as-of match to the
# window that starts at/before the row, kept only if the row falls inside that window's span
def annotate(orig: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    out = orig.copy()
    for c in FUSED_COLS:
        out[c] = pd.NA
    if windows.empty:
        return out
    w = windows[["t_start_ms", *FUSED_COLS]].sort_values("t_start_ms").reset_index(drop=True)
    left = out.reset_index()[["index", TIME_COL]].sort_values(TIME_COL)
    m = pd.merge_asof(left, w, left_on=TIME_COL, right_on="t_start_ms", direction="backward")
    m = m[m[TIME_COL] < m["t_start_ms"] + WINDOW_MS].set_index("index")
    for c in FUSED_COLS:
        out.loc[m.index, c] = m[c].to_numpy()
    return out


# write every labeled trial to results/<same name>.csv with the fused columns appended.
# returns (files_written, rows_with_a_fused_label). regenerates the fused table if absent.
def export(out_dir: Path = RESULTS_DIR, s4_dir: Path = S4_OUT_DIR) -> tuple[int, int]:
    fused_path = s4_dir / FUSED_CSV
    if not fused_path.exists():
        run_s4(s4_dir)
    fused = pd.read_csv(fused_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    written, labeled_rows = 0, 0
    for p in all_labeled_trials():
        orig = pd.read_csv(p)
        orig.columns = [c.strip() for c in orig.columns] # some trials carry stray whitespace
        w = fused[(fused["rev"] == rev_of(p)) & (fused["trial"] == trial_of(p))]
        annotated = annotate(orig, w)
        annotated.to_csv(out_dir / p.name, index=False)
        written += 1
        labeled_rows += int(annotated["fused_label"].notna().sum())
    return written, labeled_rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Export labeled trials with fused predictions to results/.")
    ap.add_argument("--out", type=Path, default=RESULTS_DIR)
    args = ap.parse_args()
    written, labeled_rows = export(args.out)
    print(f"[s4] exported {written} labeled trials -> {args.out} "
          f"({labeled_rows:,} rows carry a fused label; the rest are transitions, dropped, "
          f"or lockbox/excluded revs)")


if __name__ == "__main__":
    main()
