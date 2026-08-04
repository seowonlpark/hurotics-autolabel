# regression guard for the raw -> lpf_view bridge
#   python -m stages.s2_ml.verify_transform
# pairs annotated trials with their raw source and asserts the reproduction still lands
# at float roundoff; run it after touching anything in transform.py
# without it, drift is train/serve skew nothing downstream catches, because both sides
# still "look like" angles

from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from stages.s2_ml.dataset import TIME_COL, _read_raw, find_trials
from stages.s2_ml.transform import (
    FEATURE_COLUMNS,
    TRUST_UNCHECKED,
    UnknownVariantError,
    load_raw_frame,
    matlab_dt,
    raw_to_features,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

# Float roundoff on ~1e5-sample IIR recursions; anything materially larger means the
# transform drifted from the MATLAB it is supposed to mirror
TOLERANCE = 1e-9


# every readable raw file, keyed by path; reads through load_raw_frame, the same reader
# serve uses- a private one here would bless a read production never performs
def index_raw_files(raw_glob: str = "data/raw/**/*.csv") -> dict[str, tuple]:
    index = {}
    for f in sorted(glob.glob(str(REPO_ROOT / raw_glob), recursive=True)):
        try:
            df, vid, _family = load_raw_frame(Path(f))
            t = df[TIME_COL].to_numpy(float)
            index[f] = (t[0], t[-1], len(t), df, vid)
        except Exception:
            continue  # unreadable/ragged files are S1's problem, not the bridge's
    return index


# annotated trials whose raw source is present, matched on the time vector
# excluded=set() on purpose: the quarantine is about LABELS and this never reads one, it
# asks whether four columns rebuild from raw; honouring it would drop rev13/4, one of only
# seven pairs behind the majority variant, for an unrelated reason
def find_pairs(index: dict[str, tuple]) -> list[tuple[Path, str, pd.DataFrame, str]]:
    pairs = []
    for p in find_trials(excluded=set()):
        t = _read_raw(p)[TIME_COL].to_numpy(float)
        for f, (r0, r1, nr, df, vid) in index.items():
            if abs(t[0] - r0) < 50 and abs(t[-1] - r1) < 5000 and nr == len(t):
                pairs.append((p, f, df, vid))
                break
    return pairs


def main() -> None:
    pairs = find_pairs(index_raw_files())
    if not pairs:
        raise SystemExit("no paired recordings found — cannot verify the bridge")

    worst, abstained, rows = 0.0, [], []
    for ann_path, _raw_path, raw, vid in pairs:
        ann = pd.read_csv(ann_path)
        ann.columns = [c.strip() for c in ann.columns]
        try:
            # TRUST_UNCHECKED deliberately: this verifies the transform MATH against
            # ground truth, on files whose features the MATLAB already produced; the
            # per-file axis check is a provenance guard for inference on new raw files;
            # applying it here would make a regression test refuse its own fixtures
            feat = raw_to_features(raw, vid, trust=TRUST_UNCHECKED,
                                   dt_s=matlab_dt(raw["Time"].to_numpy(float)))
        except UnknownVariantError:
            abstained.append((ann_path.name, vid))
            continue
        err = max(
            float(np.max(np.abs(feat[c].to_numpy() - ann[c].to_numpy(float))))
            for c in FEATURE_COLUMNS
        )
        rows.append((ann_path.name, vid, err))
        worst = max(worst, err)

    for name, vid, err in sorted(rows, key=lambda r: -r[2]):
        print(f"  {name:<34} {vid:>9}  max_err={err:.3g}")
    for name, vid in abstained:
        print(f"  {name:<34} {vid:>9}  ABSTAINED (no axis mapping)")

    print(f"\n{len(rows)} pairs verified, {len(abstained)} abstained")
    print(f"worst max-abs error: {worst:.3g}  (tolerance {TOLERANCE:g})")
    if worst > TOLERANCE:
        print("FAIL: the bridge no longer reproduces the labeled features.")
        sys.exit(1)
    print("PASS: raw -> lpf_view reproduction is exact to float roundoff.")


if __name__ == "__main__":
    main()
