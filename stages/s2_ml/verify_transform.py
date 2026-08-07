# regression guard for the raw -> lpf_view bridge; python -m stages.s2_ml.verify_transform

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from stages.s2_ml.corpus import load_raw, pairs as manifest_pairs
from stages.s2_ml.transform import (
    FEATURE_COLUMNS,
    TRUST_UNCHECKED,
    UnknownVariantError,
    matlab_dt,
    raw_to_features,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

# float roundoff on ~1e5-sample IIR recursions; materially larger means drift from the MATLAB
TOLERANCE = 1e-9


# the pairings the manifest DECLARES; include_excluded stays on- no label is read here (§1.6)
def find_pairs() -> list:
    return manifest_pairs(include_excluded=True)


def main() -> None:
    pairs = find_pairs()
    if not pairs:
        raise SystemExit("no paired recordings found — cannot verify the bridge")

    worst, abstained, rows = 0.0, [], []
    for entry in pairs:
        ann_path = entry.annotated
        raw, vid = load_raw(entry)
        ann = pd.read_csv(ann_path)
        ann.columns = [c.strip() for c in ann.columns]
        try:
            # TRUST_UNCHECKED deliberately: this verifies the MATH, and the axis check refuses fixtures
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
