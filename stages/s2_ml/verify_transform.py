"""Regression guard for the raw -> rev2 bridge.

    python -m stages.s2_ml.verify_transform

The model trains on the labeled rev2 features but runs on raw CSVs. If
`transform.raw_to_features` ever stops reproducing the labeled columns, the classifier
silently sees a different distribution than it learned — train/serve skew that no test
downstream would catch, because both sides would still "look like" angles.

So this pairs annotated trials with their raw source (identical `Time` vector and row
count — the same recording exported twice) and asserts the reproduction still lands at
float roundoff. Run it after touching anything in `transform.py`.
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from stages.s1_clean.census import fingerprint, read_header, strip_prefix
from stages.s2_ml.dataset import TIME_COL, _read_raw, find_trials
from stages.s2_ml.transform import (
    FEATURE_COLUMNS,
    TRUST_UNCHECKED,
    UnknownVariantError,
    matlab_dt,
    raw_to_features,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

# Float roundoff on ~1e5-sample IIR recursions. Anything materially larger means the
# transform drifted from the MATLAB it is supposed to mirror.
TOLERANCE = 1e-9


def load_raw_by_name(path: str) -> pd.DataFrame:
    """Raw device CSV with columns resolved BY NAME (never by position, §1.3)."""
    names = [strip_prefix(c) for c in read_header(Path(path))]
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df.loc[:, [c for c in df.columns if not c.startswith("Unnamed")]]
    df.columns = names[: len(df.columns)]
    return df


def index_raw_files(raw_glob: str = "data/raw/**/*.csv") -> dict[str, tuple]:
    index = {}
    for f in sorted(glob.glob(str(REPO_ROOT / raw_glob), recursive=True)):
        try:
            df = load_raw_by_name(f)
            t = df["Time"].to_numpy(float)
            index[f] = (t[0], t[-1], len(t), df, fingerprint(read_header(Path(f))))
        except Exception:
            continue  # unreadable/ragged files are S1's problem, not the bridge's
    return index


def find_pairs(index: dict[str, tuple]) -> list[tuple[Path, str, pd.DataFrame, str]]:
    """Annotated trials whose raw source is present, matched on the time vector."""
    pairs = []
    for p in find_trials():
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
            # ground truth, on files whose features the MATLAB already produced. The
            # per-file axis check is a provenance guard for inference on new raw files;
            # applying it here would make a regression test refuse its own fixtures.
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
    print("PASS: raw -> rev2 reproduction is exact to float roundoff.")


if __name__ == "__main__":
    main()
