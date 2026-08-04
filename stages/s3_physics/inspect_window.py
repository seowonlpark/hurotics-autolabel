# print the raw interleg trace around one suspect window, for a human to adjudicate
#   python -m stages.s3_physics.inspect_window rev6 6 --t 522.2
# a READER: it computes no verdict and writes nothing

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from stages.s2_ml.dataset import (
    DEFAULT_LOCKBOX_REVS,
    EXCLUDED_TRIALS,
    FEATURES,
    LABEL_COL,
    LABELED_DIR,
    TIME_COL,
    assign_split,
    find_trials,
    load_trial,
    rev_of,
    trial_of,
)
from stages.console import use_replacement_encoding
from stages.s2_ml.features import rest_reference
from stages.s2_ml.rest import SWAP_DELTA_DEG
from stages.s1_clean.config import CANONICAL_HZ

CLASS_NAME = {0: "stand", 10: "walk", -1: "human_unknown", 255: "machine_unknown"}


# index of every commit past +delta or -delta, as the swap count sees them
def crossings(d: np.ndarray, delta: float = SWAP_DELTA_DEG) -> np.ndarray:
    out, state = [], 0
    for i, v in enumerate(d):
        if v > delta and state != 1:
            out.append(i)
            state = 1
        elif v < -delta and state != -1:
            out.append(i)
            state = -1
    return np.asarray(out, dtype=int)


def inspect(rev: str, trial_no: int, t_center_s: float, span_s: float = 12.0,
            every_ms: int = 100) -> None:
    # `excluded=set()`, and it is load-bearing rather than defensive; `find_trials` applies
    # `EXCLUDED_TRIALS` by default, so with the default this tool could not open the one
    # trial in the corpus that has been excluded- while the entry in `EXCLUDED_TRIALS`
    # asks for exactly the internal, model-free evidence this module prints, and re-checking
    # an old exclusion is as legitimate as justifying a new one; a READER that hides the
    # quarantine cannot audit the quarantine
    paths = [p for p in find_trials(LABELED_DIR, excluded=set())
             if rev_of(p) == rev and trial_of(p) == trial_no]
    if not paths:
        raise SystemExit(f"no labeled trial for {rev} trial {trial_no}")
    if (rev, trial_no) in EXCLUDED_TRIALS:
        print(f"NOTE: {rev} trial {trial_no} is in dataset.EXCLUDED_TRIALS — it trains "
              f"nothing and is scored nowhere. You are reading it to check that call.\n")
    # The trial's real split, not a made-up one: `Trial.split` is a three-value field and
    # inventing a fourth value here would make the record lie about a lockbox trial
    trial = load_trial(paths[0], assign_split(rev, DEFAULT_LOCKBOX_REVS, ()))
    frame = trial.frame.reset_index(drop=True)

    # The SAME rest zero S2 and S3 use; measuring a fresh one here would adjudicate the
    # window against an origin neither stage used
    _zeros, center, trusted = rest_reference(frame, CANONICAL_HZ)

    t = frame[TIME_COL].to_numpy(float) / 1000.0
    lo, hi = t_center_s - span_s / 2, t_center_s + span_s / 2
    m = (t >= lo) & (t <= hi)
    if not m.any():
        raise SystemExit(f"{rev}/t{trial_no} has no samples in [{lo:.1f}, {hi:.1f}] s")

    sub = frame[m].reset_index(drop=True)
    ts = t[m]
    l = sub[FEATURES[0]].to_numpy(float)
    r = sub[FEATURES[1]].to_numpy(float)
    d = (l - r) - center
    xs = crossings(d)

    labels = pd.to_numeric(sub[LABEL_COL], errors="coerce")
    print(f"{rev} trial {trial_no}  t=[{ts[0]:.2f}, {ts[-1]:.2f}] s  "
          f"n={len(sub):,}  rest_center={center:+.3f} deg "
          f"({'trusted' if trusted else 'FALLBACK — treat the verdict as soft'})")
    print(f"swap commits in span: {len(xs)}  "
          f"(the rule needs >=2 within one span to say WALKING)")
    seg = sub["segment"].unique() if "segment" in sub else []
    print(f"segments present: {list(seg)}   labels present: "
          f"{[CLASS_NAME.get(int(v), v) for v in sorted(labels.dropna().unique())]}\n")

    step = max(1, int(round(every_ms * CANONICAL_HZ / 1000)))
    mark = np.zeros(len(sub), dtype=bool)
    mark[xs] = True
    print(f"{'t_s':>9}{'L_ang':>9}{'R_ang':>9}{'d':>9}  {'label':<14}swap")
    for i in range(0, len(sub), step):
        j = slice(i, min(i + step, len(sub)))
        lab = labels.iloc[i]
        flag = "  <<< commit" if mark[j].any() else ""
        print(f"{ts[i]:>9.2f}{l[i]:>9.2f}{r[i]:>9.2f}{d[i]:>9.2f}  "
              f"{CLASS_NAME.get(int(lab), '-') if pd.notna(lab) else '-':<14}{flag}")

    if len(xs) >= 2:
        gaps = np.diff(ts[xs])
        print(f"\ninter-commit intervals (s): "
              f"{', '.join(f'{g:.2f}' for g in gaps)}")
        print(f"median {np.median(gaps):.2f} s -> implied stride "
              f"{2 * np.median(gaps):.2f} s. Steady intervals are gait; one long gap "
              f"between two commits is a weight shift and back.")


def main() -> None:
    # explicit text, not __doc__: the module header is comments now
    use_replacement_encoding()
    ap = argparse.ArgumentParser(
        description="Print the raw interleg trace around one window.")
    ap.add_argument("rev")
    ap.add_argument("trial", type=int)
    ap.add_argument("--t", type=float, required=True,
                    help="centre of the span to print, in SECONDS")
    ap.add_argument("--span-s", type=float, default=12.0)
    ap.add_argument("--every-ms", type=int, default=100,
                    help="print one row per this many ms (crossings are still detected "
                         "at full rate and marked)")
    a = ap.parse_args()
    inspect(a.rev, a.trial, a.t, a.span_s, a.every_ms)


if __name__ == "__main__":
    main()
