# profile the legacy rule-based algorithm (`loco`) through OUR error taxonomy
# tests the [reported] claim that the rule-based algo fails differently from our
# classifier (few steady_confusion, many swallowed) -- if true, a mandate to build S3.
# runs `loco` as a prediction against the same ground truth on the paired recordings.
# does NOT resurrect loco (§4.5 severed it); measuring how it fails != trusting it. see README.

from __future__ import annotations

import glob
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from stages.s1_clean.census import read_header, strip_prefix
from stages.s2_ml.dataset import STAND, TIME_COL, TRAIN_CLASSES, WALK, _read_raw, find_trials
from stages.s2_ml.predict import labeled_runs
from stages.s2_ml.taxonomy import ERROR_BUCKETS, aggregate, bucket_errors

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCO_COL = "loco"


# read a raw csv only if it carries both `loco` and Time; else None
def load_raw_with_loco(path: str) -> pd.DataFrame | None:
    names = [strip_prefix(c) for c in read_header(Path(path))]
    if LOCO_COL not in names or "Time" not in names:
        return None
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return None
    df = df.loc[:, [c for c in df.columns if not c.startswith("Unnamed")]]
    df.columns = names[: len(df.columns)]
    return df


# annotated trials whose raw source carries `loco`, matched on the time vector
def find_pairs() -> list[tuple[Path, pd.DataFrame]]:
    raws = {}
    for f in sorted(glob.glob(str(REPO_ROOT / "data/raw/**/*.csv"), recursive=True)):
        df = load_raw_with_loco(f)
        if df is None:
            continue
        t = df["Time"].to_numpy(float)
        raws[f] = (t[0], t[-1], len(t), df)

    pairs = []
    for p in find_trials():
        t = _read_raw(p)[TIME_COL].to_numpy(float)
        for _f, (r0, r1, nr, df) in raws.items():
            if abs(t[0] - r0) < 50 and abs(t[-1] - r1) < 5000 and nr == len(t):
                pairs.append((p, df))
                break
    return pairs


# learn loco-code -> {stand, walk} by maximising agreement with ground truth
# the legacy encoding is undocumented, so it's measured; fitting to labels is deliberately
# generous to the incumbent, so any weakness the taxonomy reports is a floor on its error
def best_mapping(pairs: list[tuple[Path, pd.DataFrame]]) -> dict[int, int]:
    codes: set[int] = set()
    for _p, raw in pairs:
        codes.update(int(v) for v in pd.unique(raw[LOCO_COL].dropna()))
    codes = sorted(codes)

    best, best_score = None, -1.0
    for assignment in itertools.product(TRAIN_CLASSES, repeat=len(codes)):
        mapping = dict(zip(codes, assignment))
        hits = total = 0
        for p, raw in pairs:
            ann = pd.read_csv(p)
            ann.columns = [c.strip() for c in ann.columns]
            gt = ann["Label"].to_numpy()
            pred = np.array([mapping[int(v)] for v in raw[LOCO_COL].to_numpy()])
            m = np.isin(gt, TRAIN_CLASSES)
            hits += int(np.sum(gt[m] == pred[m]))
            total += int(m.sum())
        score = hits / total if total else 0.0
        if score > best_score:
            best, best_score = mapping, score
    print(f"[loco] learned mapping {best} (frame agreement {best_score:.4f}, "
          f"fitted to labels = generous to the incumbent)")
    return best


# pair recordings, learn the mapping, score `loco` through the taxonomy, write the json
def main() -> None:
    pairs = find_pairs()
    if not pairs:
        raise SystemExit("no paired recordings carrying `loco` — cannot profile")
    print(f"[loco] {len(pairs)} paired recordings carry both `loco` and ground truth")

    mapping = best_mapping(pairs)

    per_run = []
    for p, raw in pairs:
        ann = pd.read_csv(p)
        ann.columns = [c.strip() for c in ann.columns]
        gt = ann["Label"].to_numpy()
        pred = np.array([mapping[int(v)] for v in raw[LOCO_COL].to_numpy()])
        t = raw["Time"].to_numpy(float)
        for g, pr, tt in labeled_runs(gt, pred, t):
            per_run.append(bucket_errors(g, pr, tt))

    agg = aggregate(per_run)
    print(f"\n[loco] row accuracy: {agg['row_accuracy']:.4f}")
    print(f"[loco] dominant error: {agg['dominant']}")
    print(f"{'bucket':<18}{'rows':>10}{'share':>9}")
    for b in ERROR_BUCKETS:
        print(f"{b:<18}{agg['counts'][b]:>10,}{agg['fractions'][b]:>9.3f}")

    out = REPO_ROOT / "runs" / "s2_ml" / "incumbent_loco_taxonomy.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"mapping": {str(k): v for k, v in mapping.items()},
                               "n_pairs": len(pairs), **agg}, indent=2), encoding="utf-8")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
