# locoeval (trimmed)

An independent scorer for per-row locomotion predictions against hand-labeled ground
truth. Kept as a second, separately-written check on the pipeline's own metrics — if the
two disagree, one of them has a bug.

Trimmed from hurotics-locotool;
the exe-runner, corpus aggregation, fault-injection tests, sample data, and binary were
removed. What remains is the direct predictions-vs-labels path.

## Usage

```bash
python score.py <pred.csv> <gt.csv>
```

- `pred.csv` — a `prediction` (or `label`) column, one row per frame
- `gt.csv` — a `label` column, one row per frame
- rows are joined **by index**; the two files must have the same row count
- a `time`/`time_ms` column in either file is used if present, else a synthetic 100 Hz timeline
- labels are the loco codes (`0` STANDING, `10` WALKING, `255` UNKNOWN); **UNKNOWN is
  excluded from macro-F1**, the headline metric

Reports macro-F1, balanced accuracy, overall accuracy, per-class precision/recall/F1, and
the confusion matrix.

## Modules

| file | responsibility |
|---|---|
| `score.py` | CLI: load a pred/gt pair, print metrics |
| `locoeval/core.py` | align pred vs gt by row index (`load_aligned_direct`); health flags |
| `locoeval/measure.py` | frame metrics: confusion, per-class P/R/F1, macro-F1 |
| `locoeval/diagnose.py` | row-level error buckets + transition timing (optional deeper read) |

## Classes

`0` STANDING · `5` BENDING · `10` WALKING · `20` SIT_TO_STAND · `25` STAND_TO_SIT ·
`30` SITTING · `255` UNKNOWN · any other value → `CLASS_<n>`
