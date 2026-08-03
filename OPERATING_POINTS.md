# Operating points

The classifier never has to answer. Every row gets a **best guess**, a **confidence**, and — when
confidence falls below the threshold — an **`ambiguous` flag with a reason**. The threshold is the
one knob that trades how much of a CSV gets labeled against how often those labels are right.

This file is the evidence for choosing it. Numbers are **leave-one-rev-out cross-validation over the
6 training subjects**, scored per row on dense inference (2 s window, 0.25 s stride, each row's
probability being the mean over every window covering it). rev8 and rev13 are the sealed lockbox and
are **not** in this table; see the "Lockbox" section.

---

## The three presets

| preset | threshold | coverage | accuracy on labeled rows | worst subject |
|---|---|---|---|---|
| `high_coverage` | 0.70 | 91.2% | 97.4% | 91.0% |
| **`balanced` (default)** | **0.85** | **80.9%** | **98.8%** | **95.2%** |
| `high_precision` | 0.95 | 59.8% | 99.8% | 98.9% |

*Coverage* = share of rows the model commits to. *Accuracy on labeled rows* = of those, how many are
right — the number your 95% target refers to. *Worst subject* = the same accuracy computed on the
single worst of the 6 held-out subjects, i.e. the realistic floor for a **new** person.

## Why 0.85 is the default and not 0.80

The two columns disagree, and the disagreement is the whole point:

| threshold | pooled accuracy | worst subject |
|---|---|---|
| 0.80 | 98.4% | **93.9%** |
| 0.85 | 98.8% | **95.2%** |

Pooled accuracy clears 95% from 0.50 onward, so it is the wrong number to set a threshold by — it
averages a new subject together with five the model has effectively seen. **0.85 is the lowest
threshold at which every held-out subject independently clears 95%.** Below it, the guarantee holds
on average but not for the next person who wears the device.

## Full curve

| threshold | coverage | accuracy on labeled | worst subject | wrong rows kept |
|---|---|---|---|---|
| 0.50 (no abstention) | 100.0% | 94.67% | 88.0% | 53,829 |
| 0.60 | 95.9% | 96.17% | 89.0% | 37,039 |
| 0.70 | 91.2% | 97.35% | 91.0% | 24,429 |
| 0.75 | 88.4% | 97.82% | 92.2% | 19,414 |
| 0.80 | 85.2% | 98.35% | 93.9% | 14,187 |
| **0.85** | **80.9%** | **98.82%** | **95.2%** | **9,660** |
| 0.90 | 74.2% | 99.31% | 97.0% | 5,185 |
| 0.95 | 59.8% | 99.77% | 98.9% | 1,359 |
| 0.98 | 44.8% | 99.96% | 99.2% | 192 |

Out of 1,009,070 scored rows.

## How to read the tradeoff

Going **up** the threshold buys accuracy at a steepening price. From 0.85 to 0.95 removes 8,301 of
the 9,660 remaining errors, but costs 21 points of coverage — roughly 210,000 rows moved to human
review to fix 8,300 labels. That is a good trade only if a wrong label is far more expensive than an
unreviewed one.

Going **down** is cheap at first and then not. 0.85 to 0.80 gains 4 points of coverage for 4,500 new
errors; 0.70 to 0.50 gains only 9 points of coverage for 29,000 errors, because by then the model is
committing to rows it has no signal on.

There is no threshold at which pooled accuracy reaches 100%. At 0.98 there are still 192 wrong rows
in 452,000. Some of those are genuine label disagreements rather than model failures (see
`CAVEATS.md`).

## Setting it

```powershell
python -m stages.s2_ml.label input.csv --preset balanced          # default
python -m stages.s2_ml.label input.csv --preset high_precision
python -m stages.s2_ml.label input.csv --threshold 0.88           # any value in [0.5, 1.0)
```

The threshold applies at label time only. It does not change the model, so sweeping it costs nothing
but a re-read of the cached probabilities, and the same trained champion serves every preset.

## Lockbox

rev8 and rev13 were sealed for the whole of feature selection, model selection and threshold
selection. They are scored **once**, after everything above is frozen, and the result is recorded in
`CAVEATS.md`. If the lockbox numbers come in materially below this table, believe the lockbox: it is
the only measurement here that was never optimized against.
