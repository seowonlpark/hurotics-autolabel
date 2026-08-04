# Row-level evaluation - leave-one-rev-out

subjects scored: rev13, rev2, rev3, rev4, rev5, rev6, rev7

| threshold | coverage | selective acc | worst subject | errors kept |
|---|---|---|---|---|
| 0.50 | 1.0000 | 0.9555 | 0.8799 | 55,357 |
| 0.60 | 0.9662 | 0.9682 | 0.8933 | 38,215 |
| 0.70 | 0.9275 | 0.9784 | 0.9099 | 24,916 |
| 0.75 | 0.9057 | 0.9824 | 0.9231 | 19,865 |
| 0.80 | 0.8794 | 0.9866 | 0.9383 | 14,676 |
| 0.85 | 0.8466 | 0.9901 | 0.9525 | 10,391 |
| 0.90 | 0.7953 | 0.9942 | 0.9663 | 5,727 |
| 0.95 | 0.6783 | 0.9980 | 0.9884 | 1,650 |
| 0.98 | 0.5485 | 0.9994 | 0.9937 | 390 |

## Per subject, at the shipped threshold 0.85

Each subject was scored by a model that never saw it. Read coverage and accuracy together per row: a subject with high accuracy over a small committed set is not one the pipeline handles well, it is one it mostly declined to answer for.

The two recalls carry their own denominators because the classes are wildly unbalanced on the committed set — a subject can hold a high accuracy while missing most of its standing, and pooled accuracy is the column that hides it.

| rev | rows scored | committed | coverage | errors kept | accuracy | stand recall | walk recall |
|---|---|---|---|---|---|---|---|
| `rev13` | 235,222 | 232,398 | 98.8% | 363 | 0.9984 | 0.9369 (n=5,754) | 1.0000 (n=226,644) |
| `rev2` | 362,401 | 277,106 | 76.5% | 1,331 | 0.9952 | 0.9689 (n=18,618) | 0.9971 (n=258,488) |
| `rev3` | 107,588 | 105,175 | 97.8% | 564 | 0.9946 | 0.9301 (n=8,064) | 1.0000 (n=97,111) |
| `rev4` | 20,726 | 16,750 | 80.8% | 75 | 0.9955 | 0.9797 (n=3,700) | 1.0000 (n=13,050) |
| `rev5` | 34,482 | 29,486 | 85.5% | 1,400 | 0.9525 | 0.6647 (n=4,175) | 1.0000 (n=25,311) |
| `rev6` | 264,693 | 209,712 | 79.2% | 3,392 | 0.9838 | 0.9211 (n=42,967) | 1.0000 (n=166,745) |
| `rev7` | 219,180 | 182,778 | 83.4% | 3,266 | 0.9821 | 0.8719 (n=23,353) | 0.9983 (n=159,425) |

## Direction of error, at the shipped threshold 0.85

1,053,405 rows committed, 190,887 abstained. **Rows, not windows** — the unit a caller receives, and the same unit as the coverage table above.

| truth → guess | rows |
|---|---|
| walk → walk | 945,747 |
| stand → stand | 97,267 |
| stand → walk ⚠ | 9,364 |
| walk → stand ⚠ | 1,027 |

**9,364 of 10,391 committed errors (90%) are stand called walk.** The residual failure is one-directional, so watch stand recall, not accuracy.

## Ambiguity reasons

`accuracy of guess` is how often the guess WOULD have been right had the row not been flagged. A reason earns its place by sitting well below the confident row.

| reason | rows | accuracy of guess |
|---|---|---|
| `(confident, kept)` | 1,053,405 | 0.9901 |
| `near_transition` | 69,600 | 0.6090 |
| `weight_shift_or_step` | 45,757 | 0.8767 |
| `model_split` | 42,576 | 0.8312 |
| `out_of_distribution` | 14,914 | 0.8427 |
| `posture_shift` | 11,698 | 0.9367 |
| `low_excursion_gait` | 6,342 | 0.7108 |

## Agreement with human `-1` (§5.2)

- rows a human marked unknown: 26,533
- model abstains on those: **55.8%**
- model abstains elsewhere: 15.3%

The model never sees `-1` in training, so this is independent evidence.

## Physics floor and ceiling, at threshold 0.85

`label.PHYSICS_CEILING` abstains where the swap rule is decisive and contradicts the model; `label.PHYSICS_FLOOR` accepts a lower confidence (0.70) where it is decisive and agrees. **Both ship OFF.** This table is what a decision to change that has to argue against.

Of the **10,391** errors the shipped policy commits to, the physics contradicts **580** (5.6%). That is the ceiling's entire addressable set — no tuning reaches an error the swap rule does not object to. The two read the same 1 Hz-filtered interleg angle, so they are wrong together (`caveats.md` §1.1c).

Physics is decisive on 1,167,595 of 1,244,292 scored rows (1,134,679 agree, 32,916 contradict).

| policy | coverage | selective acc | worst subject | errors kept | vs threshold-only at matched coverage |
|---|---|---|---|---|---|
| `off` | 0.8466 | 0.9901 | 0.9525 | 10,391 | *(the baseline)* |
| `ceiling` | 0.8433 | 0.9906 | 0.9524 | 9,811 | **-11 errors** at coverage 0.8427 (thr 0.8550) — better |
| `floor` | 0.9034 | 0.9839 | 0.9230 | 18,125 | **-1,290 errors** at coverage 0.9034 (thr 0.7550) — better |
| `both` | 0.9001 | 0.9843 | 0.9228 | 17,545 | **-1,133 errors** at coverage 0.9007 (thr 0.7600) — better |

**How to read the last column.** A policy that changes coverage cannot be judged on accuracy at a fixed threshold — giving up the rows you are least sure of always raises accuracy. Each gated arm is therefore matched against the threshold-only policy at the same coverage, and compared on surviving errors. Positive means the gate spent coverage worse than the threshold would have; negative means it spent it better.

### Across the threshold, because the objection is threshold-conditional

The retraction in `anchors.py` is specific: physics contradicts ~12% of S2's high-confidence errors and **none at p >= 0.95**. A gate measured at one operating point cannot speak to that, and the addressable-set column below is the direct check — it is the ceiling's ceiling.

| threshold | committed errors | physics objects to | ceiling vs thr-only | floor vs thr-only |
|---|---|---|---|---|
| 0.50 | 55,357 | 16,003 (28.9%) | -1,958 | +8,309 |
| 0.60 | 38,215 | 8,255 (21.6%) | -1,986 | +3,218 |
| 0.70 | 24,916 | 3,470 (13.9%) | -1,092 | +0 |
| 0.75 | 19,865 | 2,033 (10.2%) | -465 | -466 |
| 0.80 | 14,676 | 1,003 (6.8%) | -422 | -1,480 |
| 0.85 | 10,391 | 580 (5.6%) | -11 | -1,290 |
| 0.90 | 5,727 | 202 (3.5%) | -26 | -628 |
| 0.95 | 1,650 | 0 (0.0%) | +0 | +1,014 |
| 0.98 | 390 | 0 (0.0%) | +0 | +1,550 |

Negative is better (fewer surviving errors at the same coverage).
