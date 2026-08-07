# Row-level evaluation - LOCKBOX (single use)

subjects scored: rev8

| threshold | coverage | selective acc | worst subject | errors kept |
|---|---|---|---|---|
| 0.50 | 1.0000 | 0.8680 | 0.8680 (`rev8`) | 5,487 |
| 0.60 | 0.9519 | 0.8834 | 0.8834 (`rev8`) | 4,612 |
| 0.70 | 0.8990 | 0.9018 | 0.9018 (`rev8`) | 3,668 |
| 0.75 | 0.8660 | 0.9060 | 0.9060 (`rev8`) | 3,384 |
| 0.80 | 0.8186 | 0.9183 | 0.9183 (`rev8`) | 2,781 |
| 0.85 | 0.7555 | 0.9269 | 0.9269 (`rev8`) | 2,297 |
| 0.90 | 0.5512 | 0.9303 | 0.9303 (`rev8`) | 1,597 |
| 0.95 | 0.2316 | 0.9064 | 0.9064 (`rev8`) | 901 |
| 0.98 | 0.0718 | 0.9373 | 0.9373 (`rev8`) | 187 |

## Per subject, at the shipped threshold 0.85

Each subject was scored by a model that never saw it. Read coverage and accuracy together per row: a subject with high accuracy over a small committed set is not one the pipeline handles well, it is one it mostly declined to answer for.

The two recalls carry their own denominators because the classes are wildly unbalanced on the committed set — a subject can hold a high accuracy while missing most of its standing, and pooled accuracy is the column that hides it.

| rev | rows scored | committed | coverage | errors kept | accuracy | stand recall | walk recall |
|---|---|---|---|---|---|---|---|
| `rev8` | 41,569 | 31,407 | 75.6% | 2,297 | 0.9269 | 0.5622 (n=5,247) | 1.0000 (n=26,160) |

## Direction of error, at the shipped threshold 0.85

31,407 rows committed, 10,162 abstained. **Rows, not windows** — the unit a caller receives, and the same unit as the coverage table above.

| truth → guess | rows |
|---|---|
| walk → walk | 26,160 |
| stand → stand | 2,950 |
| stand → walk ⚠ | 2,297 |

**2,297 of 2,297 committed errors (100%) are stand called walk.** The residual failure is one-directional, so watch stand recall, not accuracy.

## Ambiguity reasons

`accuracy of guess` is how often the guess WOULD have been right had the row not been flagged. A reason earns its place by sitting well below the confident row.

| reason | rows | accuracy of guess |
|---|---|---|
| `(confident, kept)` | 31,407 | 0.9269 |
| `model_split` | 4,180 | 0.8012 |
| `near_transition` | 3,957 | 0.5555 |
| `weight_shift_or_step` | 1,650 | 0.8182 |
| `low_excursion_gait` | 225 | 0.3333 |
| `posture_shift` | 150 | 0.0000 |

## Agreement with human `-1` (§5.2)

- rows a human marked unknown: 1,506
- model abstains on those: **34.1%**
- model abstains elsewhere: 24.4%

The model never sees `-1` in training, so this is independent evidence.

## Physics floor and ceiling, at threshold 0.85

`label.PHYSICS_CEILING` abstains where the swap rule is decisive and contradicts the model; `label.PHYSICS_FLOOR` accepts a lower confidence (0.70) where it is decisive and agrees. **Both ship OFF.** This table is what a decision to change that has to argue against.

Of the **2,297** errors the shipped policy commits to, the physics contradicts **25** (1.1%). That is the ceiling's entire addressable set — no tuning reaches an error the swap rule does not object to. The two read the same 1 Hz-filtered interleg angle, so they are wrong together (`caveats.md` §1.1c).

Physics is decisive on 37,825 of 41,569 scored rows (37,450 agree, 375 contradict).

| policy | coverage | selective acc | worst subject | errors kept | vs threshold-only at matched coverage |
|---|---|---|---|---|---|
| `off` | 0.7555 | 0.9269 | 0.9269 (`rev8`) | 2,297 | *(the baseline)* |
| `ceiling` | 0.7549 | 0.9276 | 0.9276 (`rev8`) | 2,272 | **-25 errors** at coverage 0.7555 (thr 0.8500) — better |
| `floor` | 0.8717 | 0.9128 | 0.9128 (`rev8`) | 3,159 | **-275 errors** at coverage 0.8708 (thr 0.7425) — better |
| `both` | 0.8711 | 0.9135 | 0.9135 (`rev8`) | 3,134 | **-300 errors** at coverage 0.8708 (thr 0.7425) — better |

**How to read the last column.** A policy that changes coverage cannot be judged on accuracy at a fixed threshold — giving up the rows you are least sure of always raises accuracy. Each gated arm is therefore matched against the threshold-only policy at the same coverage, and compared on surviving errors. Positive means the gate spent coverage worse than the threshold would have; negative means it spent it better.

### Across the threshold, because the objection is threshold-conditional

The retraction in `anchors.py` is specific: physics contradicts ~12% of S2's high-confidence errors and **none at p >= 0.95**. A gate measured at one operating point cannot speak to that, and the addressable-set column below is the direct check — it is the ceiling's ceiling.

| threshold | committed errors | physics objects to | ceiling vs thr-only | floor vs thr-only |
|---|---|---|---|---|
| 0.50 | 5,487 | 294 (5.4%) | -45 | -51 |
| 0.60 | 4,612 | 219 (4.7%) | -69 | +0 |
| 0.70 | 3,668 | 100 (2.7%) | -84 | +0 |
| 0.75 | 3,384 | 50 (1.5%) | -50 | -75 |
| 0.80 | 2,781 | 50 (1.8%) | -50 | -175 |
| 0.85 | 2,297 | 25 (1.1%) | -25 | -275 |
| 0.90 | 1,597 | 0 (0.0%) | +0 | -375 |
| 0.95 | 901 | 0 (0.0%) | +0 | -450 |
| 0.98 | 187 | 0 (0.0%) | +0 | -450 |

Negative is better (fewer surviving errors at the same coverage).
