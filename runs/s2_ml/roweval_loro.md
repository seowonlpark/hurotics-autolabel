# Row-level evaluation - leave-one-rev-out

subjects scored: rev13, rev2, rev3, rev4, rev5, rev6, rev7

| threshold | coverage | selective acc | worst subject | errors kept |
|---|---|---|---|---|
| 0.50 | 1.0000 | 0.9553 | 0.8805 | 55,573 |
| 0.60 | 0.9666 | 0.9684 | 0.8927 | 38,010 |
| 0.70 | 0.9281 | 0.9784 | 0.9096 | 24,908 |
| 0.75 | 0.9058 | 0.9825 | 0.9216 | 19,737 |
| 0.80 | 0.8788 | 0.9865 | 0.9381 | 14,734 |
| 0.85 | 0.8446 | 0.9904 | 0.9517 | 10,064 |
| 0.90 | 0.7937 | 0.9944 | 0.9672 | 5,517 |
| 0.95 | 0.6732 | 0.9983 | 0.9904 | 1,413 |
| 0.98 | 0.5340 | 0.9995 | 0.9952 | 340 |

## Ambiguity reasons

`accuracy of guess` is how often the guess WOULD have been right had the row not been flagged. A reason earns its place by sitting well below the confident row.

| reason | rows | accuracy of guess |
|---|---|---|
| `(confident, kept)` | 1,050,881 | 0.9904 |
| `near_transition` | 67,001 | 0.6022 |
| `weight_shift_or_step` | 43,989 | 0.8657 |
| `model_split` | 42,267 | 0.8269 |
| `out_of_distribution` | 22,839 | 0.8769 |
| `posture_shift` | 11,723 | 0.9369 |
| `low_excursion_gait` | 5,592 | 0.6284 |

## Agreement with human `-1` (§5.2)

- rows a human marked unknown: 26,533
- model abstains on those: **56.1%**
- model abstains elsewhere: 15.5%

The model never sees `-1` in training, so this is independent evidence.