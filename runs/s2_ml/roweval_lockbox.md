# Row-level evaluation - LOCKBOX (single use)

subjects scored: rev8

| threshold | coverage | selective acc | worst subject | errors kept |
|---|---|---|---|---|
| 0.50 | 1.0000 | 0.8680 | 0.8680 | 5,487 |
| 0.60 | 0.9549 | 0.8819 | 0.8819 | 4,687 |
| 0.70 | 0.9038 | 0.8997 | 0.8997 | 3,768 |
| 0.75 | 0.8684 | 0.9063 | 0.9063 | 3,384 |
| 0.80 | 0.8252 | 0.9175 | 0.9175 | 2,831 |
| 0.85 | 0.7639 | 0.9308 | 0.9308 | 2,197 |
| 0.90 | 0.5596 | 0.9314 | 0.9314 | 1,597 |
| 0.95 | 0.2360 | 0.9184 | 0.9184 | 801 |
| 0.98 | 0.0664 | 0.9594 | 0.9594 | 112 |

## Ambiguity reasons

`accuracy of guess` is how often the guess WOULD have been right had the row not been flagged. A reason earns its place by sitting well below the confident row.

| reason | rows | accuracy of guess |
|---|---|---|
| `(confident, kept)` | 31,753 | 0.9308 |
| `near_transition` | 3,957 | 0.5492 |
| `model_split` | 3,705 | 0.7690 |
| `weight_shift_or_step` | 1,750 | 0.8143 |
| `low_excursion_gait` | 250 | 0.3000 |
| `posture_shift` | 154 | 0.0260 |

## Agreement with human `-1` (§5.2)

- rows a human marked unknown: 1,506
- model abstains on those: **42.1%**
- model abstains elsewhere: 23.6%

The model never sees `-1` in training, so this is independent evidence.