# S2 champion - leave-one-rev-out CV

- windows: **5,984**
- **macro-F1: 0.9196**  (headline, §5.4)
- accuracy: 0.9624  ·  balanced accuracy: 0.9013

| class | support | precision | recall | F1 |
|---|---|---|---|---|
| stand | 853 | 0.9110 | 0.8159 | 0.8609 |
| walk | 5,131 | 0.9699 | 0.9867 | 0.9783 |

confusion (rows = truth):

| | pred stand | pred walk |
|---|---|---|
| **stand** | 696 | 157 |
| **walk** | 68 | 5,063 |

per-rev held-out scores (each rev = one subject/day, §7):

| rev | macro-F1 | window accuracy |
|---|---|---|
| `rev13` | 0.9664 | 99.57% |
| `rev2` | 0.8927 | 96.29% |
| `rev3` | 0.9687 | 99.05% |
| `rev4` | 0.9540 | 96.63% |
| `rev5` | 0.8553 | 92.41% |
| `rev6` | 0.9147 | 93.21% |
| `rev7` | 0.9111 | 95.16% |

## Selective accuracy

Accuracy on the rows the model commits to, against the share it commits to. `worst rev` is the same accuracy on the single worst held-out subject - the floor for a new person.

| threshold | coverage | selective acc | worst rev | errors kept |
|---|---|---|---|---|
| 0.50 | 1.0000 | 0.9624 | 0.9241 | 225 |
| 0.60 | 0.9796 | 0.9708 | 0.9371 | 171 |
| 0.70 | 0.9490 | 0.9789 | 0.9433 | 120 |
| 0.75 | 0.9283 | 0.9825 | 0.9568 | 97 |
| 0.80 | 0.9084 | 0.9864 | 0.9635 | 74 |
| 0.85 | 0.8800 | 0.9884 | 0.9699 | 61 |
| 0.90 | 0.8307 | 0.9907 | 0.9695 | 46 |
| 0.95 | 0.7294 | 0.9938 | 0.9813 | 27 |
| 0.98 | 0.5974 | 0.9980 | 0.9890 | 7 |

## Feature importance (top 12)

- `ang_LR_corr`: 0.1114
- `ileg_minhalf`: 0.0918
- `ileg_swaps`: 0.0523
- `L_ang_LPF_std`: 0.0481
- `ileg_minquarter`: 0.0470
- `R_ang_p95_p05`: 0.0452
- `L_ang_LPF_ptp`: 0.0438
- `R_ang_LPF_ptp`: 0.0425
- `L_ang_p95_p05`: 0.0404
- `R_angvel_LPF_absmean`: 0.0362
- `R_ang_LPF_std`: 0.0361
- `L_angvel_LPF_std`: 0.0357
