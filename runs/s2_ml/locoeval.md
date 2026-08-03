# S2 champion - leave-one-rev-out CV

- windows: **5,984**
- **macro-F1: 0.9193**  (headline, §5.4)
- accuracy: 0.9624  ·  balanced accuracy: 0.8999

| class | support | precision | recall | F1 |
|---|---|---|---|---|
| stand | 853 | 0.9142 | 0.8124 | 0.8603 |
| walk | 5,131 | 0.9694 | 0.9873 | 0.9783 |

confusion (rows = truth):

| | pred stand | pred walk |
|---|---|---|
| **stand** | 693 | 160 |
| **walk** | 65 | 5,066 |

per-rev macro-F1 (each rev = one subject/day, §7):

- `rev13`: 0.9592
- `rev2`: 0.8968
- `rev3`: 0.9687
- `rev4`: 0.9540
- `rev5`: 0.8553
- `rev6`: 0.9156
- `rev7`: 0.9056

## Selective accuracy

Accuracy on the rows the model commits to, against the share it commits to. `worst rev` is the same accuracy on the single worst held-out subject - the floor for a new person.

| threshold | coverage | selective acc | worst rev | errors kept |
|---|---|---|---|---|
| 0.50 | 1.0000 | 0.9624 | 0.9241 | 225 |
| 0.60 | 0.9781 | 0.9720 | 0.9371 | 164 |
| 0.70 | 0.9484 | 0.9787 | 0.9371 | 121 |
| 0.75 | 0.9268 | 0.9827 | 0.9433 | 96 |
| 0.80 | 0.9069 | 0.9860 | 0.9635 | 76 |
| 0.85 | 0.8780 | 0.9888 | 0.9699 | 59 |
| 0.90 | 0.8299 | 0.9911 | 0.9767 | 44 |
| 0.95 | 0.7263 | 0.9942 | 0.9725 | 25 |
| 0.98 | 0.5812 | 0.9983 | 0.9894 | 6 |

## Feature importance (top 12)

- `ang_LR_corr`: 0.0902
- `ileg_minhalf`: 0.0747
- `ileg_minquarter`: 0.0669
- `R_ang_p95_p05`: 0.0557
- `ileg_swaps`: 0.0541
- `L_ang_p95_p05`: 0.0481
- `R_ang_LPF_ptp`: 0.0437
- `L_ang_LPF_std`: 0.0420
- `L_ang_LPF_ptp`: 0.0413
- `R_ang_LPF_std`: 0.0409
- `R_angvel_LPF_std`: 0.0350
- `R_angvel_LPF_absmean`: 0.0347
