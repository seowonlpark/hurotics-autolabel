# S2 champion - leave-one-rev-out CV

- windows: **5,984**
- **macro-F1: 0.9168**  (headline, §5.4)
- accuracy: 0.9612  ·  balanced accuracy: 0.8977

| class | support | precision | recall | F1 |
|---|---|---|---|---|
| stand | 853 | 0.9091 | 0.8089 | 0.8561 |
| walk | 5,131 | 0.9688 | 0.9866 | 0.9776 |

confusion (rows = truth):

| | pred stand | pred walk |
|---|---|---|
| **stand** | 690 | 163 |
| **walk** | 69 | 5,062 |

per-rev macro-F1 (each rev = one subject/day, §7):

- `rev13`: 0.9664
- `rev2`: 0.8849
- `rev3`: 0.9687
- `rev4`: 0.9540
- `rev5`: 0.8553
- `rev6`: 0.9138
- `rev7`: 0.9071

## Selective accuracy

Accuracy on the rows the model commits to, against the share it commits to. `worst rev` is the same accuracy on the single worst held-out subject - the floor for a new person.

| threshold | coverage | selective acc | worst rev | errors kept |
|---|---|---|---|---|
| 0.50 | 1.0000 | 0.9612 | 0.9241 | 232 |
| 0.60 | 0.9768 | 0.9716 | 0.9371 | 166 |
| 0.70 | 0.9470 | 0.9785 | 0.9433 | 122 |
| 0.75 | 0.9275 | 0.9820 | 0.9500 | 100 |
| 0.80 | 0.9074 | 0.9855 | 0.9562 | 79 |
| 0.85 | 0.8758 | 0.9887 | 0.9699 | 59 |
| 0.90 | 0.8307 | 0.9905 | 0.9697 | 47 |
| 0.95 | 0.7279 | 0.9933 | 0.9889 | 29 |
| 0.98 | 0.5901 | 0.9966 | 0.9897 | 12 |

## Feature importance (top 12)

- `ang_LR_corr`: 0.1026
- `ileg_minhalf`: 0.0937
- `ileg_swaps`: 0.0594
- `L_ang_LPF_std`: 0.0497
- `ileg_minquarter`: 0.0473
- `L_ang_p95_p05`: 0.0459
- `R_ang_p95_p05`: 0.0454
- `R_ang_LPF_std`: 0.0436
- `L_ang_LPF_ptp`: 0.0422
- `L_angvel_LPF_absmean`: 0.0403
- `R_ang_LPF_ptp`: 0.0402
- `R_angvel_LPF_absmean`: 0.0345
