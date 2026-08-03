# S2 champion - leave-one-rev-out CV

- windows: **5,984**
- **macro-F1: 0.9155**  (headline, §5.4)
- accuracy: 0.9577  ·  balanced accuracy: 0.9250

| class | support | precision | recall | F1 |
|---|---|---|---|---|
| stand | 853 | 0.8333 | 0.8792 | 0.8557 |
| walk | 5,131 | 0.9797 | 0.9708 | 0.9752 |

confusion (rows = truth):

| | pred stand | pred walk |
|---|---|---|
| **stand** | 750 | 103 |
| **walk** | 150 | 4,981 |

per-rev macro-F1 (each rev = one subject/day, §7):

- `rev13`: 0.9735
- `rev2`: 0.8424
- `rev3`: 0.9687
- `rev4`: 0.9707
- `rev5`: 0.8857
- `rev6`: 0.9341
- `rev7`: 0.9089

## Selective accuracy

Accuracy on the rows the model commits to, against the share it commits to. `worst rev` is the same accuracy on the single worst held-out subject - the floor for a new person.

| threshold | coverage | selective acc | worst rev | errors kept |
|---|---|---|---|---|
| 0.50 | 1.0000 | 0.9577 | 0.9377 | 253 |
| 0.60 | 0.9659 | 0.9683 | 0.9509 | 183 |
| 0.70 | 0.9228 | 0.9788 | 0.9571 | 117 |
| 0.75 | 0.8956 | 0.9828 | 0.9704 | 92 |
| 0.80 | 0.8643 | 0.9870 | 0.9697 | 67 |
| 0.85 | 0.8194 | 0.9888 | 0.9754 | 55 |
| 0.90 | 0.7553 | 0.9916 | 0.9829 | 38 |
| 0.95 | 0.6544 | 0.9954 | 0.9808 | 18 |
| 0.98 | 0.5281 | 0.9978 | 0.9891 | 7 |

## Feature importance (top 12)

- `ileg_minhalf`: 0.1593
- `ileg_minquarter`: 0.1016
- `ileg_swaps`: 0.0975
- `L_ang_p95_p05`: 0.0725
- `R_ang_p95_p05`: 0.0612
- `L_ang_LPF_std`: 0.0579
- `ang_LR_corr`: 0.0461
- `R_ang_LPF_std`: 0.0456
- `R_ang_LPF_ptp`: 0.0397
- `L_ang_LPF_ptp`: 0.0391
- `R_angvel_LPF_absmean`: 0.0294
- `L_angvel_LPF_absmean`: 0.0244
