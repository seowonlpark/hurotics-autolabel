# S2 champion - leave-one-rev-out CV

- windows: **5,984**
- **macro-F1: 0.9176**  (headline, §5.4)
- accuracy: 0.9616  ·  balanced accuracy: 0.8989

| class | support | precision | recall | F1 |
|---|---|---|---|---|
| stand | 853 | 0.9093 | 0.8113 | 0.8575 |
| walk | 5,131 | 0.9692 | 0.9866 | 0.9778 |

confusion (rows = truth):

| | pred stand | pred walk |
|---|---|---|
| **stand** | 692 | 161 |
| **walk** | 69 | 5,062 |

per-rev macro-F1 (each rev = one subject/day, §7):

- `rev13`: 0.9664
- `rev2`: 0.8894
- `rev3`: 0.9687
- `rev4`: 0.9540
- `rev5`: 0.8553
- `rev6`: 0.9136
- `rev7`: 0.9071

## Selective accuracy

Accuracy on the rows the model commits to, against the share it commits to. `worst rev` is the same accuracy on the single worst held-out subject - the floor for a new person.

| threshold | coverage | selective acc | worst rev | errors kept |
|---|---|---|---|---|
| 0.50 | 1.0000 | 0.9616 | 0.9241 | 230 |
| 0.60 | 0.9773 | 0.9714 | 0.9371 | 167 |
| 0.70 | 0.9499 | 0.9784 | 0.9437 | 123 |
| 0.75 | 0.9306 | 0.9819 | 0.9500 | 101 |
| 0.80 | 0.9066 | 0.9860 | 0.9635 | 76 |
| 0.85 | 0.8793 | 0.9882 | 0.9699 | 62 |
| 0.90 | 0.8316 | 0.9906 | 0.9690 | 47 |
| 0.95 | 0.7326 | 0.9938 | 0.9815 | 27 |
| 0.98 | 0.6019 | 0.9981 | 0.9895 | 7 |

## Feature importance (top 12)

- `ang_LR_corr`: 0.1184
- `ileg_minhalf`: 0.0793
- `ileg_minquarter`: 0.0557
- `R_ang_p95_p05`: 0.0532
- `L_ang_p95_p05`: 0.0467
- `R_ang_LPF_ptp`: 0.0459
- `L_ang_LPF_std`: 0.0443
- `ileg_swaps`: 0.0436
- `R_ang_LPF_std`: 0.0431
- `R_angvel_LPF_absmean`: 0.0394
- `L_angvel_LPF_absmean`: 0.0383
- `L_ang_LPF_ptp`: 0.0372
