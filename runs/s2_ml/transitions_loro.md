# Transitions - leave-one-rev-out

What the deliverable's biggest abstention bucket is actually made of. `near_transition` suppresses more rows than any other reason and its suppressed guess is the least accurate of any; until now nothing measured the boundaries it fires on (`needtowrite.md` §5.4, gap 5).

**267 timeable transitions**, from 308 annotated stand/walk boundaries. Rejected: 41 short flank, 0 near segment edge.

A boundary is timeable when both sides hold at least 1 full window (2 s) of their own class and it sits at least 1 window from either end of its segment — below that the model has no pure window to place a change from, and scoring its timing would measure the window length. **The rejections are counted, not dropped.** Everything here encodes the raw `truth` column with the unknowns still in it, and reads `-1` as the boundary's width rather than removing it — filtering first splits one boundary into many.

## How many, and which way

| direction | transitions |
|---|---|
| `stand->walk` | 136 |
| `walk->stand` | 131 |

## How long is a transition

The annotator's own `-1` interval between the two runs — ground truth does not step instantaneously, and this is the width of its admitted doubt rather than a model of it. A width of 0 means the boundary was called to the sample.

| median | mean | p90 | max | called to the sample |
|---|---|---|---|---|
| 0.14 s | 0.33 s | 0.58 s | 12.62 s | 0 of 267 |

## Is the pipeline's timing right

Offset from each annotated boundary to the nearest state change the model predicted. **Positive is late.** Measured on the guess, not the committed label: where the model thinks the boundary is does not depend on whether it cleared the threshold on either side.

- transitions with a predicted change in the segment: **266 of 267** (1 segment(s) never change class at all)
- median offset: **-0.18 s** (117 late, 148 early)
- median |offset|: **0.73 s**, p90 4.12 s
- within ±1 s (the radius `near_transition` marks): **170 of 266** (64%)
- within ±2 s (one window): **223 of 266** (84%)

## Does `near_transition` fire where the boundaries are

**218 of 267 (82%)** annotated transitions have a `near_transition` row within ±1 s. That is the flag's recall.

Its precision, from the other side: the 77,025 flagged rows form **384 contiguous regions**, of which **167 (43%)** contain no annotated transition within ±2 s of their span.

Regions, not rows: a stretch of flagged rows is one event however wide it is, and a row count says nothing about how many boundaries the flag claims. An unmatched region is not necessarily wrong — the model may be changing class somewhere the annotation does not — but it is the flag firing where ground truth sees no boundary, which is what its 0.60 suppressed-guess accuracy has to be read against.

## Per subject

| rev | transitions | median width | median &#124;offset&#124; | within ±1 s | flagged |
|---|---|---|---|---|---|
| `rev13` | 6 | 0.36 s | 1.81 s | 17% | 50% |
| `rev2` | 66 | 0.20 s | 0.63 s | 76% | 88% |
| `rev3` | 10 | 0.03 s | 0.66 s | 80% | 80% |
| `rev4` | 16 | 0.01 s | 0.39 s | 88% | 88% |
| `rev5` | 18 | 0.01 s | 6.41 s | 29% | 33% |
| `rev6` | 74 | 0.18 s | 0.66 s | 66% | 85% |
| `rev7` | 77 | 0.12 s | 0.93 s | 56% | 86% |

## Every timeable transition

| rev | trial | seg | t (s) | direction | width | offset | flagged |
|---|---|---|---|---|---|---|---|
| rev13 | 1 | 0 | 583.10 | `stand->walk` | 1.71 s | +0.74 s | yes |
| rev13 | 2 | 0 | 821.71 | `stand->walk` | 2.39 s | -2.12 s | NO |
| rev13 | 3 | 0 | 1323.32 | `stand->walk` | 0.32 s | -1.50 s | yes |
| rev13 | 5 | 0 | 1789.26 | `stand->walk` | 0.05 s | -3.50 s | NO |
| rev13 | 6 | 0 | 180.52 | `stand->walk` | 0.41 s | -3.76 s | NO |
| rev13 | 7 | 0 | 935.99 | `stand->walk` | 0.21 s | -1.48 s | yes |
| rev2 | 1 | 1 | 244.60 | `stand->walk` | 0.29 s | -1.59 s | NO |
| rev2 | 1 | 1 | 362.93 | `walk->stand` | 0.01 s | +0.84 s | yes |
| rev2 | 1 | 1 | 369.18 | `stand->walk` | 0.57 s | -1.91 s | NO |
| rev2 | 1 | 1 | 550.57 | `walk->stand` | 0.86 s | +0.94 s | yes |
| rev2 | 1 | 1 | 556.82 | `stand->walk` | 1.43 s | -2.05 s | NO |
| rev2 | 1 | 1 | 679.26 | `walk->stand` | 0.30 s | +0.76 s | yes |
| rev2 | 2 | 0 | 684.51 | `stand->walk` | 0.01 s | -0.32 s | yes |
| rev2 | 3 | 1 | 1162.50 | `stand->walk` | 0.92 s | +0.84 s | yes |
| rev2 | 3 | 1 | 1285.45 | `walk->stand` | 0.46 s | +0.14 s | yes |
| rev2 | 3 | 1 | 1290.53 | `stand->walk` | 0.61 s | -0.43 s | yes |
| rev2 | 3 | 1 | 1473.89 | `walk->stand` | 0.05 s | +1.21 s | yes |
| rev2 | 3 | 1 | 1478.72 | `stand->walk` | 0.29 s | -0.62 s | yes |
| rev2 | 3 | 1 | 1601.59 | `walk->stand` | 0.01 s | +0.26 s | yes |
| rev2 | 3 | 1 | 1608.41 | `stand->walk` | 0.01 s | -0.81 s | yes |
| rev2 | 3 | 1 | 1848.87 | `walk->stand` | 0.01 s | +0.48 s | yes |
| rev2 | 4 | 1 | 152.46 | `stand->walk` | 0.01 s | +0.63 s | yes |
| rev2 | 4 | 1 | 275.99 | `walk->stand` | 0.12 s | -0.40 s | yes |
| rev2 | 4 | 1 | 279.83 | `stand->walk` | 0.07 s | -0.49 s | yes |
| rev2 | 4 | 1 | 463.73 | `walk->stand` | 0.42 s | -0.14 s | yes |
| rev2 | 4 | 1 | 468.12 | `stand->walk` | 0.01 s | -0.78 s | yes |
| rev2 | 4 | 1 | 592.16 | `walk->stand` | 0.18 s | -0.82 s | yes |
| rev2 | 4 | 1 | 595.73 | `stand->walk` | 0.07 s | +0.11 s | yes |
| rev2 | 4 | 1 | 837.14 | `walk->stand` | 0.22 s | +0.94 s | yes |
| rev2 | 5 | 1 | 140.63 | `stand->walk` | 0.01 s | +0.41 s | yes |
| rev2 | 5 | 1 | 262.62 | `walk->stand` | 0.20 s | -0.09 s | yes |
| rev2 | 5 | 1 | 267.69 | `stand->walk` | 0.23 s | +0.10 s | yes |
| rev2 | 5 | 1 | 450.19 | `walk->stand` | 0.31 s | +0.35 s | yes |
| rev2 | 5 | 1 | 455.90 | `stand->walk` | 0.20 s | -0.36 s | yes |
| rev2 | 5 | 1 | 578.78 | `walk->stand` | 0.38 s | -1.75 s | yes |
| rev2 | 5 | 1 | 584.70 | `stand->walk` | 0.17 s | +0.34 s | yes |
| rev2 | 5 | 1 | 826.81 | `walk->stand` | 0.19 s | -1.02 s | yes |
| rev2 | 6 | 1 | 1166.76 | `stand->walk` | 0.29 s | +0.63 s | yes |
| rev2 | 6 | 1 | 1220.03 | `walk->stand` | 0.01 s | -0.89 s | yes |
| rev2 | 6 | 1 | 1224.15 | `stand->walk` | 0.30 s | +0.49 s | yes |
| rev2 | 6 | 1 | 1312.64 | `walk->stand` | 0.01 s | -1.75 s | yes |
| rev2 | 6 | 1 | 1318.75 | `stand->walk` | 0.29 s | +0.89 s | yes |
| rev2 | 6 | 1 | 1343.47 | `walk->stand` | 0.30 s | -0.57 s | yes |
| rev2 | 6 | 1 | 1347.07 | `stand->walk` | 0.13 s | -0.18 s | yes |
| rev2 | 6 | 1 | 1384.45 | `walk->stand` | 0.06 s | -0.56 s | yes |
| rev2 | 6 | 1 | 1393.64 | `stand->walk` | 0.01 s | +0.75 s | yes |
| rev2 | 6 | 1 | 1447.84 | `stand->walk` | 8.78 s | -2.19 s | NO |
| rev2 | 6 | 1 | 1467.81 | `walk->stand` | 0.44 s | -0.92 s | yes |
| rev2 | 6 | 1 | 1475.67 | `stand->walk` | 0.12 s | +0.47 s | yes |
| rev2 | 6 | 1 | 1502.82 | `walk->stand` | 0.33 s | -0.93 s | yes |
| rev2 | 6 | 1 | 1523.50 | `stand->walk` | 0.14 s | -0.10 s | yes |
| rev2 | 6 | 1 | 1541.74 | `walk->stand` | 0.01 s | -0.60 s | yes |
| rev2 | 6 | 1 | 1550.24 | `stand->walk` | 0.20 s | -0.84 s | yes |
| rev2 | 6 | 1 | 1569.52 | `walk->stand` | 0.01 s | -0.63 s | yes |
| rev2 | 7 | 1 | 1699.21 | `stand->walk` | 0.01 s | -0.10 s | yes |
| rev2 | 7 | 1 | 1706.57 | `walk->stand` | 0.09 s | -1.96 s | yes |
| rev2 | 7 | 1 | 1711.94 | `stand->walk` | 0.01 s | +5.42 s | NO |
| rev2 | 7 | 1 | 1729.80 | `walk->stand` | 0.09 s | -1.69 s | yes |
| rev2 | 7 | 1 | 1737.16 | `stand->walk` | 0.01 s | +1.45 s | yes |
| rev2 | 7 | 1 | 1760.10 | `walk->stand` | 0.09 s | -0.49 s | yes |
| rev2 | 7 | 1 | 1766.47 | `stand->walk` | 0.09 s | -6.86 s | NO |
| rev2 | 7 | 1 | 1807.95 | `walk->stand` | 0.01 s | -1.34 s | yes |
| rev2 | 8 | 1 | 1996.33 | `stand->walk` | 0.21 s | +0.18 s | yes |
| rev2 | 8 | 1 | 2037.89 | `walk->stand` | 0.21 s | +1.37 s | yes |
| rev2 | 8 | 1 | 2055.40 | `stand->walk` | 0.60 s | -0.64 s | yes |
| rev2 | 8 | 1 | 2095.27 | `walk->stand` | 0.01 s | +1.24 s | NO |
| rev2 | 8 | 1 | 2114.36 | `stand->walk` | 0.41 s | -0.35 s | yes |
| rev2 | 8 | 1 | 2157.91 | `walk->stand` | 0.41 s | -0.15 s | yes |
| rev2 | 8 | 1 | 2161.29 | `stand->walk` | 0.41 s | -0.28 s | yes |
| rev2 | 8 | 1 | 2192.21 | `walk->stand` | 0.21 s | -0.45 s | yes |
| rev2 | 8 | 1 | 2200.96 | `stand->walk` | 0.21 s | -0.45 s | NO |
| rev2 | 8 | 1 | 2266.19 | `walk->stand` | 0.21 s | +0.57 s | yes |
| rev3 | 1 | 0 | 97.47 | `stand->walk` | 0.01 s | -0.53 s | yes |
| rev3 | 1 | 0 | 338.24 | `walk->stand` | 0.18 s | +3.96 s | NO |
| rev3 | 2 | 0 | 66.39 | `stand->walk` | 0.01 s | -0.66 s | yes |
| rev3 | 2 | 0 | 296.51 | `walk->stand` | 0.01 s | -230.78 s | NO |
| rev3 | 3 | 0 | 349.98 | `stand->walk` | 0.08 s | -0.17 s | yes |
| rev3 | 3 | 0 | 616.16 | `walk->stand` | 0.06 s | +0.39 s | yes |
| rev3 | 3 | 0 | 623.72 | `stand->walk` | 0.06 s | -0.67 s | yes |
| rev3 | 3 | 0 | 677.64 | `walk->stand` | 0.01 s | +0.66 s | yes |
| rev3 | 3 | 0 | 684.73 | `stand->walk` | 0.04 s | -0.93 s | yes |
| rev3 | 3 | 0 | 867.20 | `walk->stand` | 0.01 s | +0.10 s | yes |
| rev4 | 1 | 1 | 267.82 | `stand->walk` | 0.01 s | +0.01 s | yes |
| rev4 | 1 | 1 | 283.31 | `walk->stand` | 0.01 s | +0.03 s | yes |
| rev4 | 1 | 1 | 288.56 | `stand->walk` | 0.01 s | +0.02 s | yes |
| rev4 | 1 | 1 | 296.44 | `walk->stand` | 0.15 s | +0.64 s | yes |
| rev4 | 1 | 1 | 298.87 | `stand->walk` | 0.15 s | -0.53 s | yes |
| rev4 | 1 | 1 | 329.40 | `walk->stand` | 0.15 s | +0.43 s | yes |
| rev4 | 1 | 1 | 334.51 | `stand->walk` | 0.15 s | -0.18 s | yes |
| rev4 | 1 | 1 | 359.09 | `walk->stand` | 0.16 s | +0.24 s | yes |
| rev4 | 1 | 1 | 362.14 | `stand->walk` | 0.01 s | -0.56 s | yes |
| rev4 | 1 | 1 | 376.43 | `walk->stand` | 0.15 s | -14.84 s | NO |
| rev4 | 1 | 1 | 379.33 | `stand->walk` | 0.01 s | -17.75 s | NO |
| rev4 | 1 | 1 | 399.79 | `walk->stand` | 0.01 s | -0.70 s | yes |
| rev4 | 1 | 1 | 404.33 | `stand->walk` | 0.01 s | +0.00 s | yes |
| rev4 | 1 | 1 | 429.48 | `walk->stand` | 0.01 s | -0.14 s | yes |
| rev4 | 1 | 1 | 443.68 | `stand->walk` | 0.01 s | -0.34 s | yes |
| rev4 | 1 | 1 | 450.29 | `walk->stand` | 0.15 s | +0.55 s | yes |
| rev5 | 1 | 1 | 74.20 | `stand->walk` | 0.02 s | -0.72 s | yes |
| rev5 | 1 | 1 | 79.89 | `walk->stand` | 0.01 s | -6.41 s | NO |
| rev5 | 1 | 1 | 82.09 | `stand->walk` | 0.01 s | -8.62 s | NO |
| rev5 | 1 | 1 | 86.44 | `walk->stand` | 0.01 s | -12.97 s | NO |
| rev5 | 2 | 0 | 221.15 | `walk->stand` | 0.02 s | +3.00 s | NO |
| rev5 | 3 | 2 | 238.43 | `stand->walk` | 0.03 s | -1.85 s | yes |
| rev5 | 3 | 2 | 250.34 | `walk->stand` | 0.03 s | +4.24 s | NO |
| rev5 | 4 | 1 | 318.89 | `walk->stand` | 0.01 s | — | NO |
| rev5 | 5 | 1 | 72.84 | `stand->walk` | 0.01 s | -0.64 s | yes |
| rev5 | 5 | 1 | 107.82 | `walk->stand` | 0.01 s | +29.63 s | NO |
| rev5 | 5 | 1 | 109.88 | `stand->walk` | 0.06 s | +27.57 s | NO |
| rev5 | 5 | 1 | 120.40 | `walk->stand` | 0.01 s | +17.05 s | NO |
| rev5 | 5 | 1 | 122.95 | `stand->walk` | 0.01 s | +14.50 s | NO |
| rev5 | 5 | 1 | 136.61 | `walk->stand` | 0.01 s | +0.84 s | yes |
| rev5 | 6 | 1 | 182.54 | `walk->stand` | 0.01 s | +26.50 s | NO |
| rev5 | 6 | 1 | 184.86 | `stand->walk` | 0.04 s | +24.18 s | NO |
| rev5 | 6 | 1 | 208.50 | `walk->stand` | 0.01 s | +0.54 s | yes |
| rev5 | 7 | 1 | 338.57 | `walk->stand` | 0.01 s | +0.23 s | yes |
| rev6 | 1 | 1 | 121.66 | `stand->walk` | 0.16 s | -0.47 s | yes |
| rev6 | 1 | 1 | 191.18 | `walk->stand` | 0.24 s | +0.27 s | yes |
| rev6 | 1 | 1 | 210.94 | `stand->walk` | 0.17 s | +0.25 s | yes |
| rev6 | 1 | 1 | 300.24 | `walk->stand` | 0.16 s | -0.04 s | yes |
| rev6 | 1 | 1 | 312.50 | `stand->walk` | 0.01 s | -0.56 s | yes |
| rev6 | 1 | 1 | 389.08 | `walk->stand` | 0.16 s | -0.39 s | yes |
| rev6 | 1 | 1 | 400.29 | `stand->walk` | 0.35 s | -0.35 s | yes |
| rev6 | 1 | 1 | 474.01 | `walk->stand` | 0.04 s | -0.32 s | yes |
| rev6 | 1 | 1 | 483.47 | `stand->walk` | 0.31 s | -0.78 s | yes |
| rev6 | 1 | 1 | 599.14 | `walk->stand` | 0.19 s | +1.30 s | yes |
| rev6 | 1 | 1 | 602.69 | `stand->walk` | 0.04 s | -0.49 s | yes |
| rev6 | 1 | 1 | 636.61 | `walk->stand` | 0.18 s | -0.41 s | yes |
| rev6 | 1 | 1 | 642.03 | `stand->walk` | 0.78 s | -0.58 s | yes |
| rev6 | 1 | 1 | 648.97 | `walk->stand` | 0.16 s | +0.23 s | yes |
| rev6 | 1 | 1 | 655.09 | `stand->walk` | 0.01 s | +0.35 s | yes |
| rev6 | 1 | 1 | 659.88 | `walk->stand` | 0.01 s | -0.19 s | yes |
| rev6 | 1 | 1 | 686.05 | `stand->walk` | 0.08 s | -0.35 s | yes |
| rev6 | 1 | 1 | 744.46 | `walk->stand` | 0.15 s | +1.48 s | yes |
| rev6 | 1 | 1 | 866.39 | `stand->walk` | 0.31 s | +0.05 s | yes |
| rev6 | 1 | 1 | 900.68 | `walk->stand` | 1.23 s | +1.01 s | yes |
| rev6 | 2 | 0 | 1050.06 | `stand->walk` | 0.18 s | -0.41 s | yes |
| rev6 | 2 | 0 | 1059.43 | `walk->stand` | 0.18 s | -0.53 s | yes |
| rev6 | 2 | 0 | 1072.98 | `stand->walk` | 0.01 s | +0.92 s | yes |
| rev6 | 2 | 0 | 1081.42 | `walk->stand` | 0.86 s | +0.73 s | yes |
| rev6 | 2 | 0 | 1135.03 | `stand->walk` | 12.62 s | -1.88 s | yes |
| rev6 | 2 | 0 | 1146.61 | `walk->stand` | 0.35 s | +0.54 s | yes |
| rev6 | 2 | 0 | 1149.34 | `stand->walk` | 0.35 s | -0.44 s | yes |
| rev6 | 2 | 0 | 1167.33 | `walk->stand` | 3.94 s | +1.82 s | yes |
| rev6 | 2 | 0 | 1172.69 | `stand->walk` | 0.35 s | -0.79 s | yes |
| rev6 | 2 | 0 | 1179.60 | `walk->stand` | 0.18 s | +1.30 s | yes |
| rev6 | 2 | 0 | 1187.27 | `stand->walk` | 0.18 s | -0.36 s | yes |
| rev6 | 2 | 0 | 1198.09 | `walk->stand` | 0.01 s | +0.56 s | yes |
| rev6 | 3 | 1 | 271.58 | `stand->walk` | 0.05 s | -0.84 s | yes |
| rev6 | 3 | 1 | 357.12 | `walk->stand` | 0.41 s | +0.62 s | yes |
| rev6 | 3 | 1 | 366.30 | `stand->walk` | 0.25 s | -0.56 s | yes |
| rev6 | 3 | 1 | 382.75 | `walk->stand` | 0.03 s | +0.24 s | yes |
| rev6 | 3 | 1 | 405.40 | `stand->walk` | 0.12 s | -4.91 s | NO |
| rev6 | 3 | 1 | 434.68 | `walk->stand` | 0.21 s | -34.19 s | NO |
| rev6 | 3 | 1 | 437.45 | `stand->walk` | 0.01 s | +33.54 s | NO |
| rev6 | 3 | 1 | 468.65 | `walk->stand` | 0.05 s | +2.34 s | NO |
| rev6 | 3 | 1 | 473.76 | `stand->walk` | 0.01 s | -1.02 s | yes |
| rev6 | 3 | 1 | 515.60 | `walk->stand` | 0.01 s | -42.86 s | NO |
| rev6 | 3 | 1 | 520.12 | `stand->walk` | 0.42 s | -47.38 s | NO |
| rev6 | 3 | 1 | 582.17 | `walk->stand` | 0.22 s | -109.42 s | NO |
| rev6 | 3 | 1 | 586.47 | `stand->walk` | 0.21 s | -113.73 s | NO |
| rev6 | 3 | 1 | 710.07 | `walk->stand` | 0.03 s | +1.42 s | yes |
| rev6 | 3 | 1 | 721.40 | `stand->walk` | 0.07 s | -0.16 s | yes |
| rev6 | 3 | 1 | 777.59 | `walk->stand` | 0.24 s | +1.91 s | yes |
| rev6 | 3 | 1 | 831.20 | `stand->walk` | 0.17 s | -0.21 s | yes |
| rev6 | 3 | 1 | 904.15 | `walk->stand` | 0.52 s | +1.59 s | yes |
| rev6 | 3 | 1 | 910.22 | `stand->walk` | 0.39 s | -1.48 s | yes |
| rev6 | 3 | 1 | 1022.89 | `walk->stand` | 0.04 s | +3.85 s | NO |
| rev6 | 4 | 1 | 97.31 | `stand->walk` | 0.01 s | -0.87 s | yes |
| rev6 | 4 | 1 | 285.09 | `walk->stand` | 0.01 s | +1.85 s | yes |
| rev6 | 4 | 1 | 312.79 | `stand->walk` | 0.29 s | -0.10 s | yes |
| rev6 | 4 | 1 | 318.61 | `walk->stand` | 0.01 s | -5.92 s | NO |
| rev6 | 4 | 1 | 376.85 | `stand->walk` | 0.01 s | -0.66 s | yes |
| rev6 | 4 | 1 | 473.87 | `walk->stand` | 0.29 s | -0.18 s | yes |
| rev6 | 6 | 1 | 179.12 | `stand->walk` | 0.01 s | -0.17 s | yes |
| rev6 | 6 | 1 | 188.92 | `walk->stand` | 0.30 s | -0.72 s | yes |
| rev6 | 6 | 1 | 223.16 | `stand->walk` | 0.01 s | +0.29 s | yes |
| rev6 | 6 | 1 | 430.96 | `walk->stand` | 0.29 s | +0.73 s | yes |
| rev6 | 6 | 1 | 444.61 | `stand->walk` | 0.29 s | -0.66 s | yes |
| rev6 | 6 | 1 | 461.37 | `walk->stand` | 0.29 s | +1.33 s | yes |
| rev6 | 6 | 1 | 476.00 | `stand->walk` | 0.01 s | -0.30 s | yes |
| rev6 | 6 | 1 | 510.65 | `walk->stand` | 0.01 s | +0.29 s | yes |
| rev6 | 6 | 1 | 513.92 | `stand->walk` | 0.30 s | -0.72 s | yes |
| rev6 | 6 | 1 | 516.76 | `walk->stand` | 0.30 s | +0.18 s | yes |
| rev6 | 6 | 1 | 542.04 | `stand->walk` | 0.29 s | -2.10 s | NO |
| rev6 | 6 | 1 | 548.15 | `walk->stand` | 0.01 s | -0.71 s | yes |
| rev6 | 6 | 1 | 553.98 | `stand->walk` | 0.29 s | -0.28 s | yes |
| rev6 | 6 | 1 | 586.08 | `walk->stand` | 0.30 s | -0.89 s | yes |
| rev6 | 6 | 1 | 617.18 | `stand->walk` | 0.01 s | +0.01 s | yes |
| rev6 | 6 | 1 | 629.54 | `walk->stand` | 0.29 s | +1.15 s | yes |
| rev7 | 1 | 1 | 248.13 | `stand->walk` | 0.01 s | -1.21 s | yes |
| rev7 | 1 | 1 | 253.11 | `walk->stand` | 0.01 s | +0.56 s | yes |
| rev7 | 1 | 1 | 260.98 | `stand->walk` | 0.01 s | -1.56 s | yes |
| rev7 | 1 | 1 | 266.85 | `walk->stand` | 0.01 s | +2.32 s | NO |
| rev7 | 1 | 1 | 269.06 | `stand->walk` | 0.01 s | +0.11 s | yes |
| rev7 | 1 | 1 | 276.91 | `walk->stand` | 0.01 s | -3.99 s | NO |
| rev7 | 3 | 1 | 74.24 | `stand->walk` | 0.71 s | -0.60 s | yes |
| rev7 | 3 | 1 | 86.02 | `walk->stand` | 0.29 s | +0.62 s | yes |
| rev7 | 3 | 1 | 107.64 | `stand->walk` | 0.01 s | +0.25 s | yes |
| rev7 | 3 | 1 | 247.24 | `walk->stand` | 0.01 s | +1.15 s | yes |
| rev7 | 3 | 1 | 272.81 | `stand->walk` | 1.03 s | -2.42 s | NO |
| rev7 | 3 | 1 | 622.97 | `walk->stand` | 0.78 s | +0.93 s | yes |
| rev7 | 3 | 1 | 627.07 | `stand->walk` | 0.23 s | -0.93 s | yes |
| rev7 | 3 | 1 | 681.67 | `walk->stand` | 0.56 s | +1.98 s | NO |
| rev7 | 3 | 1 | 687.53 | `stand->walk` | 1.01 s | -1.64 s | yes |
| rev7 | 3 | 1 | 702.02 | `walk->stand` | 6.51 s | -1.13 s | yes |
| rev7 | 3 | 1 | 713.82 | `stand->walk` | 0.51 s | -2.68 s | NO |
| rev7 | 3 | 1 | 742.48 | `walk->stand` | 0.42 s | +5.42 s | NO |
| rev7 | 3 | 1 | 751.46 | `stand->walk` | 0.01 s | -1.57 s | yes |
| rev7 | 3 | 1 | 768.31 | `walk->stand` | 0.63 s | +0.58 s | yes |
| rev7 | 3 | 1 | 788.66 | `stand->walk` | 1.26 s | -1.51 s | yes |
| rev7 | 3 | 1 | 881.85 | `walk->stand` | 0.02 s | +0.80 s | yes |
| rev7 | 4 | 1 | 120.32 | `stand->walk` | 0.74 s | +0.24 s | yes |
| rev7 | 4 | 1 | 166.09 | `walk->stand` | 0.46 s | +0.23 s | yes |
| rev7 | 4 | 1 | 184.23 | `stand->walk` | 0.38 s | -0.67 s | yes |
| rev7 | 5 | 1 | 298.81 | `stand->walk` | 1.10 s | -1.09 s | yes |
| rev7 | 5 | 1 | 456.38 | `walk->stand` | 0.01 s | +0.83 s | yes |
| rev7 | 5 | 1 | 459.77 | `stand->walk` | 0.01 s | -0.81 s | yes |
| rev7 | 5 | 1 | 520.73 | `walk->stand` | 0.14 s | +1.99 s | yes |
| rev7 | 5 | 1 | 551.73 | `stand->walk` | 0.07 s | -1.52 s | yes |
| rev7 | 5 | 1 | 588.59 | `walk->stand` | 0.65 s | +1.12 s | yes |
| rev7 | 5 | 1 | 592.04 | `stand->walk` | 0.47 s | -1.58 s | yes |
| rev7 | 5 | 1 | 596.96 | `walk->stand` | 0.36 s | +1.25 s | yes |
| rev7 | 5 | 1 | 602.81 | `stand->walk` | 0.08 s | -0.84 s | yes |
| rev7 | 5 | 1 | 651.05 | `walk->stand` | 0.95 s | +0.91 s | yes |
| rev7 | 5 | 1 | 654.68 | `stand->walk` | 0.28 s | -1.22 s | yes |
| rev7 | 5 | 1 | 661.64 | `walk->stand` | 0.56 s | +0.57 s | yes |
| rev7 | 5 | 1 | 664.94 | `stand->walk` | 0.95 s | -1.48 s | yes |
| rev7 | 5 | 1 | 704.20 | `walk->stand` | 0.19 s | +0.76 s | yes |
| rev7 | 5 | 1 | 708.41 | `stand->walk` | 0.01 s | -0.95 s | yes |
| rev7 | 5 | 1 | 761.43 | `walk->stand` | 0.09 s | +0.28 s | yes |
| rev7 | 5 | 1 | 764.21 | `stand->walk` | 0.06 s | -0.49 s | yes |
| rev7 | 5 | 1 | 790.53 | `walk->stand` | 0.16 s | -0.07 s | yes |
| rev7 | 5 | 1 | 794.30 | `stand->walk` | 0.04 s | -1.58 s | yes |
| rev7 | 5 | 1 | 799.23 | `walk->stand` | 0.21 s | +0.73 s | yes |
| rev7 | 5 | 1 | 808.30 | `stand->walk` | 0.27 s | -2.09 s | NO |
| rev7 | 5 | 1 | 838.08 | `walk->stand` | 0.09 s | +0.38 s | yes |
| rev7 | 5 | 1 | 840.41 | `stand->walk` | 0.01 s | -0.70 s | yes |
| rev7 | 5 | 1 | 850.90 | `walk->stand` | 0.07 s | +10.31 s | NO |
| rev7 | 5 | 1 | 855.05 | `stand->walk` | 0.07 s | +6.16 s | NO |
| rev7 | 5 | 1 | 860.81 | `walk->stand` | 0.10 s | +0.41 s | yes |
| rev7 | 5 | 1 | 863.98 | `stand->walk` | 0.01 s | -1.52 s | yes |
| rev7 | 5 | 1 | 870.63 | `walk->stand` | 0.25 s | +0.58 s | yes |
| rev7 | 5 | 1 | 882.28 | `stand->walk` | 0.09 s | -0.32 s | yes |
| rev7 | 5 | 1 | 913.17 | `walk->stand` | 0.29 s | +0.79 s | yes |
| rev7 | 5 | 1 | 915.98 | `stand->walk` | 0.44 s | -1.26 s | yes |
| rev7 | 5 | 1 | 936.00 | `walk->stand` | 0.08 s | +0.71 s | yes |
| rev7 | 5 | 1 | 939.61 | `stand->walk` | 0.14 s | -0.90 s | yes |
| rev7 | 5 | 1 | 944.32 | `walk->stand` | 0.14 s | +0.65 s | yes |
| rev7 | 5 | 1 | 955.34 | `stand->walk` | 0.10 s | -3.12 s | NO |
| rev7 | 5 | 1 | 968.91 | `walk->stand` | 0.01 s | +1.05 s | yes |
| rev7 | 5 | 1 | 985.31 | `stand->walk` | 0.96 s | -1.34 s | yes |
| rev7 | 5 | 1 | 998.12 | `walk->stand` | 0.05 s | +0.59 s | yes |
| rev7 | 5 | 1 | 1001.36 | `stand->walk` | 0.18 s | -0.65 s | yes |
| rev7 | 5 | 1 | 1016.59 | `walk->stand` | 0.01 s | +0.37 s | yes |
| rev7 | 5 | 1 | 1021.12 | `stand->walk` | 0.01 s | -1.91 s | yes |
| rev7 | 5 | 1 | 1034.97 | `walk->stand` | 0.72 s | +1.25 s | yes |
| rev7 | 5 | 1 | 1045.80 | `stand->walk` | 0.09 s | -6.09 s | NO |
| rev7 | 5 | 1 | 1052.47 | `walk->stand` | 0.32 s | +0.99 s | yes |
| rev7 | 6 | 1 | 64.97 | `stand->walk` | 0.01 s | -0.44 s | yes |
| rev7 | 6 | 1 | 67.87 | `walk->stand` | 0.10 s | +0.17 s | yes |
| rev7 | 6 | 1 | 94.73 | `stand->walk` | 0.01 s | -0.20 s | yes |
| rev7 | 6 | 1 | 281.62 | `walk->stand` | 0.01 s | +1.16 s | yes |
| rev7 | 6 | 1 | 285.49 | `stand->walk` | 0.18 s | -0.95 s | yes |
| rev7 | 6 | 1 | 425.86 | `walk->stand` | 0.07 s | -0.33 s | yes |
| rev7 | 6 | 1 | 429.36 | `stand->walk` | 0.12 s | -0.57 s | yes |
| rev7 | 6 | 1 | 476.40 | `walk->stand` | 0.04 s | +0.14 s | yes |