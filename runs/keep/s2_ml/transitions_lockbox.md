# Transitions - LOCKBOX (single use)

What the deliverable's biggest abstention bucket is actually made of. `near_transition` suppresses more rows than any other reason and its suppressed guess is the least accurate of any; until now nothing measured the boundaries it fires on (`archive/needtowrite.md` §5.4, gap 5).

**19 timeable transitions**, from 20 annotated stand/walk boundaries. Rejected: 1 short flank, 0 near segment edge.

A boundary is timeable when both sides hold at least 1 full window (2 s) of their own class and it sits at least 1 window from either end of its segment — below that the model has no pure window to place a change from, and scoring its timing would measure the window length. **The rejections are counted, not dropped.** Everything here encodes the raw `truth` column with the unknowns still in it, and reads `-1` as the boundary's width rather than removing it — filtering first splits one boundary into many.

## How many, and which way

| direction | transitions |
|---|---|
| `stand->walk` | 11 |
| `walk->stand` | 8 |

## How long is a transition

The annotator's own `-1` interval between the two runs — ground truth does not step instantaneously, and this is the width of its admitted doubt rather than a model of it. A width of 0 means the boundary was called to the sample.

| median | mean | p90 | max | called to the sample |
|---|---|---|---|---|
| 0.42 s | 0.69 s | 1.06 s | 5.43 s | 0 of 19 |

## Is the pipeline's timing right

Offset from each annotated boundary to the nearest state change the model predicted. **Positive is late.** Measured on the guess, not the committed label: where the model thinks the boundary is does not depend on whether it cleared the threshold on either side.

- transitions with a predicted change in the segment: **19 of 19**
- median offset: **-1.54 s** (7 late, 12 early)
- median |offset|: **2.58 s**, p90 21.73 s
- within ±1 s (the radius `near_transition` marks): **4 of 19** (21%)
- within ±2 s (one window): **7 of 19** (37%)

## Does `near_transition` fire where the boundaries are

**7 of 19 (37%)** annotated transitions have a `near_transition` row within ±1 s. That is the flag's recall.

Its precision, from the other side: the 4,175 flagged rows form **19 contiguous regions**, of which **9 (47%)** contain no annotated transition within ±2 s of their span.

Regions, not rows: a stretch of flagged rows is one event however wide it is, and a row count says nothing about how many boundaries the flag claims. An unmatched region is not necessarily wrong — the model may be changing class somewhere the annotation does not — but it is the flag firing where ground truth sees no boundary, which is what its 0.60 suppressed-guess accuracy has to be read against.

## Per subject

| rev | transitions | median width | median &#124;offset&#124; | within ±1 s | flagged |
|---|---|---|---|---|---|
| `rev8` | 19 | 0.42 s | 2.58 s | 21% | 37% |

## Every timeable transition

| rev | trial | seg | t (s) | direction | width | offset | flagged |
|---|---|---|---|---|---|---|---|
| rev8 | 1 | 1 | 205.68 | `stand->walk` | 0.15 s | -0.32 s | yes |
| rev8 | 1 | 1 | 250.92 | `walk->stand` | 0.01 s | -33.06 s | NO |
| rev8 | 1 | 1 | 253.19 | `stand->walk` | 0.01 s | -35.33 s | NO |
| rev8 | 1 | 1 | 322.30 | `walk->stand` | 0.15 s | +0.31 s | yes |
| rev8 | 1 | 1 | 335.65 | `stand->walk` | 1.57 s | -0.79 s | yes |
| rev8 | 1 | 1 | 353.76 | `walk->stand` | 0.01 s | -18.90 s | NO |
| rev8 | 1 | 1 | 360.01 | `stand->walk` | 0.01 s | +18.10 s | NO |
| rev8 | 1 | 1 | 376.28 | `walk->stand` | 0.44 s | +1.83 s | yes |
| rev8 | 1 | 1 | 384.16 | `stand->walk` | 0.30 s | -1.54 s | yes |
| rev8 | 3 | 1 | 962.03 | `stand->walk` | 5.43 s | -3.12 s | NO |
| rev8 | 4 | 1 | 1003.67 | `stand->walk` | 0.09 s | -4.51 s | NO |
| rev8 | 4 | 1 | 1027.57 | `walk->stand` | 0.83 s | +1.59 s | yes |
| rev8 | 4 | 1 | 1035.74 | `stand->walk` | 0.42 s | -2.58 s | NO |
| rev8 | 4 | 1 | 1056.28 | `walk->stand` | 0.34 s | +11.63 s | NO |
| rev8 | 4 | 1 | 1073.20 | `stand->walk` | 0.56 s | -4.29 s | NO |
| rev8 | 4 | 1 | 1094.27 | `walk->stand` | 0.43 s | +0.89 s | yes |
| rev8 | 4 | 1 | 1114.28 | `stand->walk` | 0.57 s | -3.12 s | NO |
| rev8 | 4 | 1 | 1148.71 | `stand->walk` | 0.86 s | -2.54 s | NO |
| rev8 | 4 | 1 | 1166.77 | `walk->stand` | 0.93 s | +2.14 s | NO |