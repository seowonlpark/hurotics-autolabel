# Label audit (physics vs annotation)

Two independent trial-level failures, both scored against the swap rule (§10), which has no trained parameter and never sees a label.

- **`disagree`** — share of label-pure, physics-decided windows whose verdict contradicts the annotation. Flags above **0.50** over at least 20 windows: the file contradicts itself more often than it agrees. That is broken data — a swapped channel or a misaligned label track.
- **`band walk`** — share of the trial's windows *inside the ambiguity band* (**2.62–20.34°** interleg swing, where both human labels genuinely occur) that it annotated `walk`. The corpus sits at **0.872**; a trial far from it is not broken, it is annotated to a different convention. Flagged by a two-sided binomial test at 0.05 Bonferroni-corrected across trials with at least 20 band windows.

The second is the one that costs accuracy quietly: the model learns the corpus convention, so a divergent trial reads as model error and no amount of retraining removes it.

## Broken (`disagree`)

| rev | trial | windows | disagree | fraction |
|---|---|---|---|---|
| rev13 | 4 | 62 | 59 | **0.9516** |

Add confirmed cases to `dataset.EXCLUDED_TRIALS` with the evidence.

## Nominated windows — run these to adjudicate

12 window(s) across 1 rev(s), strongest contradiction first, spread at least 5 s apart so a reader can tell a swapped channel (wrong everywhere) from a misaligned label track (wrong in a stretch). Capped at 12 per trial; **47 further contradicting window(s) not listed**.

A nomination is *not* a verdict. The swap rule can be wrong about a window, which is exactly why this hands you the raw trace instead of a decision.

| rev | trial | t (s) | label | physics | swaps | interleg | command |
|---|---|---|---|---|---|---|---|
| rev13 | 4 | 1523.91 | `stand` | `WALKING` | 5 over 2.8 s | 49.91° | `python -m stages.s3_physics.inspect_window rev13 4 --t 1523.91` |
| rev13 | 4 | 1507.91 | `stand` | `WALKING` | 4 over 3.7 s | 45.84° | `python -m stages.s3_physics.inspect_window rev13 4 --t 1507.91` |
| rev13 | 4 | 1515.91 | `stand` | `WALKING` | 4 over 3.2 s | 56.15° | `python -m stages.s3_physics.inspect_window rev13 4 --t 1515.91` |
| rev13 | 4 | 1541.91 | `stand` | `WALKING` | 4 over 3.0 s | 50.96° | `python -m stages.s3_physics.inspect_window rev13 4 --t 1541.91` |
| rev13 | 4 | 1529.91 | `stand` | `WALKING` | 4 over 2.9 s | 61.25° | `python -m stages.s3_physics.inspect_window rev13 4 --t 1529.91` |
| rev13 | 4 | 1535.91 | `stand` | `WALKING` | 4 over 3.1 s | 51.54° | `python -m stages.s3_physics.inspect_window rev13 4 --t 1535.91` |
| rev13 | 4 | 1549.91 | `stand` | `WALKING` | 4 over 3.2 s | 45.48° | `python -m stages.s3_physics.inspect_window rev13 4 --t 1549.91` |
| rev13 | 4 | 1555.91 | `stand` | `WALKING` | 4 over 3.1 s | 56.67° | `python -m stages.s3_physics.inspect_window rev13 4 --t 1555.91` |
| rev13 | 4 | 1589.91 | `stand` | `WALKING` | 4 over 3.1 s | 58.24° | `python -m stages.s3_physics.inspect_window rev13 4 --t 1589.91` |
| rev13 | 4 | 1561.91 | `stand` | `WALKING` | 4 over 3.0 s | 54.97° | `python -m stages.s3_physics.inspect_window rev13 4 --t 1561.91` |
| rev13 | 4 | 1567.91 | `stand` | `WALKING` | 4 over 3.3 s | 46.76° | `python -m stages.s3_physics.inspect_window rev13 4 --t 1567.91` |
| rev13 | 4 | 1573.91 | `stand` | `WALKING` | 4 over 3.1 s | 48.51° | `python -m stages.s3_physics.inspect_window rev13 4 --t 1573.91` |

Machine-readable copy: `label_audit_windows.jsonl`.

## Divergent convention (`band walk`)

| rev | trial | band windows | band walk | vs corpus | p |
|---|---|---|---|---|---|
| rev6 | 2 | 22 | **0.318** | -0.554 | 2.8e-09 |
| rev8 | 4 | 52 | **0.692** | -0.180 | 0.00056 |
| rev7 | 5 | 135 | **0.719** | -0.153 | 1.9e-06 |
| rev6 | 1 | 170 | **0.776** | -0.095 | 0.00051 |
| rev6 | 4 | 139 | **0.777** | -0.095 | 0.002 |
| rev2 | 4 | 79 | **0.987** | +0.115 | 0.00053 |
| rev2 | 5 | 133 | **1.000** | +0.128 | 2.6e-08 |
| rev2 | 3 | 83 | **1.000** | +0.128 | 2e-05 |

**Do not exclude these.** A self-consistent minority convention is not bad data; it is the part of the residual error that is annotation policy rather than model failure. Re-annotating them to the corpus convention — or accepting them and reporting the ceiling — are both defensible. Silently counting them as model error is not.

## All trials, worst first

| rev | trial | windows | disagree | physics abstains | band n | band walk | rest |
|---|---|---|---|---|---|---|---|
| rev13 | 4 | 62 | 0.9516 | 0.0159 | — | — | yes |
| rev5 | 3 | 10 | 0.3000 | 0.0909 | 2 | 0.000 | NO |
| rev5 | 2 | 5 | 0.2000 | 0.1667 | 2 | 0.500 | NO |
| rev5 | 4 | 15 | 0.2000 | 0.0625 | 4 | 0.000 | NO |
| rev8 | 4 | 77 | 0.1429 | 0.0610 | 52 | 0.692 ⚑ | yes |
| rev6 | 2 | 68 | 0.1324 | 0.0933 | 22 | 0.318 ⚑ | yes |
| rev6 | 4 | 180 | 0.1000 | 0.1743 | 139 | 0.777 ⚑ | yes |
| rev7 | 5 | 332 | 0.0994 | 0.0292 | 135 | 0.719 ⚑ | yes |
| rev8 | 1 | 84 | 0.0833 | 0.0667 | 69 | 0.899 | yes |
| rev7 | 3 | 320 | 0.0719 | 0.1940 | 221 | 0.932 | yes |
| rev2 | 4 | 313 | 0.0703 | 0.1032 | 79 | 0.987 ⚑ | yes |
| rev6 | 6 | 191 | 0.0681 | 0.1435 | 154 | 0.857 | yes |
| rev8 | 3 | 16 | 0.0625 | 0.0000 | 8 | 1.000 | yes |
| rev6 | 3 | 322 | 0.0621 | 0.1274 | 202 | 0.891 | yes |
| rev7 | 1 | 17 | 0.0588 | 0.1053 | 1 | 0.000 | yes |
| rev5 | 6 | 18 | 0.0556 | 0.0000 | 5 | 0.800 | NO |
| rev7 | 6 | 205 | 0.0439 | 0.0376 | 143 | 0.944 | yes |
| rev3 | 1 | 139 | 0.0360 | 0.0071 | 6 | 0.333 | yes |
| rev13 | 5 | 32 | 0.0312 | 0.0000 | 1 | 0.000 | yes |
| rev6 | 1 | 375 | 0.0293 | 0.0183 | 170 | 0.776 ⚑ | yes |
| rev4 | 1 | 85 | 0.0235 | 0.0449 | 54 | 0.870 | yes |
| rev2 | 7 | 49 | 0.0204 | 0.0392 | 42 | 0.857 | yes |
| rev2 | 8 | 126 | 0.0159 | 0.0667 | 66 | 0.955 | yes |
| rev13 | 2 | 85 | 0.0118 | 0.0116 | 2 | 0.000 | yes |
| rev2 | 1 | 219 | 0.0091 | 0.0090 | 38 | 0.974 | yes |
| rev2 | 2 | 113 | 0.0088 | 0.0813 | 35 | 1.000 | yes |
| rev2 | 5 | 331 | 0.0060 | 0.0322 | 133 | 1.000 ⚑ | yes |
| rev2 | 3 | 335 | 0.0060 | 0.0205 | 83 | 1.000 ⚑ | yes |
| rev13 | 1 | 659 | 0.0000 | 0.0000 | 23 | 1.000 | yes |
| rev13 | 6 | 163 | 0.0000 | 0.0061 | 3 | 0.333 | yes |
| rev13 | 7 | 150 | 0.0000 | 0.0000 | — | — | yes |
| rev13 | 3 | 81 | 0.0000 | 0.0000 | 2 | 0.000 | yes |
| rev2 | 6 | 187 | 0.0000 | 0.0053 | 137 | 0.912 | yes |
| rev5 | 5 | 35 | 0.0000 | 0.0000 | 5 | 1.000 | yes |
| rev5 | 1 | 8 | 0.0000 | 0.0000 | 2 | 0.500 | yes |
| rev3 | 3 | 263 | 0.0000 | 0.0000 | 1 | 1.000 | yes |
| rev3 | 2 | 125 | 0.0000 | 0.0000 | 1 | 1.000 | yes |
| rev5 | 8 | 27 | 0.0000 | 0.0000 | 4 | 1.000 | NO |
| rev5 | 7 | 24 | 0.0000 | 0.0000 | 12 | 1.000 | yes |
| rev7 | 4 | 59 | 0.0000 | 0.0328 | 1 | 0.000 | yes |
| rev8 | 2 | 10 | 0.0000 | 0.0000 | 3 | 1.000 | NO |