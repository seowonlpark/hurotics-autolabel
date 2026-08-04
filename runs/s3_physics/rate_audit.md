# S3 rate-invariance audit

## Rate-invariance audit

Every window decimated 100 Hz -> 50 Hz through S1's anti-aliasing FIR, anchors recomputed, tolerance 0.1. Gated to the 3,841 windows the swap rule calls WALKING (of 6,339) — standing has no cadence to compare, and the gate keeps the audit label-free.

Each window is decimated with **128 samples of real context** either side, then trimmed back: on a bare window the filter's edge transient covers over half the samples and reads as a rate dependence that is not there (it condemned `antiphase` at Δ 0.1734 vs 0.0005 corrected). 10 windows sat too close to a segment boundary to pad on both sides and were skipped; 3,831 were audited.

| anchor | metric | median Δ | p90 Δ | verdict |
|---|---|---|---|---|
| `periodicity` | abs | 0.0273 | 0.0324 | **invariant** |
| `antiphase` | abs | 0.0005 | 0.0024 | **invariant** |
| `grav_stab` | abs | 0.0001 | 0.0003 | **invariant** |
| `gyro_energy` | rel | 0.4994 | 0.5017 | **rate_dependent** |

## Reading a failure

**An anchor that fails is not a bug.** `gyro_energy` is defined the way that fails, on purpose, as this audit's negative control: it sums over samples, so halving the sample count halves it — a claim about the sampling grid, exactly what §2.3 warns about. An audit that has never rejected anything is not evidence that the rest passed (§11.1).

This run: **1 of 4** anchors rate-dependent (`gyro_energy`). The negative control fired, so the passes mean something.
