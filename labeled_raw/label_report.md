# S2 Label — corpus sweep

- generated: **2026-08-04T06:03:22.891275+00:00**
- model: `runs/s2_ml`   threshold: **0.85**
- raw files: **91**, labelled: **78**, abstained: **13**
- **partition (gate): 78 labelled + 13 abstained = 91 of 91 raw files, 0 unaccounted**
- rows labelled: **2,594,823**, confident: **2,209,944** (85.2%)

Coverage and accuracy are a PAIR (`caveats.md` §2.5): this table reports coverage only, since no
label is read here. What fraction of rows clear the threshold is a fact about the
corpus; whether those calls are right is measured in `OPERATING_POINTS.md`.

## Servable, by variant

| variant | sagittal axis | files |
|---|---|---|
| `fb5ea2c2` | Deg_Y | 62 |
| `0fda484e` | Deg_Y | 11 |
| `4bfd6ab2` | Deg_Y | 5 |

## Abstentions

### UnknownVariantError — 13 file(s)

no measured sagittal-axis mapping for variant 'e5f2660f'. Known: ['0fda484e', '4bfd6ab2', 'fb5ea2c2']. Signal-based detection scores below chance (DOMAIN_NOTES 6.2) — add a mapping from one paired raw+annotated recording instead of guessing.

- `data\raw\20260515\00095_70_2026_5_15_11_38_0.csv`
- `data\raw\20260520\00220_100_2026_5_20_13_33_0.csv`
- `data\raw\20260520\00221_100_2026_5_20_13_35_0.csv`
- `data\raw\20260520\00222_100_2026_5_20_13_37_0.csv`
- `data\raw\20260520\00225_100_2026_5_20_13_45_0.csv`
- `data\raw\20260520\00226_100_2026_5_20_13_47_0.csv`
- `data\raw\20260522\00054_69_2026_5_22_11_27_0.csv`
- `data\raw\20260522\00055_69_2026_5_22_11_27_1.csv`
- `data\raw\20260522\00056_69_2026_5_22_11_49_0.csv`
- `data\raw\20260522\00057_69_2026_5_22_11_49_1.csv`
- `data\raw\20260522\00058_69_2026_5_22_11_49_2.csv`

no measured sagittal-axis mapping for variant '86069795'. Known: ['0fda484e', '4bfd6ab2', 'fb5ea2c2']. Signal-based detection scores below chance (DOMAIN_NOTES 6.2) — add a mapping from one paired raw+annotated recording instead of guessing.

- `data\raw\20260601\00192_100_2026_6_1_14_52_0.csv`
- `data\raw\20260601\00194_100_2026_6_1_15_10_0.csv`

## Dominant ambiguity reason, by file

| reason | files where it dominates |
|---|---|
| `near_transition` | 50 |
| `model_split` | 18 |
| `uncovered` | 6 |
| `out_of_distribution` | 2 |
| `weight_shift_or_step` | 1 |

## Plausibility — does each file's output hold together?

File-level bounds from `s3_physics/plausibility.py`. These report; they never refuse. A flagged file still has its labels — it has them *and* a stated doubt.

**2 file(s) tripped a hard bound.**

### physics_loud — 2 file(s)

the swap rule finds sustained leg alternation across the file, but the model commits almost none of it to walking

- `data\raw\20251230\00026_69_2025_12_30_15_25_0.csv` — physics finds alternation in 57.6% of the file; model calls only 0.0% of its 3,325 committed rows walk (bound 5%)
- `data\raw\20260109\00043_69_2025_11_6_14_28_0.csv` — physics finds alternation in 81.0% of the file; model calls only 0.0% of its 1,750 committed rows walk (bound 5%)

**Remedy:** the recording is outside the model's distribution, or a channel it reads is degenerate while the interleg angle still swings — the dead-sensor signature.

### rest_untrusted — 10 file(s)

the recording never rests, so the interleg zero is a whole-recording median rather than a measured standing posture (§10.2)

- `data\raw\20251024\00322_63_2025_10_24_15_34_0.csv`
- `data\raw\20251029\00330_63_2025_10_29_13_45_0.csv`
- `data\raw\20260120\00060_69_2026_1_20_18_18_0.csv`
- `data\raw\20260121\00019_69_2026_1_21_14_20_0.csv`
- `data\raw\20260121\00077_69_2026_1_21_14_51_0.csv`
- `data\raw\20260128\00075_69_2026_1_28_14_26_0.csv`
- `data\raw\20260128\00077_69_2026_1_28_14_28_0.csv`
- `data\raw\20260128\00080_69_2026_1_28_14_52_0.csv`
- `data\raw\20260128\00082_69_2026_1_28_15_10_0.csv`
- `data\raw\20260128\00103_69_2026_1_28_13_38_0.csv`

**Remedy:** not a fault on its own. It makes every physics verdict in this file SOFT, so read any bound above as weaker evidence, not as a clean contradiction.
