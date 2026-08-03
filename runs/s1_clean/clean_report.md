# S1 Clean Report

- source files: **91**, written: **90**, failed: **1**
- **partition (gate): 90 clean + 1 quarantined = 91 of 91 raw files, 0 unaccounted**
- segments: **138** (93 usable, 45 dropped)
- canonical rate: **100.0 Hz**
- usable duration: **477.9 min**
- columns kept: **30 measured + 0 documented exceptions**
- format: **parquet**

Measured-only: every column that churns position between variants is a *computed* one,
so this collapses every schema variant into a single canonical shape.

## Gyro trust / normalization

Detected per file (not asserted): d(Deg_A)/dt regressed against every gyro axis, for
every Deg axis A, recovers the Deg->Gyro permutation and the unit. Every gyro channel
is normalized to deg/s. The r-floor is per axis, so an axis with no signal abstains
rather than contributing a noise argmax (Deg_Z drifts, so it often has none).

This resolves which gyro axis measures which angle axis. It does NOT resolve which axis
is SAGITTAL — that has no in-file signature (DOMAIN_NOTES 6.2) and is a variant lookup
in stages/s2_ml/transform.py.

- side-channels normalized rad/s -> deg/s: **90**
- sides that abstained (too static; fell back to documented convention): **56**
- confident anomalies (detected axis/unit disagree with the documented rule): **2**

- anomaly: `data\raw\20260114\00001_69_2026_1_14_10_4_0.csv` side B: conflicts=['Y'] map={'Y': 'Y', 'Z': 'Y'} unit=rad/s r=-0.9879 (documented: {'X': 'X', 'Y': 'Z', 'Z': 'Y'})
- anomaly: `data\raw\20260115\00038_69_2026_1_15_11_28_0.csv` side B: conflicts=['Y'] map={'Y': 'Y'} unit=rad/s r=0.9583 (documented: {'X': 'X', 'Y': 'Z', 'Z': 'Y'})

## Yaw / drift trust

A Deg channel whose value tracks session time is measuring integration drift, not
orientation (§4.2). Flagged per channel, not dropped — the raw superset is kept.

- Deg channels flagged drift-contaminated (|corr(Deg,Time)| >= 0.9): **14** across **12** files

- drift: `data\raw\20251024\00324_63_2025_10_24_15_56_0.csv` channel `L_Deg_Z`
- drift: `data\raw\20260108\00041_69_2026_1_8_17_19_0.csv` channel `R_Deg_Z`
- drift: `data\raw\20260109\00045_69_2025_11_6_14_40_0.csv` channel `B_Deg_Z`
- drift: `data\raw\20260109\00046_69_2025_11_6_14_52_0.csv` channel `L_Deg_Z`
- drift: `data\raw\20260109\00046_69_2025_11_6_14_52_0.csv` channel `B_Deg_Z`
- drift: `data\raw\20260114\00008_69_2026_1_14_10_48_0.csv` channel `L_Deg_Z`
- drift: `data\raw\20260115\00038_69_2026_1_15_11_28_0.csv` channel `R_Deg_Z`
- drift: `data\raw\20260121\00020_69_2026_1_21_15_33_0.csv` channel `B_Deg_Z`
- drift: `data\raw\20260121\00073_69_2026_1_21_14_44_0.csv` channel `B_Deg_Z`
- drift: `data\raw\20260128\00078_69_2026_1_28_14_30_0.csv` channel `L_Deg_Z`
- drift: `data\raw\20260128\00084_69_2026_1_28_15_45_0.csv` channel `L_Deg_Z`
- drift: `data\raw\20260128\00101_69_2026_1_28_13_38_0.csv` channel `L_Deg_Z`
- drift: `data\raw\20260128\00101_69_2026_1_28_13_38_0.csv` channel `R_Deg_Z`
- drift: `data\raw\20260626\00156_92_2026_6_26_9_52_0.csv` channel `B_Deg_Z`

## Method

| method | segments |
|---|---|
| `grid_aligned` | 73 |
| `decimate_5x_fir` | 14 |
| `interp_to_grid` | 6 |

## Dropped segments

- `data\raw\20251021\00203_30_2025_10_21_18_18_0_edit.csv` seg 0: 0.090 s < 1.0 s (10 rows @ 100.0 Hz)
- `data\raw\20251024\00321_63_2025_10_24_15_32_0.csv` seg 0: 0.090 s < 1.0 s (10 rows @ 100.0 Hz)
- `data\raw\20251024\00322_63_2025_10_24_15_34_0.csv` seg 0: 0.088 s < 1.0 s (10 rows @ 100.0 Hz)
- `data\raw\20251024\00323_63_2025_10_24_15_35_0.csv` seg 0: 0.090 s < 1.0 s (10 rows @ 100.0 Hz)
- `data\raw\20251024\00324_63_2025_10_24_15_56_0.csv` seg 0: 0.092 s < 1.0 s (10 rows @ 100.0 Hz)
- `data\raw\20251024\00325_63_2025_10_24_15_56_0.csv` seg 0: 0.088 s < 1.0 s (10 rows @ 100.0 Hz)
- `data\raw\20251024\00326_63_2025_10_24_16_29_0.csv` seg 0: 0.090 s < 1.0 s (10 rows @ 100.0 Hz)
- `data\raw\20251029\00328_63_2025_10_29_13_45_0.csv` seg 0: 0.090 s < 1.0 s (10 rows @ 100.0 Hz)
- `data\raw\20251029\00330_63_2025_10_29_13_45_0.csv` seg 0: 0.090 s < 1.0 s (10 rows @ 100.0 Hz)
- `data\raw\20251029\00331_63_2025_10_29_13_45_0.csv` seg 0: 0.090 s < 1.0 s (10 rows @ 100.0 Hz)
- `data\raw\20251029\00332_63_2025_10_29_13_45_0.csv` seg 0: 0.090 s < 1.0 s (10 rows @ 100.0 Hz)
- `data\raw\20251215\00081_86_2025_12_12_14_54_0_edit.csv` seg 0: 0.090 s < 1.0 s (10 rows @ 100.0 Hz)
- `data\raw\20251215\00085_86_2025_12_12_17_13_0_edit.csv` seg 0: 0.090 s < 1.0 s (10 rows @ 100.0 Hz)
- `data\raw\20260107\00036_69_2026_1_7_16_52_0.csv` seg 0: 0.092 s < 1.0 s (10 rows @ 100.0 Hz)
- `data\raw\20260108\00038_69_2026_1_8_15_11_0.csv` seg 0: 0.490 s < 1.0 s (43 rows @ 100.0 Hz)
- `data\raw\20260108\00038_69_2026_1_8_15_11_0.csv` seg 1: 0.220 s < 1.0 s (19 rows @ 100.0 Hz)
- `data\raw\20260108\00038_69_2026_1_8_15_11_0.csv` seg 3: 0.460 s < 1.0 s (39 rows @ 100.0 Hz)
- `data\raw\20260108\00038_69_2026_1_8_15_11_0.csv` seg 4: 0.210 s < 1.0 s (19 rows @ 100.0 Hz)
- `data\raw\20260109\00046_69_2025_11_6_14_52_0.csv` seg 0: 0.090 s < 1.0 s (10 rows @ 100.0 Hz)
- `data\raw\20260109\00047_69_2025_11_6_16_46_0.csv` seg 0: 0.089 s < 1.0 s (10 rows @ 100.0 Hz)
- `data\raw\20260120\00057_69_2026_1_20_17_37_0.csv` seg 0: 0.089 s < 1.0 s (10 rows @ 100.0 Hz)
- `data\raw\20260120\00058_69_2026_1_20_18_18_0.csv` seg 0: 0.090 s < 1.0 s (10 rows @ 100.0 Hz)
- `data\raw\20260121\00078_69_2026_1_21_15_5_0.csv` seg 0: 0.090 s < 1.0 s (10 rows @ 100.0 Hz)
- `data\raw\20260126\00061_69_2026_1_26_10_52_0.csv` seg 0: 0.090 s < 1.0 s (10 rows @ 100.0 Hz)
- `data\raw\20260128\00075_69_2026_1_28_14_26_0.csv` seg 0: 0.090 s < 1.0 s (10 rows @ 100.0 Hz)
- `data\raw\20260128\00081_69_2026_1_28_15_10_0.csv` seg 0: 0.090 s < 1.0 s (10 rows @ 100.0 Hz)
- `data\raw\20260128\00084_69_2026_1_28_15_45_0.csv` seg 0: 0.090 s < 1.0 s (10 rows @ 100.0 Hz)
- `data\raw\20260128\00100_69_2026_1_28_13_32_0.csv` seg 0: 0.090 s < 1.0 s (10 rows @ 100.0 Hz)
- `data\raw\20260520\00220_100_2026_5_20_13_33_0.csv` seg 0: 0.004 s < 1.0 s (3 rows @ 500.0 Hz)
- `data\raw\20260520\00220_100_2026_5_20_13_33_0.csv` seg 1: 0.014 s < 1.0 s (7 rows @ 500.0 Hz)
- `data\raw\20260520\00221_100_2026_5_20_13_35_0.csv` seg 0: 0.002 s < 1.0 s (2 rows @ 500.0 Hz)
- `data\raw\20260520\00221_100_2026_5_20_13_35_0.csv` seg 1: 0.016 s < 1.0 s (8 rows @ 500.0 Hz)
- `data\raw\20260520\00222_100_2026_5_20_13_37_0.csv` seg 0: 0.020 s < 1.0 s (10 rows @ 500.0 Hz)
- `data\raw\20260520\00225_100_2026_5_20_13_45_0.csv` seg 0: 0.002 s < 1.0 s (2 rows @ 500.0 Hz)
- `data\raw\20260520\00225_100_2026_5_20_13_45_0.csv` seg 1: 0.014 s < 1.0 s (8 rows @ 500.0 Hz)
- `data\raw\20260520\00226_100_2026_5_20_13_47_0.csv` seg 0: 0.014 s < 1.0 s (7 rows @ 500.0 Hz)
- `data\raw\20260520\00226_100_2026_5_20_13_47_0.csv` seg 1: 0.004 s < 1.0 s (3 rows @ 500.0 Hz)
- `data\raw\20260522\00054_69_2026_5_22_11_27_0.csv` seg 0: 0.022 s < 1.0 s (11 rows @ 500.0 Hz)
- `data\raw\20260522\00056_69_2026_5_22_11_49_0.csv` seg 0: 0.016 s < 1.0 s (8 rows @ 500.0 Hz)
- `data\raw\20260522\00058_69_2026_5_22_11_49_2.csv` seg 0: 0.018 s < 1.0 s (10 rows @ 500.0 Hz)
- `data\raw\20260601\00192_100_2026_6_1_14_52_0.csv` seg 0: 0.010 s < 1.0 s (6 rows @ 500.0 Hz)
- `data\raw\20260601\00192_100_2026_6_1_14_52_0.csv` seg 1: 0.008 s < 1.0 s (4 rows @ 500.0 Hz)
- `data\raw\20260601\00194_100_2026_6_1_15_10_0.csv` seg 0: 0.016 s < 1.0 s (8 rows @ 500.0 Hz)
- `data\raw\20260626\00156_92_2026_6_26_9_52_0.csv` seg 0: rate 248.062 Hz fits no known family
- `data\raw\20260626\00157_92_2026_6_26_9_52_0.csv` seg 0: rate 248.062 Hz fits no known family

## Quarantined files (whole-file rejects -> quarantine.jsonl ledger)

- `data\raw\20260515\00095_70_2026_5_15_11_38_0.csv`: degenerate time base: median dt <= 0 (duplicate/backward timestamps) (needs_human)