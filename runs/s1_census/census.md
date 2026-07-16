# S1 Schema Census

- files: **88**
- header variants: **5**
- families: **raw_device**

## Variants

| variant | family | cols | files | sessions |
|---|---|---|---|---|
| `fb5ea2c2` | raw_device | 67 | 62 | 20251230_2, 20260107, 20260108, 20260109 +8 |
| `0fda484e` | raw_device | 67 | 11 | 20251024, 20251029, 20251226 |
| `e5f2660f` | raw_device | 91 | 10 | 20260520, 20260522 |
| `4bfd6ab2` | raw_device | 83 | 3 | 20251021, 20251215 |
| `86069795` | raw_device | 79 | 2 | 20260601 |

## Stable prefix per family (the real contract)

### raw_device — 45 columns

```
Time, L_Deg_X, L_Deg_Y, L_Deg_Z, L_Gyro_X, L_Gyro_Y, L_Gyro_Z, L_Acc_X, L_Acc_Y, L_Acc_Z, R_Deg_X, R_Deg_Y, R_Deg_Z, R_Gyro_X, R_Gyro_Y, R_Gyro_Z, R_Acc_X, R_Acc_Y, R_Acc_Z, B_Deg_X, B_Deg_Y, B_Deg_Z, B_Gyro_X, B_Gyro_Y, B_Gyro_Z, B_Acc_X, B_Acc_Y, B_Acc_Z, L LC, R LC, Batt_V, C_A, L_GCP, R_GCP, L_Ref_Force, R_Ref_Force, Hip_Deg_L, Hip_Deg_R, Hip_ROM_L, Hip_ROM_R, Stride Velocity_L, Stride Velocity_R, Stride Length_L, Stride Length_R, Total Gait Length
```

## Names that change position within `raw_device`

- `Cadence`: positions [45, 46]
- `Step`: positions [46, 47]

Positional indexing is unsafe here. Resolve by name.
