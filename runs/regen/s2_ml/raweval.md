# Raw device path - end-to-end accuracy

Raw device CSVs labelled through `label_csv` at threshold **0.85** and scored against the human annotation of the same recording.

- paired recordings: **13** across **3** subjects (rev13, rev4, rev7)
- rows returned: 480,700  →  scorable: **474,773** (dropped 5,927 human `-1`, 0 with no annotated sample in range)

**The lockbox is not in this table.** `rev8`'s four pairs are refused in code (§7), so every subject here is a development subject. Read this against `roweval_loro`'s 0.9901, not against rev8's 0.9308.

| | |
|---|---|
| coverage | **0.9091** |
| selective accuracy | **0.9914** |
| worst subject | 0.9821 (`rev7`) |
| committed rows | 431,612 |
| errors kept | 3,695 |

## Against the transitive claim

The number this replaces: raw equals `lpf_view` (`verify_serve`), `lpf_view` scores X (`roweval_loro`), therefore raw scores X. Same three subjects, same held-out discipline, so the two rows below are comparable.

| route | rows | coverage | selective acc |
|---|---|---|---|
| `lpf_view` (roweval, these 3 revs) | 475,260 | 0.9088 | 0.9914 |
| **raw device (this file)** | 474,773 | **0.9091** | **0.9914** |

The two row sets are not identical — `roweval` scores the annotated export's grid and this scores the raw file's own rows — so exact equality is not the bar. A gap that changed the operating point would be.

## Per subject

Each labelled by a champion refit without it.

| rev | trials | rows | coverage | accuracy | errors | stand recall | walk recall |
|---|---|---|---|---|---|---|---|
| `rev13` | 6 | 234,752 | 98.9% | 0.9985 | 354 | 0.9363 (n=5,559) | 1.0000 (n=226,525) |
| `rev4` | 1 | 20,751 | 80.7% | 0.9955 | 75 | 0.9797 (n=3,700) | 1.0000 (n=13,050) |
| `rev7` | 6 | 219,270 | 83.4% | 0.9821 | 3,266 | 0.8719 (n=23,353) | 0.9983 (n=159,425) |

## Per hardware variant

The axis map is what the raw route adds over the `lpf_view` route, so it is the thing this measurement is really testing. A variant whose accuracy sits apart from the others is a mis-mapped sagittal axis, which is exactly the failure that shipped undetected until 2026-08-03 (`caveats.md` §5).

| variant | revs | rows | coverage | accuracy | errors |
|---|---|---|---|---|---|
| `0fda484e` | rev7 | 219,270 | 83.4% | 0.9821 | 3,266 |
| `4bfd6ab2` | rev4 | 20,751 | 80.7% | 0.9955 | 75 |
| `fb5ea2c2` | rev13 | 234,752 | 98.9% | 0.9985 | 354 |

## Direction of error

| truth → guess | rows |
|---|---|
| walk → walk | 398,725 |
| stand → stand | 29,192 |
| stand → walk ⚠ | 3,420 |
| walk → stand ⚠ | 275 |
