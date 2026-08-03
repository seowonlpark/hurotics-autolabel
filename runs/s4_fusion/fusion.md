# S4 Fusion

- windows: **6,339** (5,984 label-pure, 355 unscoreable)
- **coverage 95.9% at confident accuracy 0.9805** (target 95%)
- confident-but-wrong: **112** windows
- accuracy if forced to call everything: 0.9622
- accuracy on the windows it abstained from: 0.5403 — below the confident number, which is what makes the abstention informative rather than arbitrary
- transition windows abstained: **19.2%** (these are genuinely mixed; abstaining is the correct behaviour)

Coverage and confident accuracy are reported together on purpose. Either alone is meaningless: abstain on all but the easiest window and accuracy reads 1.000.

## By confidence tier

| tier | windows | accuracy |
|---|---|---|
| `high` | 3,893 | 0.9972 |
| `medium` | 1,843 | 0.9452 |
| `low` | 248 | 0.5403 |

`low` is the abstention tier. Its accuracy is *supposed* to be poor — those are the windows the pipeline declines to claim.

`high` additionally requires interleg swing **outside 2.62–20.34°** — the range where the human annotations themselves are not separable (below it 99% of annotated walking sits above; above it 99% of annotated standing sits below). Inside that band both labels genuinely occur, so agreement between the model and the physics is not evidence there: both read interleg amplitude. The cap moves windows to `medium`, never to `low`, so **it cannot change coverage** — it only stops `high` from claiming what it cannot support.

## Operating points

Every row clears the 95% target, so the choice is how much coverage a point of accuracy is worth — a judgement, not an optimum. Maximising coverage subject to the target alone would drive the floor to its minimum and produce a policy that ignores the model's probability entirely.

| policy | coverage | confident accuracy | confident errors |
|---|---|---|---|
| floor 0.50 | 96.7% | 0.9777 | 129 |
| floor 0.60 | 96.3% | 0.9788 | 122 |
| floor 0.70 ← shipped | 95.9% | 0.9805 | 112 |
| floor 0.80 | 95.1% | 0.9823 | 101 |
| floor 0.90 | 94.2% | 0.9832 | 95 |
| HIGH tier only (agree + outside band) | 65.8% | 0.9964 | 14 |
| no abstention (call everything) | 100.0% | 0.9622 | 226 |

The shipped floor is **0.70**, set in `DEFAULT_PROBA_FLOOR` (`stages/s4_fusion/fuse.py`). It is a deliberate point on this curve: it keeps the confident set well clear of the target while still claiming most windows. Move it knowingly — and re-run this stage, which re-measures the whole curve, rather than trusting the number in the source.