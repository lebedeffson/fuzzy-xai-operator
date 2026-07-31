# H10-C7R R10M error analysis

## Boundary

This is development-only analysis on 40 disclosed incidents. No new held-out set was created or scored.

## Stage decomposition

| Stage | Incidents |
|---|---:|
| FINAL_RERANK_MISS | 14 |
| SYMBOL_POOL_MISS | 5 |
| TOP20_HIT | 21 |

## Interpretation

R10M retained the correct file at rank 10 in 38/40 incidents, but the symbol pool retained a Gold symbol in only 35/40 and the final top-20 retained it in 21/40. The model contour therefore improved file retrieval but did not solve repository-independent symbol compression. No threshold, model, channel, or budget was changed after scoring.
