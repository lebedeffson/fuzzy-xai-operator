# H10-C7R final report

## Status

```text
H10_C7R_NOT_SUPPORTED
Scientific result: NOT_SUPPORTED
```

Held-out: 40 incidents from
30 repositories.

- R5 Recall@20: 0.550000
- B_BM25 Recall@160: 0.775000
- R5 coverage: 1.000000
- Mean R5 search-space reduction:
  0.990383
- Mean B_BM25 search-space reduction:
  0.923068
- Repository-cluster 95% CI:
  [0.047706, 0.096248]

Contract-family inference is descriptive and did not affect this status.
The result concerns candidate-space reduction, not automatic root-cause
confirmation or repair.

Although the repository-cluster confidence interval for the reduction
difference is strictly positive, the primary recall condition failed.
Therefore the larger reduction cannot be interpreted as supported practical
search-space reduction at the registered recall level.
