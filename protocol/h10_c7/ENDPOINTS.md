# H10-C7 endpoints

## H10-C7a primary

`search_space_reduction_at_recall_20_ge_0_80`

Support requires Recall@20 at least 0.80, coverage at least 0.80, false
localization no worse than the best baseline, and a repository-cluster 95%
confidence interval strictly above zero for the reduction difference.

## H10-C7b secondary

Recall@10 after at most three registered evidence requests minus Recall@10
before requests must be at least 0.10. Median request count must not exceed two
and no request may create a critical violation.

## H10-C7c secondary

Expressible repairs require simultaneous Fail-to-Pass, registered regression,
route recertification, zero new critical violations and successful rollback.
Arbitrary source-code defects are localization-only and are not counted as
repairable failures.
