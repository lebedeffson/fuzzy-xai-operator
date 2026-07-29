# H10-C7 Open Replay Report

## Boundary

The replay uses the 30 disclosed H10-C5c development incidents from eight
repositories. It does not create or score held-out data and is not a scientific
result. H10-C5b and H10-C5c remain unchanged.

The source runtime artifact has SHA256
`7b7bd0bba2eb9eef3955d2b5e313ecf4bccd02703427117f741c382e7658db09`.
The autonomous bundle requires no network, project setup, dependency
installation, or failing-test re-execution.

## Baseline Replay

The fail-closed verifier reproduced the locked H10-C5c values:

| Metric | O_ROUTE |
| --- | ---: |
| Recall@10 | 0.5666666667 |
| Recall@20 | 0.5666666667 |
| Contract accuracy | 0.1333333333 |
| Published coverage | 0.8000000000 |
| Joint Hit@3 | 0.0000000000 |

The locked status reports pre-threshold coverage `0.8`; the immutable final
per-incident CSV represents post-threshold coverage as `0.7666666667`. Both
representations are retained and the discrepancy is explicit in
`BASELINE_REPLAY_STATUS.json`.

## Structural Tournament

Only `R0`, `R1`, `R3`, `R5`, and `R6` ran. Neural encoders and the cross-encoder
were not executed.

| Metric | H10-C5c O_ROUTE | Best structural R3 |
| --- | ---: | ---: |
| Recall@10 | 0.5667 | 0.7000 |
| Recall@20 | 0.5667 | 0.8333 |
| Contract score | 0.1333 accuracy | 0.4070 macro-F1 |
| Coverage | 0.8000 | 1.0000 |
| False localization | 0.7667 | 0.2333 |
| Joint Hit@3 | 0.0000 | 0.1667 |
| Median candidate symbols | not registered | 20 |

The contract metrics use different registered aggregations and are shown as
development diagnostics, not as a direct statistical comparison.

## Decision

Status: `H10_C7_OPEN_REPLAY_NO_GO`.

All incident-level metric thresholds passed for R3, but Recall@10 improved in
only four of eight repositories. The registered internal criterion requires at
least six. Therefore:

- no new development incidents are collected;
- neural variants are not promoted;
- no method lock is created;
- held-out data are not created or scored;
- the scientific result remains `NOT_EVALUATED`.

The remaining error counts for R3 are:

| Error class | Count |
| --- | ---: |
| Candidate absent from top 20 | 5 |
| Candidate ranked 11-20 | 4 |
| Contract inference miss | 14 |
| Joint localization miss | 2 |
| Match | 5 |

The next engineering work, if resumed, must improve repository-level
transfer on this open replay without weakening the six-of-eight criterion.
