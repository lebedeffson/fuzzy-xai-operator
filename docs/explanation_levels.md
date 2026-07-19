# Explanation levels E0-E5

Levels disclose evidence depth, not model quality.

| Level | Available evidence |
|---|---|
| E0 | prediction and call trace |
| E1 | data profile and perturbation/data evidence |
| E2 | native contributions, native rules, or labelled surrogate rules |
| E3 | class concepts, prototypes, or similar cases |
| E4 | observed training history, checkpoints, or forgetting evidence |
| E5 | operator alignment, reduction, risk, counterfactuals, and audit route |

Every result lists `available_channels`, `missing_channels`, `native_channels`, and `surrogate_channels`. A callable model may reach E3 through reference data and similar cases while still disclosing that internal rules and training history are unavailable.
