# H10-C7R R9 error analysis

## Boundary

H10-C7R-v1 remains `H10_C7R_NOT_SUPPORTED`. Its 40 disclosed incidents are
used here only as development data. R9 has scientific status `NOT_EVALUATED`;
no new held-out set was created or scored.

## Observed result

| Diagnostic stage | Incidents retained | Recall@20 |
|---|---:|---:|
| Broad union of structural channels | 39/40 | 0.975 |
| Fixed R9-A compressor | 20/40 | 0.500 |
| LORO LambdaMART R9-B | 15/40 | 0.375 |
| LORO-selected R9-A/R9-B | 16/40 | 0.400 |

The only incident absent from every retrieval channel was
`python__mypy-19290`. The candidate schema nevertheless contained at least
one registered diagnostic Gold atom for every incident.

## Interpretation

The schema and broad retrieval changes repaired the early candidate-loss
boundary: expanded node kinds and channel limits make a relevant candidate
available in 39 incidents. They did not solve repository-independent
compression to 20 symbols.

The compact LambdaMART model was trained and evaluated by
leave-one-repository-out folds. It underperformed the fixed R9-A rule outside
the training repositories. Therefore the available normalized channel ranks
and runtime-distance features do not support the registered Recall@20 gate
with reliable cross-repository transfer.

This is not evidence that a different threshold should be selected. The
registered gate remains unchanged and the development result is `NO_GO`.
Neural R9-C was implemented as a fail-closed optional route but was not
executed because locked local model weights were not part of this run.

## Next admissible work

Do not create a new held-out set. Any continuation requires new observable
candidate-specific signals or a separately preregistered compressor study.
The disclosed 40 incidents may remain an open engineering set; they cannot be
reused as confirmatory evidence.
