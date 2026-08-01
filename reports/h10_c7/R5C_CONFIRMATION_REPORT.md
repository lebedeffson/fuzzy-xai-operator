# H10-C7-R5C calibrated confirmation

Status: `H10_C7_OPEN_REPLAY_NO_GO`

Scientific result: `NOT_EVALUATED`.

R5 retrieval and the 30 disclosed replay incidents were unchanged.
All confirmation decisions are leave-one-repository-out predictions.

| Metric | Value |
| --- | ---: |
| Confirmed | 0 |
| Confirmed correct | 0 |
| Confirmed false | 0 |
| Selective precision | 1.0000 |
| Confirmation coverage | 0.0000 |

## Model comparison

| Model | Confirmed | Correct | False | Selective precision | Coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| C0 | 0 | 0 | 0 | 1.0000 | 0.0000 |
| C1 | 7 | 0 | 7 | 0.0000 | 0.2333 |

C0 was selected by the locked lexicographic rule because it preserved the
required precision by failing closed. C1 did not transfer safely across
excluded repositories.

The result is an open-development engineering gate, not scientific support for H10-C7.
