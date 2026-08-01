# H10-C7 open replay refactor report

Status: `H10_C7_OPEN_REPLAY_NO_GO`

Scientific result: `NOT_EVALUATED`.

This development-only replay reused the 30 disclosed H10-C5c incidents. It collected no new incidents, installed no project environment, executed no failing test, and used no neural model.

| Variant | Recall@10 | Recall@20 | MRR | Contract macro-F1 | Joint Hit@3 | Selective precision | Confirmation coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| R0 | 0.5667 | 0.5667 | 0.2909 | 0.6239 | 0.3667 | 0.4286 | 0.2333 |
| R1 | 0.4000 | 0.5000 | 0.1412 | 0.6239 | 0.1000 | 0.2500 | 0.1333 |
| R3 | 0.8333 | 0.9000 | 0.4947 | 0.6239 | 0.4667 | 0.8571 | 0.2333 |
| R5 | 0.8667 | 0.9333 | 0.5298 | 0.6239 | 0.4667 | 0.8571 | 0.2333 |
| R6 | 0.8667 | 0.9333 | 0.5065 | 0.6239 | 0.4667 | 0.8571 | 0.2333 |

## Interpretation

Retrieval and contract gates are engineering diagnostics, not a confirmatory scientific result. New data and neural variants remain blocked unless every registered open-replay gate passes.

`false_localization` remains the population-normalized error count for compatibility. `selective_precision` uses confirmed diagnoses as its denominator and is the primary safety interpretation for confirmation.
