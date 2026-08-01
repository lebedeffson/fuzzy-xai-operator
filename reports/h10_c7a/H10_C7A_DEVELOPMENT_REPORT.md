# H10-C7A development report

## Status

```text
H10_C7A_BLOCKED_DEVELOPMENT_GATE
Scientific result: NOT_EVALUATED
Held-out created/scored: false/false
```

The original H10-C7a endpoint was evaluated on the completed open development
set of 40 incidents from 16 repositories. The frozen R5 prefix and all five
registered open-replay metrics were reproduced exactly. No retrieval,
candidate ranking, or contract-inference rule was changed.

## Matched-recall result

The search-space endpoint passed on development:

| Method | Frozen budget | Recall | Mean search-space reduction |
| --- | ---: | ---: | ---: |
| R5 | 20 | 0.9000 | 0.9680 |
| B_BM25 | 160 | 0.8750 | 0.8089 |

R5 reached the registered recall level with one eighth of the requested
candidate budget. This is an engineering development result, not a
confirmatory scientific result.

## Development gates

| Gate | Result |
| --- | --- |
| Incidents >= 40 | PASS (40) |
| Repositories >= 10 | PASS (16) |
| Recall@10 >= 0.75 | PASS (0.7750) |
| Recall@20 >= 0.85 | PASS (0.9000) |
| Contract macro-F1 >= 0.55 | **FAIL (0.5360)** |
| Coverage >= 0.80 | PASS (1.0000) |
| False localization <= B_GREEDY | PASS |
| Candidate symbols <= 20 | PASS |
| Structurally distinct top-k | PASS |
| Gold leakage = 0 | PASS |
| Matched-recall endpoint | PASS |

Because one preregistered development gate failed, no method lock, budget
lock, baseline lock, or confirmatory held-out set was created.

## Interpretation

The extension supports the narrower engineering observation that R5 reduces
the candidate search space at matched recall on the open development data.
It does not authorize the H10-C7a scientific claim. The remaining blocking
condition is contract-family inference on repository-disjoint development
incidents, not retrieval recall or candidate budget.

The blocked result must not be repaired by lowering the contract macro-F1
threshold, changing the frozen R5 method, or selecting different disclosed
incidents after observing these metrics.
