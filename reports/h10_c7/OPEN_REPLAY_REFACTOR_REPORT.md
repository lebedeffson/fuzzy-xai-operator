# H10-C7 open replay refactor

## Boundary

This is a development-only engineering replay of 30 disclosed H10-C5c
incidents from eight repositories. It is not a new experiment or a scientific
H10-C7 result. The replay used the existing runtime events, executed slices,
source snapshots, repository graphs, assertions, and development Gold.

The run:

- collected no new incident;
- cloned no repository;
- installed no project dependency;
- reran no failing test;
- executed no neural model;
- opened no held-out data.

## Corrected implementation defects

- `IncidentNormalizer` removes collector, launcher, dependency-warning, and
  temporary-path noise before retrieval and contract inference.
- Identifier tokenization now splits snake case, CamelCase, dotted symbols,
  paths, and simple plural variants.
- `CandidateReservoir` unions exact, BM25, graph, runtime, and legacy R0
  channels before top-k truncation and retains up to 300 candidates.
- Runtime ranking uses directed caller/callee distance, execution frequency,
  last-touch proximity, and failing-test frequency.
- R5 ranks candidate-contract pairs over the wider pool before selecting the
  final 20 symbols.
- Contract inference separates incident hypotheses from candidate
  compatibility and introduces observable behavioral subfamilies mapped to
  the published ontology.
- R6 applies saved traceback observations, records rank and entropy before and
  after the probe, and reports unavailable evidence explicitly.
- Confirmation requires two independent direct contract observations.
- Metrics distinguish retrieval, contract, confirmation, and repair coverage.
  Selective precision uses confirmed diagnoses as its denominator.

## Result

| Metric | R0 | R3 | R5 |
| --- | ---: | ---: | ---: |
| Recall@10 | 0.5667 | 0.8333 | 0.8667 |
| Recall@20 | 0.5667 | 0.9000 | 0.9333 |
| MRR | 0.2909 | 0.4947 | 0.5298 |
| Contract macro-F1 | 0.6239 | 0.6239 | 0.6239 |
| Joint Hit@3 | 0.3667 | 0.4667 | 0.4667 |
| Selective precision | 0.4286 | 0.8571 | 0.8571 |
| Confirmation coverage | 0.2333 | 0.2333 | 0.2333 |

R5 improved Recall@10 in six of eight repositories, preserved PySnooper at
1.0, and reached FastAPI Recall@10 of 1.0. R3 and R5 passed the registered
retrieval, ranking, contract, joint-localization, repository, and selective
precision checks.

The confirmation coverage check failed: 0.2333 against the fixed minimum
0.40. Consequently:

```text
H10_C7_OPEN_REPLAY_NO_GO
Scientific result: NOT_EVALUATED
New development data: BLOCKED
Neural variants: NOT_EXECUTED
Held-out created/scored: false/false
```

The next implementation problem is calibrated confirmation coverage, not
candidate retrieval. The threshold must not be weakened after observing these
results.
