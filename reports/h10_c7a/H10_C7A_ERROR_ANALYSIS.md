# H10-C7A development error analysis

This report is descriptive only. It was written after the one development
scoring run and does not authorize a method, ontology, threshold, or incident
selection change under H10-C7A.

## Added incidents

The ten disclosed H10-C5b incidents contributed eight new repositories. R5
retrieved the Gold symbol within the frozen top-20 for eight of the ten
incidents. The two misses were:

- `sympy__sympy-15346`
- `pytest-dev__pytest-5221`

Contract-family inference was correct for six of the ten added incidents.
The four incorrect predictions were:

| Incident | Gold | Predicted |
| --- | --- | --- |
| `django__django-16229` | `DATA_CONTRACT` | `CONFIGURATION` |
| `pydata__xarray-3364` | `CONFIGURATION` | `DATA_CONTRACT` |
| `pallets__flask-4992` | `SERIALIZATION` | `CONFIGURATION` |
| `matplotlib__matplotlib-23913` | `CONFIGURATION` | `UNKNOWN_CONTRACT` |
The pytest contract prediction is correct, but its Gold symbol is absent from
the top-20. Across all 40 incidents the resulting contract macro-F1 is
`0.5360451977`, below the locked `0.55` development gate.

## Boundary

The matched-recall retrieval endpoint passed despite the contract gate
failure. Therefore the observed blocker is not evidence that R5 failed to
reduce the candidate space. It is evidence that the complete preregistered
development gate, which includes independent contract-family quality, was not
met. The confirmatory cycle remains unopened.
