# Typed F0 composition: controlled research validation (2026-11-16)

This is an **exploratory, synthetic fault-injection study**, not a sealed
confirmatory dissertation result, field-deployment validation, or clinical audit.

## Reproduction

```bash
python benchmarks/tisu_composition_validation.py --out-dir tisu_validation_outputs
python -m pytest -q tests/test_tisu_checked_composition.py
```

Source: `lebedeffson/fuzzy-xai-operator`, base revision
`aae7aa717bff913240cbb0026807cd1ccfa67160`. The checked operator is
a new opt-in research extension; the base `compose()` was not modified.

Dataset: scikit-learn Wisconsin Diagnostic Breast Cancer, binary malignant label
mapped to 1. RandomForestClassifier (120 estimators, maximum depth 6,
random_state=42); stratified train/test split 426/143, seed 42.
20 predefined controlled perturbations for each held-out prediction: 2,860
status comparisons (ACCEPT, REJECT, UNKNOWN). These repeated perturbations are
**not** 2,860 independent patients or naturally observed incidents.

| Decision procedure | Correct status / 2,860 |
| --- | ---: |
| Scalar threshold | 858 |
| Existing `compose()` | 722 |
| Typed interface without membership check | 2,438 |
| `compose_checked()` with declared piecewise-linear F0 membership check | 2,860 |

The expected status was specified by the perturbation protocol. Therefore this
agreement measures conformance to predeclared synthetic cases only and does
not establish unknown-fault detection accuracy, superiority to an equally
specified ordinary contract checker, or clinical effectiveness.

The typed-only comparator shares event/provenance/rule checks, so the difference
of 422 expected statuses comes from the additionally declared membership
checks. A deliberately narrow membership perturbation occurs between nodes of
the existing 101-point F0 grid; piecewise-linear knot-union comparison
identifies it. The method assumes honestly declared canonical event metadata
and an externally provided origin record. Neither record is authenticated,
and arbitrary callable membership functions are **rejected as insufficient
evidence (UNKNOWN)**: only a direct `MembershipSpec.value` binding establishes
the declared piecewise-linear form. The sup-norm check on knot unions is
exact only under that assumption.

Local verification: 95 focused checks passed; a smoke test exercised
RandomForestClassifier and LogisticRegression predictions. CI review and
wheel-only import smoke must still be checked before merger. The validated
scope is scalar binary F0 and unary risk rules, not all FuzzyXAI model or
uncertainty representation families.
