# Active task: observer evidence binding (developmental research)

Branch: `research/observer-evidence-binding-20261111`
Review: https://github.com/lebedeffson/fuzzy-xai-operator/pull/4
Base: `research/tisu-core-safety-20261108` (locked study artifacts untouched)

## Objective

Improve the actual FuzzyXAI observer before revising any publication or
dissertation chapters. The observer must distinguish missing evidence from
incompatible evidence and must not certify a local explanation intended for
another sample, model, target or ordered feature axis. It must not mistake
shape validity for explanation faithfulness.

## Implemented

- Typed local evidence status and identity/shape checks;
- opt-in practical guard integration ahead of budget allocation;
- independent linear-additive score-response verification from a trusted
  target-logit callable, including feature-wise reference probes;
- structural-only and source-response baseline comparisons;
- grouped OOF scaler moved inside each training fold;
- adversarial and held-out sklearn Breast Cancer/Wine/Digits tests;
- explicit scientific limitations and 3.11/3.12 focused GitHub CI.

## Validation

Earlier scoped identity/OOF revision at commit `376a9fd1` passed focused
Python 3.11 and 3.12 pytest and Ruff in GitHub Actions
(`43084123056`, `43084130742`).
Later additive-effect and real-model changes are **not yet fully CI-certified**:
GitHub Actions jobs were queued when this state was written.
No full regression or new confirmatory scientific claim is asserted.

## Next executable action

Inspect the newest `Observer evidence-binding regression` workflow on this
branch. Fix the exact failures, rerun; then inspect the relevant full
framework/practical controller checks. Reproduce and retain the real-model
per-fault counts. Only then mark the draft PR ready for review.
Do not modify protected dissertation, Q1/H10 records, or chapters.
