# Active task — Observer evidence binding (developmental)

Repository: lebedeffson/fuzzy-xai-operator
Branch: research/observer-evidence-binding-20261111
Draft review: https://github.com/lebedeffson/fuzzy-xai-operator/pull/4
Base: research/tisu-core-safety-20261108

## Scope and scientific constraints

Improve observer reliability before revising research papers/chapters.
Protect all locked/scored research and confirmatory evidence. Never claim
global XAI faithfulness, causal relevance, general SHAP/LIME guarantees,
independent expert evaluation, or preregistered confirmation from this
developmental benchmark.

## Implemented in PR

- Opt-in local evidence identity checks for model, object, target class,
  ordered feature axis, scope, finite contribution vector, source digest.
- Optional canonical numeric input hash at prediction, explanation and
  model-effect stages; stale input blocks even with the same object ID.
- Separate missing=>review and contradictory=>block semantics integrated
  with public practical_controller and budget allocation.
- Independent linear-additive target-logit probes, plus an O(d+2) row
  batched collector requiring only one source-model API invocation.
- Grouped OOF preprocessing isolated within every training fold.
- Real fitted sklearn logistic model benchmark on Breast Cancer/Wine/
  Digits with rehashed semantic errors, strong structure+logit-sum baseline,
  random zero-sum attribute corruption and negative interaction control.
- SHA256 fingerprint of actual fitted coefficients/intercept/class axis/
  scaler parameters, plus runtime environment and dataset metadata.
- Additional developmental random-seed stability tests.

## Verified reproducibility evidence

Observer evidence-binding run 43094340021, commit d7f5b40f:
70 pytest cases passed on Python 3.11 and 3.12, Ruff passed.
Its downloadable 3.11 artifact ID 13890656129 was inspected:
n=36 held-out examples, 504 cases; all 36 clean explanations accepted.
Four distinct sum-preserving mutation families produced 144/144
full-verifier blocks but 144/144 structural+sum baseline accepts.
Stale-input same-object substitutions: 36/36 blocked while simple
identity+hash baseline accepted 36/36; absent binding: 36/36 review.
Fitted-model maximum clean logit residual ~4.66e-15. The rows are not
independent observations: 36 source predictions over three fitted models.
This is deliberate fault injection, not external unknown-error testing.

Selective observer formative gate run 43094340097 succeeded;
Chapter 4 v13 smoke run 43094339949 succeeded.
Full readiness run 43094339994 was still in progress after installing
FastAPI, HTTPX and optional NiceGUI test/UI dependencies.
Final practical closure and H10 smoke workflows remain red from broader
existing Ruff issues in unrelated folders; do not silently hide/skip
them or alter protected scientific records to manufacture a green badge.

## Next actual executable step

Inspect the newest Observer evidence-binding regression workflow for the
latest branch SHA (including alternate seeds and nonlinear negative
control). Fix exact pytest/Ruff failures, inspect its JSON artifact.
Inspect full readiness and any further missing optional dependencies.
Update PR text with verified SHA/run IDs, keep draft until CI/review
gates are legitimately satisfied. Papers/chapters remain unchanged.
