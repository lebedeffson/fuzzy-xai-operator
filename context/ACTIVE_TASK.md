# Active task: FuzzyXAI evidence-first observer — scientific engineering

Branch: research/observer-evidence-binding-20261111
Draft review: https://github.com/lebedeffson/fuzzy-xai-operator/pull/4
Protected base: research/tisu-core-safety-20261108
All locked/scored source evidence and publications are untouched.

## Objective

Strengthen source-bound local-explanation checking before revising the four articles
or any dissertation chapter. Distinguish incompatible declarations (BLOCK)
from missing evidence (REVIEW), and numeric validity from semantic fidelity.

## Implemented

- Opt-in object/model/target/local-scope/ordered-feature-axis attribution binding;
- canonical numeric input identity across prediction/explanation/model probes;
- SHA256 fingerprinting of actual fitted logistic model/scaler state;
- source-model linear-additive target-logit per-feature response checks;
- vectorized single-call model probes on d+2 rows, plus sequential parity tests;
- strict background/reference identity between explainer and source-model probes;
- original public practical controller propagation of REVIEW/BLOCK ahead of budgets;
- grouped OOF model normalization fitted within training folds only;
- deliberate rehashed source-fault, stale-input, background-substitution, zero-sum,
  nonlinear interaction and genuine native SHAP LinearExplainer test coverage;
- reproducibility metadata (seed, Python/sklearn/NumPy versions and dataset stats).

## Verified baseline and current uncertainty

Verified GitHub observer Actions run 43094340021 at commit d7f5b40f:
70 pytest + Ruff passed on Python 3.11 and 3.12. Downloaded artifact
13890656129: 36 held-out source objects, 504 developer-generated cases;
36 clean accepted, all 144 same-sum corruptions blocked by per-feature model
oracle while the strong structure+sum baseline accepted all 144; 36 stale-input
blocks and 36 missing-input reviews; max clean logit residual ~4.66e-15.
These cases are correlated, do not represent independent natural errors.

Added after the verified artifact: actual model fingerprints, extra random-seed
stability, native SHAP, reference-background matching and 2 new reference-fault
families (expected benchmark case count 36 * 16 = 576, NOT YET VERIFIED).
The latest tip's scoped Actions run 43096160084 was queued when recorded;
do not claim it passed before observing the actual result.

Standalone independent SHAP sanity check (not the GitHub test): with default
masker sub-sampling, using the full-train reference caused ~0.238 maximum
feature discrepancy and ~0.269 sum discrepancy, whereas matching the real
all-train masker reference gave ~5e-15 discrepancy. This motivated an
explicit background SHA256 binding, not merely increasing tolerance.

Previous selective observer formative gate and chapter-4 smoke were green.
Broader practical/H10 workflows have longstanding unrelated Ruff failures.
The complete framework readiness workflow had earlier missing test/UI optional
dependencies; its install configuration now includes dev, ui, FastAPI, HTTPX.
No all-green full-release claim is made without a completed current run.

## Next executable action

Inspect Actions for latest SHA, especially observer regression 43096160084
(and any newer current-tip run). Fix actual pytest/Ruff failures without
weakening checks. Verify and download the new 576-case JSON report.
Check standard readiness/full integration separately; keep PR as draft
and no merge or scientific claim until the required checks have passed.
Do not opportunistically modify protected scores, H10/Q1 records,
article chapters, or unrelated framework modules.

## Additional class-conditional compatibility audit (Nov 12)

- Independently probed native `shap.LinearExplainer` with `Independent` maskers on Breast Cancer binary targets 0/1 and Wine/Digits multiclass targets 0/2 and 0/9. Both backgrounds (50 and 100 samples) have matching class-target logit attribution and true source-model single-feature effects (offline sklearn/shap sanity check, residuals ~1e-15).
- Added executable targeted regression for all six target/dataset cases and two background sizes: correctly selected target must CERTIFY; independently rehashed incorrect-class vector must BLOCK, while structural-only baseline remains CERTIFIED.
- The offline calculation is **not** a substitute for current-PR integration CI. New CI on this commit must pass before updating the review claim. This is a compatibility test for SHAP LinearExplainer only, not an arbitrary SHAP/LIME result.
- Current-tip Actions were queued (run 43096416081) when checked; no new green claim. Preserve read-only protected protocol and chapter files.
