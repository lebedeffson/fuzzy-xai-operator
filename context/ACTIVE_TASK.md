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

## Baseline-scope repair at commit following b665d77d

- GitHub observer workflow 43105867008 reported 3 failed/88 passed: two benchmark regressions and the older native SHAP baseline expected CERTIFIED after removing source-model effects, but left `require_reference_binding=True`. This yielded REVIEW for missing model-effect reference evidence. Strict production verifier behaved correctly.
- Repaired the explicitly weaker baseline *in the experiment and its comparison test* by disabling `require_reference_binding` together with `require_local_effect_verification`. Strict/full guard and strict reference checks remain unchanged. Strong comparison still independently checks sum reconstruction; do not describe reduced baseline as verifying an independently bound reference.
- Re-run and inspect current-tip 3.11/3.12 GitHub CI; no new PASS claim before completion.

## Verified observer milestone (Nov 12, after SHAP-class and baseline repair)

- Commit 2d4f845b129ccc446f72f7de36e6da2d9b1c8964, Actions run 43106307484: Python 3.11 and 3.12 both passed 91 scoped pytest tests, benchmark generation, and Ruff.
- Downloaded benchmark artifacts 13894551490 (3.11), 13893738070 (3.12): 36 held-out objects, 576 controlled cases, 36/36 clean accepted.
- Four same-sum attribution corruption families: 144/144 rejected by source-model feature-response checks but accepted by the deliberately weaker structural-plus-logit-sum baseline; all were rehashed consistently.
- Wrong background: 36/36 blocked; missing background: 36/36 REVIEW. Maximum clean logit discrepancy ~4.66e-15 on Python 3.11 and ~4.88e-15 on Python 3.12.
- Native SHAP LinearExplainer regression (12 dataset/target/background combinations): binary target 0/1, Wine target 0/2, Digits target 0/9, masker sizes 50 and 100. Correct target certified; wrong-class rehashed attributions blocked. These are only linear-logit compatibility checks.
- Independent offline background sanity check: masker 50 versus 100 changes some per-feature contributions by ~0.1266 (Breast Cancer), up to ~0.1594 (Wine), up to ~0.2491 (Digits). Matching each real masker reference resolves the difference.
- NumPy versions differ (3.11: 2.4.6; 3.12: 2.5.4); resulting fitted model fingerprints differ. Behavior counts coincide. This is a reproducibility boundary, not identical byte-level training.
- Strict gate was never weakened to make CI pass. Failures came from an intentionally weaker baseline that disabled model effects but still required the independent model-effect reference. The baseline explicitly disables both requirements; production verifier remains strict.
- Scoped CI PASS does not equal release-wide PASS; H10 workflows include unrelated failures. No changes to locked scores, H10/Q1 protocols, articles or dissertation chapters.

## Exact continuation

- Verify scope CI on the documentation/state-only child commit, keep PR #4 draft.
- Next research milestone is a genuinely independent unknown-fault protocol and other XAI semantics; do not interpret 576 developer-generated, correlated cases as an external estimate of detector recall.
