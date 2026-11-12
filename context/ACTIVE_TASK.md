# Current scoped task: read-only ML lifecycle observer

User objective (2026-11-12): FuzzyXAI should simply observe and check
**data, model training and output**, without fixing models, retraining,
inventing explanations or choosing operational actions.

- Branch: research/ml-lifecycle-observer-readonly-20261112
- Draft PR: https://github.com/lebedeffson/fuzzy-xai-operator/pull/5
- New module: framework/fuzzyxai/fuzzyxai/ml_observer.py
- Usage: python -m fuzzyxai.ml_observer collected_run.json --output observation.json
- Numeric tabular classification, regression, clustering are supported;
  other AI modalities require adapters rather than fabricated checks.
- No model fit/predict, repair, state mutation, or automatic decision.
- Explicit "unknown" for missing source evidence; no general quality score.
- Tests: tests/test_ml_lifecycle_observer.py; Python 3.11/3.12 scoped CI
  under .github/workflows/ml-lifecycle-observer.yml.
- At this checkpoint 11 scoped tests passed on both versions at prior
  code state (Actions 43233813218) but Ruff had a type-union lint error.
  This was corrected; final PR head CI remains pending verification.
- Next action: validate latest CI at PR head, fix exact failures without
  weakening checks; keep draft, no merge. Previous observer research remains
  below and is not reinterpreted as universal accuracy evidence.

---

# Active task: FuzzyXAI Observer — exact source-bound explanation checks

Branch: research/observer-evidence-binding-20261111
Draft PR: https://github.com/lebedeffson/fuzzy-xai-operator/pull/4
Protected base: research/tisu-core-safety-20261108
Never modify locked/scored dissertation evidence or publication chapters.

## Parent objective

Increase practical and mathematically defensible observer coverage before
revising four articles. Fail closed: contradictions BLOCK; missing or
unsupported evidence REVIEW; neither state is model-quality/causal truth.

## Validated earlier scope

Local evidence identity, input SHA256, background/reference SHA256, linear
additive logit responses, SHAP LinearExplainer class semantics, carefully
guarded candidate-only linear reference rebasing, and train-fold-only OOF
scaling were already added. GitHub Actions 43108180566 passed 98 tests
and Ruff on Python 3.11 and 3.12. The older 576-case, 36-heldout-object,
developmental linear mutation benchmark is NOT independent external evidence.

## New capability: bounded exact empirical-interventional Shapley

- New module fuzzyxai.selective_observer.exact_shapley (public SDK exports).
- Defined v(S) = mean_{r in B} f_target(x_S, r_notS), using the exact
  background rows and the SAME selected scalar model output for every mask.
- Enumerates all coalitions and computes the standard Shapley weights.
- A trusted adapter must supply actual deterministic model calls; SHA256
  declarations do not authenticate underlying model bytes or explainers.
- Strict 1..8 input-feature and 8192 model-score-row caps; invalid model
  outputs, inconsistent dimensions or non-deterministic repeated inputs
  cause explicit exceptions, not fabricated measurements.
- ExactShapleyEvidence is bound to LocalExplanationEvidence by object,
  model, target class, ordered feature axis, input vector SHA256 and
  SHA256 of the COMPLETE finite numeric empirical background matrix.
- A different/missing background is distinguished from an ordinary
  reference mean. For nonlinear interactions equal marginal means can
  coexist with different explanation values.
- New opt-in practical-controller guard: require_exact_shapley_verification.
  Contradictory declared/local effects BLOCK; missing evidence/unsupported
  protocols REVIEW before budget allocation; old linear gate remains separate.
- Analytic counterexample: f(x0,x1)=x0*x1, x=(1,1), B={(0,0)} has exact
  Shapley (0.5, 0.5) while independent one-feature drops give (1, 1).
  The earlier linear-additive oracle would wrongly reject valid SHAP here.
- Tests cover native SHAP ExactExplainer, a fitted nonlinear random forest,
  equal-mean/different-joint-background negative control, NP float32
  inputs, cost boundaries, rehashed zero-sum errors, stale inputs,
  missing/incorrect backgrounds, invalid scores, and public action gating.

## Nonlinear model benchmark — recorded evidence

GitHub run 43111380695 generated artifacts:
- 13896821884: Python 3.11, NumPy 2.4.6, scikit-learn 1.9.1, SHAP 0.51.0;
- 13895274744: Python 3.12, NumPy 2.5.4, scikit-learn 1.9.1, SHAP 0.51.0.
Both real JSON reports were downloaded and inspected.
Three trained nonlinear classifiers (DecisionTree/RandomForest),
Breast Cancer, Wine, Digits, with 4 selected raw numeric features each.
36 held-out objects, 216 correlated scenarios, 6 scenarios/object.
Counts agree on both environments:
- 36/36 clean native SHAP explanations CERTIFIED;
- 72/72 rehashed zero-sum attribution corruptions BLOCKED, while
  structure-plus-sum baseline accepts all 72;
- 36/36 wrong background digests BLOCKED, while structure+sum accepts 36;
- 36/36 missing background digests REVIEW, 36/36 stale inputs BLOCKED;
- native SHAP ExactExplainer vs independently evaluated source model
  coalition formula max absolute discrepancy 1.1102230246251565e-16.
These are deliberately created faults; model families, dataset selectors,
masker size and scoring channel all belong to this developmental design.
NOT 216 independent observations, unknown incident detection, or external
scientific efficacy. Independent data/protocol evaluation is still needed.

## CI and current work

The same 43111380695 run executed 117 passing pytest tests on both Python
versions. Its Ruff gate flagged B023 in the nonlinear benchmark:
the nested source scorer referenced an outer model loop variable.
Fixed by binding the actual fitted estimator as a default callback argument
in later commit 807e11e44054e07f5ce59e64aa57af5eb98dcad3.
Added explicit public assess_action fail-closed regression under exhausted
review budget at commit 3469fded4bd9f68c6a9adb6fa81a8562a24b0bbf.
CI for the latest code has NOT yet been checked as green.

## Exact next action

Inspect newest Observer evidence-binding regression for the branch head,
fix actual pytest/Ruff errors without weakening scientific gates. Confirm
JSON artifact count and provenance; then update PR with the verified code
SHA and tests. Keep PR draft and no merge. Next external research task
is an independently defined unknown-fault protocol. Avoid code claiming
general SHAP/LIME/causal faithfulness from this exact empirical game.

## Latest end-to-end action regression checkpoint

- Actions run 43112095660 at code 807e11e44054e07f5ce59e64aa57af5eb98dcad3 completed successfully on Python 3.11 and 3.12: 117 pytest passed, Ruff passed, benchmark JSON uploaded.
- Public assess_action regression added next: 118 pytest passed in run 43112270083 on both versions; its Ruff found only import order in test_observer_exact_shapley.py. The requested import order was fixed in commit 3c6dacd7543a4c31fbc38a2bf6702ec3601bf853.
- Final-code SHA 3c6dacd7543a4c31fbc38a2bf6702ec3601bf853 scope CI runs 43112582904 and 43112592007 were queued at this checkpoint; DO NOT mark that commit fully validated without observing completion.
- Changes after the code fix were to this context and framework/fuzzyxai/docs/OBSERVER_EVIDENCE_BINDING.md, not to runtime semantics. Full repo readiness remains separate and is NOT certified.
- Next executable action: check both Python jobs for final-code SHA, fix any real failures, then freeze a developmental benchmark receipt; leave PR #4 draft and preserve all locked protocol/publication material.


## Transfer and semantic-compatibility experiment (new work, pending CI)

**Scientific hypothesis:** An exact empirical interventional-Shapley oracle
must NOT block honest conditional/approximate explanations simply because
they disagree with the exact interventional game. Nor should a complete
source identity for an explanation omit the model's output score channel.
Earlier checks over linear models and a fixed interventional game did not
measure genuine alternate-background admissions or sub-tolerance misses.

Changes:
- LocalExplanationEvidence now optionally declares attribution_protocol
  and score_channel. Canonical source digest includes both.
- Strict exact-Shapley gate requires an explicit exact interventional
  protocol declaration and matching source score channel. A different or
  missing explanation-game declaration results in REVIEW, not false BLOCK
  based only on different attribution values. Contradictory object/class/
  input/background/score-space identity still BLOCKs.
- Separate, developer-fixed transfer protocol committed FIRST at
  experiments/observer_transfer_protocol_v1.json (commit 24b08fe7543a).
  This is not independently preregistered or blind.
- New datasets: Iris classification, Diabetes scalar regression,
  Linnerud first regression target; novel fitted model families HGB,
  ExtraTrees and k-nearest-neighbor regression. Seed 173; 5 held-out
  objects/task; two disjoint six-row training backgrounds per task.
- Separate native SHAP ExactExplainer vs exhaustive source-model score
  probes, with 9 scenarios/source object: two truly valid explanations
  including changed background, explicit 0.05 zero-sum corruption, a
  deliberately *sub-tolerance* 1e-8 perturbation, a valid donor's
  explanation rebalanced to match total score, background and score
  channel identity faults, an honestly unsupported approximate method,
  and missing background. Strong baseline checks complete provenance and
  efficiency, but not Shapley per-feature values.
- Report records case-level outcomes, numerical fault opportunities,
  per-family and per-task counts, true background shift, environment,
  fitted-model fingerprints, and the SHA256 of the separately committed
  protocol. Any misses, false blocks or inability to distinguish donor
  attributions must be reported, not silently tuned away.
- Latest code source: experiments/observer_transfer_evaluation.py,
  tests/practical_controller/test_observer_transfer_evaluation.py;
  CI adds 3.11/3.12 targeted regression and an archived JSON receipt.
  Source commits 0073f241, 8b23f8fc, 551b1d2f, 7001542c.
- **No latest-tip result is yet confirmed.** Earlier 118-test 3.11/3.12
  success only proves the previous nonlinear exact protocol. No result
  is to be attributed to new transfer models until actual Actions JSON
  has been inspected.

## Exact next executable action after this checkpoint

Inspect newest Observer evidence-binding regression Actions at current
branch SHA; fix exact pytest/Ruff failures without loosening checks or
optimizing on held-out outcomes. Download and inspect genuine transfer
JSON artifacts on both Python versions; report all controls and misses,
including below-tolerance false admissions. Only then consider updating
draft PR scientific claims. Never modify protected H10/Q1 scores or articles.
