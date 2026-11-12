# Verified scope: single-step numeric preprocessing evidence

Date: 2026-11-12
Branch: research/ml-transform-chain-20261112
Draft PR: https://github.com/lebedeffson/fuzzy-xai-operator/pull/9
Base: research/ml-output-call-witness-20261112 (draft PR #8)

## Parent purpose and limits
FuzzyXAI here is a read-only observer of data, training and model output
for numeric tabular ML. The observer itself NEVER trains, calls transform,
predicts, repairs, claims full run authenticity or autonomously acts.

## Implementation
- Optional TransformRecorder.transform forwards ONE explicitly requested
  transformer call and records ordered raw source row-ID/content SHA256,
  numeric transformed output SHA256, width and declared transformer type/tag.
- MLRunEvidence.transform_records feeds audit_transform_chain: raw train/eval
  source binding, preprocessor fit input/type agreement, transformed model
  fit and inference input links, with explicit UNKNOWN for missing, duplicate,
  unsupported, sparse/multi-stage, or unselected batch witnesses.
- Existing fit_trace_model_content/output_trace_predict_inputs remain UNKNOWN
  for transformed features; new specific transform_model_fit_link and
  transform_predict_link report the supported cross-stage checks.
- Numeric-only 1-step StandardScaler path. Actual transformer object identity
  and whether all execution paths were instrumented are NOT authenticated.
- Tests include real sklearn StandardScaler -> fit -> predict/proba, no
  automatic model calls, erroneous model-fit and inference transform values,
  raw evaluation report modifications, fit-on-test leakage, forged digest,
  repeated/missing events, unavailable string output, target mismatch and
  reopen from JSON.
- Frozen developmental protocol (committed before benchmark):
  experiments/ml_transform_chain_protocol_v1.json SHA256
  a2bc62868e2b1984e45cf9cbad22e317341109ddb495dc8e5383d3a9079dbdaa.
  Four sklearn pipelines x six known correlated scenarios = 24 cases.
  Not an independent blind study.

## Exact verified evidence
Semantic source SHA 386e8c334a871519271d5c9478b20c04e1ed3dac.
GitHub Actions 43298398664 SUCCESS:
  Python 3.11: 93 pytest PASS + Ruff PASS.
  Python 3.12: 93 pytest PASS + Ruff PASS.
Archived results 13980416621 (Python 3.11) and
13979489922 (Python 3.12) downloaded; all 24 per-case records,
protocol SHA, comparison counts and declared limits MATCH.
Clean 4/4 no false problems. Added transform evidence found 12/12
developer-controlled fit/inference/source content substitutions; baseline
ID checks found none of these. Both new observer and equal-access
comparator caught all 16/16 injected inconsistencies, including four
preprocessing fit-on-test events that standard fit-ID checks ALSO caught.
Missing transform events 4/4 -> UNKNOWN, not a false acceptance.

Docs: framework/fuzzyxai/docs/ML_TRANSFORM_PROVENANCE.md.

## Scientific/release boundary
The 24 cases share four datasets and were designed by the developers.
No general unknown-incident sensitivity, model quality guarantee,
transform/model identity attestation, data origin or universal ML claim.
The current code/CI are a scoped check: do not assert the full-repository
release suite is green. Existing protected scientific experiments,
dissertation chapters and four articles were not edited.
PR #9 remains DRAFT and UNMERGED, stacked on earlier draft observer PRs.

## Next meaningful independent research acceptance criterion
Use externally sourced, independently prepared fit/input/output traces
(including unscripted real preprocessing failures), compare to tools with
equivalent access, report clean false alarms and UNKNOWN rates. Do not add
autonomous repair or claim standards-complete model provenance.

---

# Scoped capability verified: observed model output provenance

Date: 2026-11-12
Branch: research/ml-output-call-witness-20261112
Draft PR: https://github.com/lebedeffson/fuzzy-xai-operator/pull/8
Base: research/ml-target-witness-20261112 (draft PR #7)

## Parent objective
FuzzyXAI is a read-only **observation/check** layer for supplied ML data,
explicitly witnessed fit/target evidence and caller-invoked predictions.
It must NEVER autonomously train, infer, repair, decide, or claim to
certify unobservable data or model quality.

## Implemented
- Explicit OutputRecorder.predict and OutputRecorder.predict_proba forward
  the caller-requested estimator call once, return its unmodified value,
  and record ordered numeric input hashes, row IDs, output shape/content
  hash and class-column order hash. A failed model call creates no
  successful witness. The observer itself never calls the model.
- MLRunEvidence.output_records and audit_output_calls compare source ID
  ordering, numeric input content, reported predict/proba values and
  declared probability class mapping to observed call evidence.
- Missing witnesses, unselected repeated batches, transformed inputs
  without a verified transform chain, string-class output and unsupported
  modalities remain UNKNOWN, not falsely OK. Probability checks for
  regression/clustering with no such channel are NOT_APPLICABLE.
- tests/test_ml_observer_end_to_end.py jointly verifies dataset, actual
  fit, target HMAC, caller-invoked output, read-only reporting,
  serialization and source-target vs output fault attribution.
- Four sklearn pipelines: Iris DecisionTree, Wine RandomForest, Diabetes
  LinearRegression, Iris KMeans. Frozen protocol:
  experiments/ml_output_witness_development_protocol_v1.json
  SHA256 1520580a0b640c258ee5204244cfe54a388fdea4ce1614caf45f982c82d7e79a.
  22 correlated developer-controlled scenarios, ordinary numeric and
  equally privileged direct-output baselines. No independent blind claim.
- An initial benchmark test failed (78 passed, 1 failure) because the
  ordinary baseline converted numeric class predictions to float then
  compared string '1.0' to class '1'. The numeric baseline implementation
  was corrected; the frozen protocol was not changed. The test now
  properly credits the ordinary baseline with detecting both classifier
  prediction/probability mismatches.

## Verified scientific/code evidence
Semantic source commit 0c56781b77387f8e01622a070979ac16d811e5a9.
GitHub Actions run 43293095587 SUCCESS:
  Python 3.11.18 / NumPy 2.4.6 / sklearn 1.9.1:
  79 pytest tests passed and Ruff passed.
  Python 3.12.16 / NumPy 2.5.4 / sklearn 1.9.1:
  79 pytest tests passed and Ruff passed.

Both real benchmark JSON artifacts were downloaded and byte-extracted
(13977288753 for Python 3.11; 13979275683 for Python 3.12).
The original protocol digest, 22 full per-case source records, counts,
strong baseline outcomes and scientific limitations match identically
on both environments. JSON bytes differ only in environment versions.

Comparison (ordinary baseline / FuzzyXAI witness / equally privileged):
  clean 4 cases: 0/0/0 alerts
  swapped predictions 4: 2/4/4 alerts
  swapped eval feature rows 4: 0/4/4 alerts
  swapped eval row IDs 4: 0/4/4 alerts
  swapped probabilities and reported column labels 2: 0/2/2 alerts
  absent output witness 4: ordinary 0; FuzzyXAI UNKNOWN 4/4.
Thus 14/14 known output faults detected by FuzzyXAI and the equally
privileged comparator; only 2/14 found by plain numeric validators.
These 14 cases are NOT independent unknown faults or general sensitivity.

Primary technical docs: framework/fuzzyxai/docs/ML_OUTPUT_PROVENANCE.md.
Reproducible outputs and checksums bundled as
/mnt/data/fuzzyxai_ml_output_verified_2026-11-12.zip
(only locally generated evidence bundle; not committed to GitHub).

## Integrity limits
Source input IDs, hashes and model identity are caller-produced, NOT
externally authenticated. A compromised caller can forge complete
records. A correct witness can be produced by an incorrect or biased
model. String labels, transformed features without transform witness,
arbitrary LLM/RAG/CV inputs, streamed/selected multi-batch inference,
and unknown real incidents are outside demonstrated support. The
general ML observer is still not a multioutput-regression quality oracle.
No autonomous repair, model fit or inference was added to the observer.

## State/next precise step
PR #8 remains intentionally DRAFT and unmerged, stacked over draft PRs
#7, #6, #5 and #4. Scoped CI and results verified, but the unrelated
full-project release suite and independent real-incident validation are
not established. Do not merge/claim universal detection merely from
the green scoped tests. Any future publication-generalization phase
needs independent test sources and comparator with equal input evidence.

---

# Scoped capability verified: target-value fit provenance

Date: 2026-11-12
Branch: research/ml-target-witness-20261112
Draft PR: https://github.com/lebedeffson/fuzzy-xai-operator/pull/7
Base: research/ml-observed-fit-events-20261112 (draft PR #6)

## Research/engineering problem
The existing feature-row fit witness could not distinguish correct
target labels y from a row-paired permutation of y if fit IDs and
feature-content hashes remained unchanged. The observer must still
remain read-only: no automatic training, corrections or operational
decision policy.

## Implemented
- An optional private-key HMAC-SHA256 of the complete ordered finite
  numeric 1-D/2-D target batch, bound to the corresponding exact
  feature-batch digest and collected only after successful explicit
  caller-initiated fit.
- No raw target values, per-class hashes or key serialized. The caller
  provides the same key separately to observe_ml_run for verification
  against independently supplied source row-to-target values.
- Fit records are backward-compatible. Missing key/reference,
  nonnumeric labels and unsafe large integer conversion yield UNKNOWN,
  no targets in an observed fit yield NOT_APPLICABLE. A disagreement
  yields PROBLEM but cannot tell which label source is correct.
- Negative tests: target shuffles, single modifications, missing sources
  and keys, malformed/replayed HMAC, numeric strings vs numeric class IDs,
  large integer precision, valid uint64 classes and unsupervised fit.
- Post-development dataset protocol committed before experiment:
  experiments/ml_target_witness_postdev_protocol_v1.json SHA256
  e05a5a7ef52596fa6a88637cc9e262d9722ac1d5ac4eeb5f26e8978726a26ea8.
  Wine, Digits, Linnerud: 3 datasets x 6 controlled cases = 18 correlated
  scenario rows; these were not in the prior fit-study set, but are
  sklearn datasets, not externally blinded sources.
- Comparative baseline: IDs and feature hashes; class histograms/target
  means; an equally privileged direct source-vs-fit target comparison
  that detects the same known mismatches. No unique model-quality or
  universal-novelty claim.

## Exact verification
Semantic source SHA f286750871e036b8387985c00bddad24b45e41f9.
GitHub Actions run 43285240892 completed SUCCESS:
Python 3.11: 60 pytest tests + Ruff PASS.
Python 3.12: 60 pytest tests + Ruff PASS.
Uploaded artifacts 13973294606 (3.11, NumPy 2.4.6) and
13974871693 (3.12, NumPy 2.5.4), sklearn 1.9.1 in both.
Both ZIP/JSONs downloaded and compared. Protocol digest and 18 source
scenario records are identical. Clean 3/3 OK; permuted labels 3/3
PROBLEM; single target modification 3/3 PROBLEM; no key and no
reference each 3/3 UNKNOWN; wrong reference 3/3 PROBLEM. The
privileged direct comparator catches all 9 observed contradictions
and demonstrates no advantage at equal access to evidence.

## Limitations
This checks source consistency, not independently established label
truth, complete unwrapped training, causal validity, predictive accuracy,
fairness, LLM/RAG/CV generality, or security against compromised local
recorders. HMAC key custody is an external requirement. The base
observer's 1-D regression metric check does not yet establish support for
multi-target evaluation metrics, although this new fit witness supports
2-D target inputs. Results are developer-created, not externally
blinded incident sensitivity. No protected scored/dissertation reports
or articles were changed. PR #7 remains draft/unmerged and stacked on
draft PRs #6/#5/#4; full-repository release approval not claimed.

## Next independent acceptance criterion
For a publication-level generality claim, use independently curated
source references and real mislabeled/contaminated fit events on new
datasets, separate from design and fixture authors. Compare against
established data-provenance tooling with equal evidence availability,
report alerts and false positives, and preserve unknown controls.

---

# Scoped fit-source provenance implementation verified

Date: 2026-11-12
Branch: research/ml-observed-fit-events-20261112
Draft PR: https://github.com/lebedeffson/fuzzy-xai-operator/pull/6
Base: research/ml-lifecycle-observer-readonly-20261112 (draft PR #5)

## Scope
Read-only FuzzyXAI diagnostics for supplied data, actual caller-initiated
fit calls and model outputs. No autonomous model training, automated repair
or unknown-result certification. Existing FuzzyXAI public API unchanged.

## Implemented
- Optional FitRecorder.fit forwards exactly one explicitly requested sklearn
  fit and records a successful call's row IDs, canonical numeric input
  fingerprints, role, feature space and batch digest.
- MLRunEvidence.fit_records / audit_fit_trace detect witnessed fit vs test
  overlap, raw source-content mismatch and contradictions with separately
  declared fit metadata. Missing/uninstrumented evidence is UNKNOWN.
- Frozen developer protocol: experiments/ml_fit_source_witness_protocol_v1.json
  SHA256 dc7cdfb78d3a2ff3540ba67152042bcc3966ff5f0021d042fa48f7c9a264e93b.
  Four model/dataset tasks x five deliberately engineered scenarios, 20
  correlated rows. Negative label-shuffle and no-witness controls retained.
- The initial experiment accidentally fitted tree/RF models rather than the
  protocol's LogisticRegression and LinearRegression, detected during code
  review and repaired WITHOUT rewriting the prior protocol. A first pass
  with raw LBFGS LogisticRegression also hit the Breast Cancer iteration
  ceiling. The final benchmark uses the declared model families, with
  convergent LogisticRegression(newton-cholesky, max_iter=800).
- Baseline fit-ID comparison catches 4/4 truthful fit/test overlaps, while
  source witness catches 8/8 deliberate cases with misleading fit IDs or
  fit metadata that baseline missed. 4 clean controls have no detection,
  4 uninstrumented fit controls are UNKNOWN (not OK).

## Exact verification
Semantic source SHA: f1fe1528cfc1e2bbf141eb60066bd84364cd1395.
GitHub Actions 43278096157: 37 pytest tests and Ruff passed on each
Python 3.11 and 3.12, including public API and existing ML observer
regression. Two saved benchmark JSON artifacts were downloaded and checked:
13971931839 (Python 3.11, NumPy 2.4.6) and
13971807226 (Python 3.12, NumPy 2.5.4);
scikit-learn 1.9.1 in both. Both reports had the same original protocol
SHA, exact estimator families, all observed LogisticRegression converged,
same 20 per-scenario records and detections. Note: doc/context updates
after semantic source SHA are documentation-only commits.

## Scientific limitations
Developer-built interventions are not blind real-world incident detection.
Four datasets are not a population of independent failures. Source row ID
and source data are caller-supplied and not cryptographically authenticated
by external trust. Unwrapped fit calls and target-label authenticity cannot
be certified. No evidence of superior general predictive quality, XAI
faithfulness, causal validity, LLM/RAG/CV generality or automatic repair.
No protected/scored study artifacts, article or dissertation chapters edited.

## State
This scoped implementation and the targeted acceptance checks are complete.
PR #6 intentionally remains draft and unmerged, stacked atop draft PR #5.
A later separate integration review / release may assess the whole project;
do not treat targeted green CI as whole-repository release approval.

---

# Scoped objective completed: read-only ML lifecycle observer

Branch: research/ml-lifecycle-observer-readonly-20261112
Draft PR: https://github.com/lebedeffson/fuzzy-xai-operator/pull/5
Base (draft PR #4): research/observer-evidence-binding-20261111

## Implemented

Evidence-only numeric tabular classification/regression/clustering checks
for **data, training, and model output**. Python API MLRunEvidence +
observe_ml_run and read-only JSON CLI. No fit/predict/repair/auto-action.

Measured channels: feature shape and order, finite values, train/evaluation
source IDs, model/preprocessor fit IDs, no-preprocessor not_applicable,
data duplicate-row warnings, descriptive mean shift, training loss trace
availability, output probability geometry and task-appropriate metrics.
Unknown evidence remains UNKNOWN, not success. Clean Iris has repeated
numeric rows that legitimately lead to a warning, not a hard block.

Independent developmental model pipelines (outside observer): 4 sklearn
datasets (Breast Cancer, Wine, Diabetes, Iris). Fixed protocol file committed
before running: experiments/ml_lifecycle_development_protocol_v1.json,
SHA256 0957fb3b51eee3d2c3ebba3c2f1a4ffd4f6fb125c78787b2e4e36947a2966395.
33 scenario rows, strong numeric and ID/schema baselines. Training/fit
provenance errors were also detected by the strong baseline. Additional
duplicate-content/mean-shift warnings are not universal scientific novelty.
Unobserved label-source permutation and natural exact duplicates are negative
controls. No independent blinded real-incident sensitivity is established.

## Verification

At semantic code commit 605404d9b5e2d67c0081f5c231e1a178166251bf:
GitHub Actions 43240790172: Python 3.11 and 3.12 each 26 pytest tests
passed and Ruff passed, including public FuzzyXAI API and no-cycle imports.
Archived JSON artifacts 13955300710 (3.11) / 13955325739 (3.12)
downloaded and independently compared: 4 models, 33 cases, same protocol
hash, identical per-scenario results and metrics. No full-repository
release readiness is asserted.

Integration revealed a stale test demanding 41 operators after PR #4 added
framework.exact_interventional_shapley_observer as the 42nd; assertion was
corrected and strengthened on PR #4 at 12177d85 and synchronized into PR #5
without deleting or skipping the test.

The actual code, protocol and measured comparisons are in the draft PR;
no main-branch merge, model weights changes, or protected study results /
paper and dissertation edits were performed.

## Next independent research criterion

If publication-level effectiveness is claimed: run external datasets,
actual instrumented provenance, prespecified fault cases by an independent
source, measure false alarms/coverage and compare to mature data and
model-checking tools. Unsupported ML modalities (NLP, CV, LLM/RAG)
require explicit dedicated evidence adapters. No claim of universal
accuracy, causality or safety should be made until established.


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
