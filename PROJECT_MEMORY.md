# Project Memory

## Full Empirical Validation

- Date: 2026-07-20
- Branch: `feat/full-empirical-validation`
- Final technical closure commit: `fc8705ececdcc58f6cc171017984d9c6af05dd45`
- Final full empirical CI run: `29707225733`
- Candidate status: full E1-E8 technical validation; stable release remains blocked

The current milestone adds a controlled, evidence-first validation layer without changing the defended chapter 2-3
operator semantics. Four deterministic modality contours contain 10,000 objects each. The full protocol measures
repeated rule ablation, explanation baselines, P1-P7 action policies, calibration, sensitivity, adaptive uncertainty
representations, critical-rupture association, and scalability. The Docker image runs the same one-command pipeline.

Measured boundaries must remain explicit:

- SHAP, LIME, Anchors, RuleFit, confidence, disagreement, reduced-history FuzzyXAI, and full FuzzyXAI are measured;
- the 50-pair rule-ablation experiment does not confirm a general subgroup-recall effect;
- adaptive selection uses `FML` for 55.22% of controlled objects and is non-inferior to always-FML in the declared risk metric;
- the critical-rupture indicator has incremental AUPRC `-0.12556024709557384`, so no safety or predictive-gain claim is allowed;
- the critical rupture remains a structural diagnostic indicator only;
- controlled multimodal results do not establish external-domain generalization;
- expert review and comprehension pilot remain `planned_not_run`; domain semantic review remains pending;
- merge to `main` and a stable tag remain forbidden while those external gates are open.

Public run `29707225733` passed protocol smoke, full E1-E8, checkout-independent Docker reproduction, final evidence
aggregation, archive verification, and the 90-item technical Definition of Done. Stable release remains blocked by
the three explicitly recorded external gates, not by a computational failure.

- Date: 2026-07-19
- Branch: `feat/universal-model-integration`
- Base: `79d39f61f02df1cf1d63fc92dd2a1ebebfef7de7`
- Universal API commit: `3dda3e3`
- Model evidence commit: `a76677a`
- CI correction commit: `fbfcdd3`
- Cross-runtime evidence commit: `39067b3`
- Candidate version: `1.3.0rc1`
- Release tag: not created

## Current Focus

The installable research framework remains the primary product. The generated DubnaXAI site stays quarantined.
This milestone adds capability-based model integration and measured explanation-quality disclosure without changing
the defended chapter 2-3 operator semantics.

## Implemented

- `ModelAdapterV2`, task/input/output contracts, capability descriptors, and native/derived/surrogate/external origin;
- priority adapter registry, explicit resolution report, lazy optional imports, and plugin entry-point discovery;
- sklearn linear, tree, ensemble, SVM, KNN, Naive Bayes, regression, and Pipeline adapters;
- optional XGBoost, LightGBM, CatBoost, PyTorch, TensorFlow/Keras, and ONNX adapters with separate tests;
- `ExplanationPlanner`, surrogate-fidelity blocking, adapter conformance, and quality reports;
- `explain_batch()`, `explain_global()`, `why_not()`, `compare_models()`, and capability reporting;
- deterministic benchmark with 24 classification and 10 regression configurations;
- checksummed runtime reports for sklearn, XGBoost, LightGBM, CatBoost, PyTorch, TensorFlow/Keras, and ONNX;
- automatic report merge, cross-version conflict detection, unified support matrix, API report, and quality report;
- frozen external A/B pilot package and external domain-language review package;
- aggregate Chapter 4 candidate evidence package.

## Measured Validation

- 34 core model configurations and six optional runtime integrations verified;
- unified model support matrix: `40/40 pass`;
- runtime reports: `14/14 pass` across Python 3.11 and Python 3.12;
- prediction parity rate: `1.0`;
- adapter conformance rate: `1.0`;
- explanation graph validation rate: `1.0`;
- strict MyPy: PASS;
- Ruff and compile checks: PASS;
- focused local compatibility/model/external-gate suite: `48 passed`, `6 skipped` optional runtimes;
- public readiness CI: `338 passed`, `6 skipped` on Python 3.11 and Python 3.12;
- public Octave JSON/dashboard smoke: PASS;
- public optional runtime CI: XGBoost, LightGBM, CatBoost, PyTorch, TensorFlow/Keras, and ONNX PASS on both Python versions;
- five public explanation APIs: PASS in all runtime reports;
- explanation-quality matrix: `40/40 pass`; measured top-reason stability in 36 configurations;
- readiness run: `29700288436`;
- universal model run: `29700288413`;
- existing measured checkpoint experiment and native rule ablation retained;
- model, pilot, domain-review, and Chapter 4 manifests: PASS.

The full regression suite was moved from the workstation to GitHub runners after the user reported Cursor
instability. Both supported Python jobs passed. This validates the feature branch, not `main` after merge.

## External Gates

- independent A/B comprehension pilot: `planned_not_run`, zero participants;
- regulated-domain dictionary review: `pending_external_review`;
- Chapter 4 computed evidence: PASS;
- demonstrated human comprehension: not claimed;
- release gate: `BLOCKED`;
- `v1.2.0` tag and merge to `main`: forbidden until external gates and main CI pass.

## Version Boundary

- `v1.2.0`: Human Explanation and empirical computational gate implemented; final tag blocked externally.
- `v1.3.0rc1`: universal tabular model integration and merged cross-runtime evidence candidate implemented on this branch.
- version path B selected: no retroactive stable `v1.2.0`; next allowed tag after external gates, main merge, and main CI is `v1.3.0rc2`.
- `v1.4.0`: anomaly, forecasting, text, image, and richer checkpoint integrations remain future scope.

## Archive Policy

- source archive is built only from a committed Git index with `python scripts/build_framework_release.py`;
- Chapter 4 candidate ZIP is reproducible but not a final-release claim;
- doctoral historical archive remains separate;
- generated site content and unrelated dirty-worktree artifacts must not enter the source archive.

## Next Step

Run the real external pilot and domain review. After both pass, review merge to `main`, require green main CI, rebuild
committed-index archives, and only then create a release tag. Do not claim `v1.4.0` support before modality-specific
adapters and benchmarks exist.
