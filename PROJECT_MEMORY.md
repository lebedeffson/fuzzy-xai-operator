# Project Memory

- Date: 2026-07-19
- Branch: `feat/universal-model-integration`
- Base: `79d39f61f02df1cf1d63fc92dd2a1ebebfef7de7`
- Universal API commit: `3dda3e3`
- Model evidence commit: `a76677a`
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
- frozen external A/B pilot package and external domain-language review package;
- aggregate Chapter 4 candidate evidence package.

## Measured Validation

- 34 core model configurations verified;
- prediction parity rate: `1.0`;
- adapter conformance rate: `1.0`;
- explanation graph validation rate: `1.0`;
- strict MyPy: PASS;
- Ruff and compile checks: PASS;
- focused compatibility/model/external-gate suite: `48 passed`, `6 skipped` optional runtimes;
- existing measured checkpoint experiment and native rule ablation retained;
- model, pilot, domain-review, and Chapter 4 manifests: PASS.

The full regression suite was deliberately not rerun after the user reported Cursor instability. It must run in CI
and before any merge or tag. Optional runtime jobs are pending public CI.

## External Gates

- independent A/B comprehension pilot: `planned_not_run`, zero participants;
- regulated-domain dictionary review: `pending_external_review`;
- Chapter 4 computed evidence: PASS;
- demonstrated human comprehension: not claimed;
- release gate: `BLOCKED`;
- `v1.2.0` tag and merge to `main`: forbidden until external gates and main CI pass.

## Version Boundary

- `v1.2.0`: Human Explanation and empirical computational gate implemented; final tag blocked externally.
- `v1.3.0rc1`: universal tabular model integration candidate implemented on this branch.
- `v1.4.0`: anomaly, forecasting, text, image, and richer checkpoint integrations remain future scope.

## Archive Policy

- source archive is built only from a committed Git index with `python scripts/build_framework_release.py`;
- Chapter 4 candidate ZIP is reproducible but not a final-release claim;
- doctoral historical archive remains separate;
- generated site content and unrelated dirty-worktree artifacts must not enter the source archive.

## Next Step

Push the branch, require green core and optional-runtime CI, fix only measured failures, and rebuild committed-index
archives. Separately run the real external pilot and domain review. Only then review merge to `main` and create a
release tag. Do not claim `v1.4.0` support before modality-specific adapters and benchmarks exist.
