# FuzzyXAI ML Vertical v1: gap analysis

Audit status: complete before implementation.

## Implemented and reusable

- Canonical model adapters, public FuzzyXAI runtime and evidence-first behavior.
- Mathematical fuzzy representations, reductions and Pareto selection primitives.
- Route graph, validator, diagnostic issue/cut, repair plan/executor and recertification records.
- Breast Cancer demonstration foundation, deterministic sklearn tooling and real SHAP usage in experiment scripts.
- NiceGUI technical/defense interfaces.
- Pinned local MLflow integration and artifact logging pattern.
- Source release builder, claim lint, operator manifest and parent-result immutability checks.

## Partial gaps

- Existing sklearn evidence may use native linear terms but does not expose a version-bound SHAP artifact with base-value consistency.
- Existing observer actions do not provide the exact five-state product contract.
- Existing uncertainty selection is spread across research modules and is not connected to one user request.
- Existing route diagnostics are generic; the requested ML/XAI contracts need deterministic registered observations.
- Existing demos show practical pieces but do not return three views of one canonical run.
- Existing MLflow demo does not log the full nine-artifact vertical run.

## Missing product pieces

1. A single `MLVerticalService` executing model, SHAP, fuzzy object, uncertainty selection, reduction, route validation, observer and presenter.
2. Typed request/prediction/evidence/claim/local-explanation/uncertainty records with deterministic canonical hashes.
3. A frozen Breast Cancer model bundle and versioned `EXPLAIN_PLAN.yaml`.
4. Registered scenario controls S1-S10 that change only observable pre-fix conditions, never Gold labels.
5. A complete REST surface and run store.
6. A defense UI page consuming the same service.
7. MLflow logging of request, prediction, explanation, route, diagnostics and all views.
8. Docker Compose services for API, UI and MLflow.
9. Acceptance runner, aggregates, immutable result status and release package.

## Implementation strategy

- Add `fuzzyxai.ml_vertical` as an orchestration package over existing canonical modules. It is not a parallel framework.
- Keep thresholds, rules and membership functions in the versioned ExplainPlan.
- Use real `LogisticRegression` and real `shap.LinearExplainer`; no label enters the request or explanation channels.
- Store one canonical run dictionary. User, engineer and auditor payloads reference its `explainable_object_hash`.
- Fail closed: missing feature becomes `REQUEST_DATA`; missing provenance/version mismatch becomes `BLOCK`; source conflict becomes `REVIEW`; excessive reduction loss becomes `WARN` or `REVIEW` according to the locked plan.
- Keep repair operations limited to recomputing explanation and restoring registered route metadata/artifacts.

## Acceptance risks and mitigations

| Risk | Mitigation |
|---|---|
| SHAP optional dependency unavailable | declare a vertical extra and fail readiness; never substitute coefficients while claiming SHAP |
| Missing MLflow | API remains fail-closed for logging status and readiness reports the limitation; Docker installs the pinned extra |
| Prediction timestamps break determinism | exclude wall-clock fields from canonical hash and use deterministic logical timestamps in scenario evidence |
| Scenario controls leak target labels | requests contain only features and registered fault controls; tests audit forbidden keys recursively |
| High probability masks a critical defect | observer checks critical issues before confidence |
| UI/API diverge | both instantiate the same service and consume the same run store |
| Old results change | compare the pre-work SHA256 list for all non-vertical `results/reports/protocol` files after implementation |

## Stop rule

If any mandatory S1-S10 scenario, false-certification condition, deterministic-hash check, MLflow artifact check, parent immutability check or full regression fails, the release status is `FUZZYXAI_ML_VERTICAL_V1_ACCEPTANCE_FAILED`. The implementation must not weaken the plan after viewing acceptance results.
