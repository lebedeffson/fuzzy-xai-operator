# FuzzyXAI ML Vertical v1: current component map

Audit base commit: `192fab11eca2bb754f479e6fe0a82c99f200acd4`.

This audit precedes ML Vertical implementation. It maps the requested vertical to the existing canonical framework and does not alter any Chapter 4 result.

| Required role | Existing implementation | Status | Reuse decision |
|---|---|---|---|
| Canonical public API | `fuzzyxai.runtime.FuzzyXAI`, `FuzzyXAI.wrap(...).explain(...)` | implemented | preserve; vertical is an orchestration service, not a competing public library |
| Adapted input | `fuzzyxai.core.types.AdaptedInput` | implemented | reuse in route input node |
| Explainable object | `fuzzyxai.core.types.ExplainableObject` and mathematical `fuzzyxai.core.explanation_object.ExplanationObject` | duplicated roles | use public `ExplainableObject` as transport; retain mathematical object in components/metadata |
| ExplainPlan | `fuzzyxai.core.explain_plan.ExplainPlan`, YAML loader/validator/hash | implemented | reuse loader/hash; add a versioned vertical plan without changing the class |
| Model adapter | `fuzzyxai.adapters.model.SklearnAdapter` and v2 registry | implemented | reuse fitted sklearn model and expose vertical prediction artifact |
| SHAP | measured scripts use `shap.LinearExplainer`; no canonical version-bound vertical artifact | partial | add a vertical `ShapExplainerAdapter` returning measured contributions and provenance |
| Fuzzy operators | memberships, rule activation, T/S norms in hierarchy/core modules | implemented but fragmented | compose through one `FuzzyExplanationBuilder` facade |
| `F0` | `hierarchy` base representations | implemented | reuse semantics, serialize in the vertical schema |
| `F_int` | `IntervalFS` and reductions | implemented | reuse semantics, preserve interval source references |
| `NAS` | `NeutrosophicFS` | implemented | reuse semantics; require distinct support/rejection sources for conflict |
| `F_ML` | `MultiLevelFS` | implemented | reuse semantics; expose prediction/explanation/rules/trace/reduction/action levels |
| Representation selection | `selection.pareto_selector` and uncertainty-selection experiments | implemented | apply fixed plan policy through a vertical selector |
| Reduction loss | `core.reduction`, `hierarchy.reductions`, risk pipeline | implemented | calculate and make it an observer input |
| Route graph | `diagnostics.contracts.RouteGraph`, `RouteNode`, `RouteEdge`, `Contract` | implemented | use as the sole audit graph |
| Route validation | `diagnostics.validator.RouteValidator` | implemented | reuse generic contracts and add registered ML/XAI contract observations |
| Diagnostic issue/cut | canonical diagnostics contracts and minimal-cut solver | implemented | reuse output types; vertical issues remain ML/XAI route issues |
| Repair plan/execution | `ActionableRepairPlanner`, `RepairExecutor` | implemented | use only registered safe route operations |
| Recertification | `RouteRecertifier` | implemented | rebuild and validate the complete route after repair |
| Observer | risk observer supports accept/audit/block semantics | partial | add deterministic `ACCEPT/WARN/REQUEST_DATA/REVIEW/BLOCK` mapping without changing prediction |
| Human presentation | evidence/human and plain-language modules | partial | add deterministic user/engineer/auditor views from one canonical run |
| Dashboard | `apps/unified_demo.py` technical hub; `apps/layered_demo.py` defense interface | implemented foundation | add ML Vertical page to `layered_demo.py`; do not create a second dashboard product |
| REST API | no dedicated complete vertical endpoint set | missing | add a small FastAPI application around the orchestration service |
| MLflow | pinned `2.22.5` integration example and tests | implemented foundation | log the nine registered JSON artifacts and version/hash tags |
| Docker | framework/practical Dockerfiles and compose exist | partial | add an isolated ML Vertical compose profile with API, UI, MLflow and artifact volume |
| Breast Cancer data | sklearn loader and real reduction example | implemented | use deterministic train/validation/test IDs; make no clinical claim |

## Canonicality boundary

The implementation must not rename or replace `AdaptedInput`, `ExplainableObject`, `RouteGraph`, `DiagnosticIssue`, `DiagnosticCut`, `RepairPlan`, `RecertificationReport`, or `DiagnosticReport`. New typed records are vertical input/output artifacts only. Repository-diagnostic R10M code is out of scope.

## Existing duplication

- `core.types.ExplainableObject` is the public route transport, while `core.explanation_object.ExplanationObject` implements the mathematical fuzzy object. The vertical will explicitly nest the mathematical payload in the transport object rather than introduce a third core type.
- `roles.interfaces` contains protocol-shaped route/repair records, while `diagnostics.contracts` contains the executable canonical records. The vertical uses `diagnostics.contracts`.
- Multiple experimental uncertainty selectors exist. The vertical adds no new mathematical class; it fixes one product policy in `EXPLAIN_PLAN.yaml` and delegates representation semantics to the existing hierarchy.

## Public product boundary

`apps/unified_demo.py` remains the technical hub. `apps/layered_demo.py` remains the defense/user interface. The vertical backend is shared by API, UI, tests and evidence generation.
