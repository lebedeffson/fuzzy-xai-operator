<div align="center">

# FuzzyXAI

### Evidence-first control for ML and XAI pipelines

**Observe the route. Verify the contracts. Explain the failure. Repair safely. Recertify everything.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-1f6f8b)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-c98b2e)](LICENSE)
[![Package](https://img.shields.io/badge/version-1.4.0a2-315c4c)](pyproject.toml)
[![Validation](https://img.shields.io/badge/external%20pipelines-4%20validated-315c4c)](results/external_ml_pipeline_v1/FINAL_STATUS.json)

</div>

FuzzyXAI is a research framework for building auditable explanations and
controlling the integrity of an executable ML/XAI route. It connects data,
splits, preprocessing, models, predictions, post-hoc explanations, fuzzy
representations, decisions, repair, and recertification in one evidence graph.

The framework is deliberately fail-closed: missing or contradictory evidence
produces an explicit limitation, review request, or blocked route. It never
invents a metric, provenance link, causal claim, or successful certification.

```text
Dataset -> Split -> Preprocessor -> Model -> Prediction -> Explanation
   |          |           |           |          |             |
   +----------+-----------+-----------+----------+-------------+
                              RouteGraph
                                  |
                         Contract validation
                                  |
                          Minimal diagnostic cut
                                  |
                    Registered repair + rollback
                                  |
                         Full route recertification
```

## Why FuzzyXAI

Ordinary component checks can tell us that a shape is valid, a metric is
finite, or an artifact exists. They often cannot prove that:

- a preprocessor was fitted only on the registered training partition;
- a model consumed the same feature schema the preprocessor produced;
- a prediction and explanation belong to the same object and model version;
- a stale artifact is the root cause of several downstream symptoms;
- a repair did not create a new critical violation elsewhere in the route.

FuzzyXAI makes those relationships explicit and machine-verifiable.

## Core Capabilities

| Capability | What the framework provides |
| --- | --- |
| Model wrapping | One public entry point: `FuzzyXAI.wrap(...).explain(...)` |
| Evidence-first explanation | Typed claims remain linked to measured evidence and provenance |
| Route modeling | Immutable nodes, edges, component versions, schemas, hashes, and evidence refs |
| Contract auditing | Local and inter-stage consistency checks over the complete route |
| Diagnosis | Violated stage, component, contract, observed value, expected value, and evidence |
| Causal reduction | Minimal diagnostic cut separating a primary cause from dependent symptoms |
| Safe repair | Registered operations with preconditions, rollback, and explicit state mutation |
| Recertification | Rebuild changed artifacts and verify every applicable contract again |
| Fuzzy representation | F0, interval, hesitant, neutrosophic, and multilevel representations |
| Integration | Python API, REST surfaces, visualizations, MLflow evidence, and adapter SDK |
| Reproducibility | Protocol locks, deterministic canonical JSON, SHA256 manifests, and source releases |

## Install

```bash
python -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[dev]"
```

Optional integrations are installed explicitly:

```bash
.venv/bin/python -m pip install -e ".[ml-vertical]"  # SHAP, REST, MLflow
.venv/bin/python -m pip install -e ".[xgboost]"
.venv/bin/python -m pip install -e ".[lightgbm]"
.venv/bin/python -m pip install -e ".[catboost]"
```

Python 3.10 or newer is required. Locked research environments are recorded in
`requirements.lock`, `uv.lock`, and the corresponding protocol directories.

## Quick Start

### Explain a trained model

```python
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from fuzzyxai import FuzzyXAI

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, _ = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=500, random_state=42),
).fit(X_train, y_train)

fx = FuzzyXAI.wrap(model, adapter="auto", task="classification")
result = fx.explain_one(
    X_test[0],
    object_id="patient-0001",
    reference_data=X_train,
    reference_labels=y_train,
)

print(result.summary(audience="domain_user", detail="short"))
print(result.quality_report())
print(fx.capability_report())
result.export_json("explanation.json")
```

The prediction is available even when optional explanation channels are
missing. Unsupported claims stay absent and are listed in the quality report.

### Diagnose a pipeline route

```python
from fuzzyxai.diagnostics import DiagnosticService

route = {
    "route_id": "training-run-42",
    "nodes": [
        {
            "node_id": "preprocessor",
            "node_type": "preprocessing",
            "component_id": "standard_scaler",
            "component_version": "1.0",
            "registered_attributes": {"fit_scope": "train"},
            "observed_attributes": {"fit_scope": "train_plus_test"},
            "mandatory": True,
            "repairable": True,
            "evidence_refs": ["manifest:preprocessor.json"],
        },
        {
            "node_id": "model",
            "node_type": "model",
            "component_id": "classifier",
            "component_version": "1.0",
            "registered_attributes": {"schema": "schema-v3"},
            "observed_attributes": {"schema": "schema-v3"},
            "mandatory": True,
            "repairable": True,
            "evidence_refs": ["manifest:model.json"],
        },
    ],
    "edges": [
        {
            "edge_id": "preprocessor-to-model",
            "source": "preprocessor",
            "target": "model",
            "relation": "transforms",
            "registered_contract": {"compatible": True},
            "observed_contract": {"compatible": True},
            "mandatory": True,
            "repairable": True,
            "evidence_refs": ["manifest:route.json"],
        }
    ],
}

report = DiagnosticService().diagnose(route=route, repair_mode="plan")
print(report.route_status)
print(report.issues)
print(report.minimal_cut)
print(report.repair_plan)
```

Repair execution is never implicit. `repair_mode="execute"` requires an
explicit `RepairExecutionContext`, registered operation, satisfied
preconditions, preserved source artifact, and rollback path.

## Framework Architecture

```text
fuzzyxai
+-- runtime.py                 Public FuzzyXAI facade
+-- adapters/                  Capability-based model adapters
+-- evidence/                  Claims, provenance, and human-facing views
+-- diagnostics/
|   +-- route_graph.py         Canonical route construction
|   +-- validator.py           Local and inter-stage contracts
|   +-- minimal_cut.py         Exact and approximate diagnostic cuts
|   +-- repair_planner.py      Registered repair planning
|   +-- repair_executor.py     Preconditions, execution, rollback
|   +-- recertification.py     Full-route verification after repair
+-- pipelines/                 Executable registered ML/XAI pipelines
+-- external_adapters/         Observation-only adapters for external projects
+-- visualization/             Canonical visual specification and renderers
+-- operators_manifest.yaml    Callable-to-test-to-schema traceability
```

The canonical package lives under `framework/fuzzyxai/fuzzyxai`. The
`fuzzyxai.visualization` namespace is authoritative; `visual` and `viz` exist
only as compatibility shims.

## Contract-Controlled Pipeline

The ML Pipeline v2 route covers:

1. dataset identity and schema;
2. target isolation and class mapping;
3. train/validation/test disjointness and split reproducibility;
4. preprocessor version, fit scope, feature order, and finite output;
5. training configuration, convergence, and training-data identity;
6. model schema and serialized artifact hash;
7. prediction input, output sanity, and object binding;
8. model/explainer compatibility and SHAP reconstruction consistency;
9. explanation provenance, fuzzy representation, reduction, and presentation;
10. registered repair, rollback, graph rebuild, and full recertification.

Contract definitions and acceptance gates are versioned under `protocol/`.
They are not changed after official scoring.

## Latest External Validation

The current framework release was connected through observation-only adapters
to four pinned public ML/XAI examples without changing the registered core:

| Pipeline | Task | Model/explainer boundary |
| --- | --- | --- |
| scikit-learn ColumnTransformer | Binary classification | Mixed preprocessing + Linear SHAP |
| SHAP TreeExplainer fixture | Multiclass classification | Random forest + TreeSHAP |
| MLflow ElasticNet fixture | Regression | Registered artifact + Linear SHAP |
| LIME tabular fixture | Multiclass classification | Independent local explainer |

The locked evaluation contains 40 cases, 200 mode decisions, eight fault
families, two valid controls per pipeline, and 200 MLflow runs. For the full
FuzzyXAI mode, the registered results were:

| Metric | Result |
| --- | ---: |
| Violation recall | 1.00 |
| Cross-stage contract recall | 1.00 |
| Stage / contract / root-cause accuracy | 1.00 / 1.00 / 1.00 |
| Evidence completeness | 1.00 |
| False certifications / false blocks | 0 / 0 |
| Registered repair / full recertification | 1.00 / 1.00 |
| New critical violations after repair | 0 |
| Rollback success | 1.00 |

Status: `FUZZYXAI_EXTERNAL_ML_PIPELINE_VALIDATION_V1_SUPPORTED`.

This is evidence for the registered consistency faults and pinned external
fixtures only. It is not evidence that FuzzyXAI detects arbitrary ML defects,
proves model correctness, reduces engineer time, or replaces MLflow.

Primary evidence:

- [protocol lock](protocol/external_ml_pipeline_v1/)
- [final status](results/external_ml_pipeline_v1/FINAL_STATUS.json)
- [baseline comparison](results/external_ml_pipeline_v1/BASELINE_COMPARISON.csv)
- [final report](reports/external_ml_pipeline_v1/FINAL_REPORT.md)
- [threats to validity](reports/external_ml_pipeline_v1/THREATS_TO_VALIDITY.md)

## Supported Model Surfaces

| Surface | Status |
| --- | --- |
| Python callable and `predict_proba` models | Core contract |
| scikit-learn linear, tree, ensemble, SVM, KNN, Naive Bayes, pipeline | Covered by adapters/tests |
| XGBoost, LightGBM, CatBoost | Optional dependency and optional CI |
| PyTorch, TensorFlow, ONNX Runtime | Optional adapters; support depends on exported evidence |
| Native fuzzy/ANFIS rules | Native rule channel when exposed by the model |
| SHAP and LIME artifacts | Versioned explanation/provenance contracts |

A model being importable is not a support claim. Use
`FuzzyXAI.wrap(model).capability_report()` to inspect the exact available
channels.

## REST, UI, and MLflow

The ML vertical exposes canonical pipeline run, diagnosis, repair, and
recertification operations through its REST application. The UI renders the
same canonical result and does not compute a separate diagnosis. MLflow stores
run parameters, metrics, hashes, and evidence artifacts; FuzzyXAI performs the
contract reasoning over those observations.

```bash
docker compose up --build
```

For focused API and UI checks:

```bash
PYTHONPATH=framework/fuzzyxai:. python -m pytest -q \
  tests/ml_pipeline_v2 \
  tests/integration \
  tests/external_ml_pipeline_v1
```

## Development

Run the smallest relevant checks first:

```bash
PYTHONPATH=framework/fuzzyxai:. python -m pytest -q tests/test_public_framework_api.py
python -m ruff check framework/fuzzyxai/fuzzyxai
python -m compileall -q framework/fuzzyxai/fuzzyxai
make operator-manifest-check
```

Run the full regression before release:

```bash
PYTHONPATH=framework/fuzzyxai:. python -m pytest -q
```

Build a clean source archive from committed files, never from a dirty worktree:

```bash
python scripts/build_framework_release.py
```

The release builder excludes generated ZIP/DOCX/PDF files, caches, private
labels, downloaded model weights, and quarantined site content.

## Repository Map

| Path | Purpose |
| --- | --- |
| `framework/fuzzyxai/fuzzyxai/` | Installable framework |
| `framework/fuzzyxai/operators_manifest.yaml` | Defended operator traceability |
| `tests/` | Unit, integration, regression, and release tests |
| `examples/` | Small executable integrations |
| `apps/` | Canonical demonstration surfaces |
| `protocol/` | Immutable experiment and method locks |
| `results/` | Machine-readable registered outcomes |
| `reports/` | Claim-scoped interpretation and limitations |
| `experiments/` | Reproducible evaluation drivers and fixtures |
| `scripts/` | Audit, validation, and source-release tooling |

Historical studies remain under their protocol/result/report paths for
traceability, but they are not part of the public runtime API. Generated
archives and the removed pre-framework implementation are available from Git
history rather than duplicated in the current tree.

## Evidence and Claim Policy

FuzzyXAI follows four release rules:

1. observable evidence must exist before a claim is scored;
2. held-out labels may be targets but never feature channels;
3. missing evidence is reported as missing or insufficient;
4. negative and blocked studies remain visible and are never rewritten as
   positive results.

See [PROJECT_MEMORY.md](PROJECT_MEMORY.md) for the release boundary and
[RELEASE_STATUS.md](RELEASE_STATUS.md) for registered scientific statuses.

## Documentation

- [Architecture](docs/architecture.md)
- [Adapters](docs/adapters.md)
- [Adapter SDK](framework/fuzzyxai/docs/ADAPTER_SDK.md)
- [Explanation contract](docs/explanation_contract.md)
- [Human explanation layer](docs/human_explanation_layer.md)
- [Visualization](docs/visualization.md)
- [Traceability](docs/traceability.md)
- [Reproducibility](docs/reproducibility.md)
- [Research limitations](docs/research_limitations.md)
- [Contributing](CONTRIBUTING.md)

## Scope and Limitations

FuzzyXAI controls registered evidence and route consistency. It does not:

- prove that a model prediction is true or clinically safe;
- infer causality from feature importance or similarity;
- diagnose arbitrary source-code bugs;
- replace data/version orchestration systems;
- establish human utility without an actual user study;
- generalize a controlled benchmark beyond its locked scope.

Medical, industrial, and safety examples in this repository are research
fixtures unless an associated protocol explicitly says otherwise.

## Citation

Citation metadata is provided in [CITATION.cff](CITATION.cff).

## License

FuzzyXAI is released under the [MIT License](LICENSE). Third-party dataset,
model, and external-project notices are recorded in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and the corresponding protocol
locks.
