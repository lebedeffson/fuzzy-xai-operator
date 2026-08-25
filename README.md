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

## Project Support

The research was carried out under the state assignment of the Ministry of
Science and Higher Education of the Russian Federation, theme No.
124112200072-2.

See [FUNDING.md](FUNDING.md) for the acknowledgment text and project scope.

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
.venv/bin/python -m pip install -e ".[torch]"
.venv/bin/python -m pip install -e ".[images]"        # Pillow, for image object_representation
```

Python 3.10 or newer is required. Locked research environments are recorded in
`requirements.lock`, `uv.lock`, and the corresponding protocol directories.

Local verbalization via Ollama needs no Python dependency at all — the
backend talks HTTP directly (stdlib `urllib`), so there is no `ollama`
package to install; only the base install above is required for the whole
library to work, verbalization included (deterministically, offline).

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

fx = FuzzyXAI.wrap(
    model,
    reference_data=X_train,
    reference_labels=y_train,
)
result = fx.explain_one(X_test[0], object_id="patient-0001")

print(result.summary())
print(result.similar_cases)       # reference-corpus objects most like this one
result.visualize()

compact = result.export_json("explanation_compact.json", detail="compact")
audit = result.export_json("explanation_audit.json", detail="audit")
```

The prediction is available even when optional explanation channels are
missing. Unsupported claims stay absent and are listed in the quality report.

### API surface

| Call | Role |
| --- | --- |
| `result.summary()` | Deterministic short text — offline, no network, always available. |
| `result.story()` | Deterministic narrative across the evidence route (data -> training -> knowledge -> decision -> action). |
| `result.verbalize()` / `.verbalize_detailed()` | Optional natural-language *presentation* of the already-built explanation — see below; no backend means byte-identical to `summary()`. |
| `result.visualize(view=..., backend=...)` | Renders one view (`explanation_story`, `object_representation`, `similar_cases`, ...) via matplotlib or plotly. |
| `result.object_representation` | The raw explained object (text/tabular/image) with evidence overlaid — `None` if no raw object and no data evidence exist. |
| `result.similar_cases` | Reference-corpus objects most similar to this one; empty unless a reference corpus was registered. |
| `result.export_json(path, detail=..., include_raw=...)` | Serializes one of three projections (`"compact"`, `"standard"`, `"audit"` — default) of the *same* canonical result; see below. |

**SLM does not independently infer why a model made a prediction. It only
verbalizes an explanation that has already been constructed from typed
evidence** — the prediction, evidence extraction, and diagnosis never depend
on whether a verbalizer backend is configured.

**Similarity evidence is comparative evidence and is not causal** unless the
model adapter explicitly establishes prototype/example-based inference — "a
similar training example was found" is never phrased as "the model chose
this because it resembles that example."

**`include_raw=False` (the default on every export) removes the raw object
payload from the JSON** — it is not a general PII anonymizer; structured
evidence derived from the raw object (spans, offsets, tabular rows, feature
values, region masks) is left untouched.

### Compact / standard / audit exports (P2)

`export_json`/`to_dict` accept `detail=` — three read-only projections of
the same already-computed result; none of them re-runs `explain()`, so
prediction/claims/similar_cases/action are identical across all three:

```python
result.export_json("out.json", detail="compact")   # prediction, top evidence, similar cases, uncertainty, action
result.export_json("out.json", detail="standard")   # + all claims, one audience's HumanExplanation, visual metadata
result.export_json("out.json", detail="audit")      # the full canonical payload (default, unchanged since before P2)
```

`compact` is roughly two orders of magnitude smaller than `audit` for a
typical tabular object — see [examples/01_tabular_sklearn.py](examples/01_tabular_sklearn.py).

### Raw-object representation and optional local verbalization

`explain_one` accepts an optional `raw_object` (a text string, or a 2D/3D
image array) so the explanation package can show evidence overlaid on the
object itself, safely HTML-escaped for text, rather than only on abstract
feature names. Tabular inputs get an honest feature/value/contribution table
by default even without a raw object:

```python
result = fx.explain_one(x, object_id="doc-1", raw_object=raw_text)

result.object_representation["modality"]           # "text", "tabular", or "image"
result.object_representation["highlighted_html"]    # safe to embed, spans HTML-escaped first
result.visualize(view="object_representation", backend="matplotlib")
```

For images, `region_masks={"name": boolean_mask, ...}` reports each named
region's real geometry (measured from the mask, never a fabricated heatmap —
FuzzyXAI has no built-in per-pixel attribution method) and, when your
contribution mapping has a matching entry, its measured contribution — see
[examples/04_image_explanation.py](examples/04_image_explanation.py).

`result.summary()` and `result.verbalize()` are deterministic by default —
**no network call, no LLM, no new dependency** is ever required to use
FuzzyXAI. `verbalize()` becomes a local-LLM rephrasing only when you pass an
explicit backend:

```python
# 1. No backend — deterministic, works everywhere, offline
text = result.verbalize()

# 2. Local Ollama backend — you install and run it yourself; the
#    library never downloads a model or starts a server for you
from fuzzyxai.verbalization.backends import OllamaBackend

backend = OllamaBackend(model="qwen3:1.7b")  # or FUZZYXAI_OLLAMA_MODEL env var
details = result.verbalize_detailed(backend=backend)
print(details.text, details.status)  # status: deterministic | generated | fallback | rejected

# 3. Your own backend — anything implementing VerbalizationBackend.generate()
class MyBackend:
    model = "my-model"
    def generate(self, prompt: str, *, response_schema=None) -> str: ...

result.verbalize_detailed(backend=MyBackend())
```

Not sure whether a local Ollama is reachable and the model is pulled?

```bash
python -m fuzzyxai.verbalization doctor
```

Two verbalization modes are available (`SLMVerbalizer(backend, mode=...)`,
default `"strict"`):

- **`strict`** — the backend only picks an *order* over already-verified
  claim IDs and a connector style; the final text is assembled by a
  deterministic renderer purely from vetted claim text. No token the backend
  writes can reach the output — a structural guarantee.
- **`rewrite`** (opt-in) — the backend writes free text with per-sentence
  claim attribution, checked afterward by surface guards (no new number, no
  new feature/class name, no unlicensed causal/certainty language). These
  are **surface checks, not proof of semantic entailment** — `rewrite`
  output is not "grounded", it has "passed surface grounding checks."

Install steps for Ollama differ by OS — see
[docs.ollama.com/linux](https://docs.ollama.com/linux),
[docs.ollama.com/macos](https://docs.ollama.com/macos), or
[ollama.com](https://ollama.com) for Windows; there is no single command
that works identically everywhere. `qwen3:1.7b` is the recommended default
(small, multilingual); `qwen3:0.6b` is a lighter but weaker alternative. See
[examples/text_explanation_with_verbalizer.py](examples/text_explanation_with_verbalizer.py)
for a runnable version of all three scenarios above.

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

The current software boundary is summarized in
[RELEASE_STATUS.md](RELEASE_STATUS.md). Detailed machine-readable outcomes and
their limitations remain colocated with the corresponding files under
`results/`, `reports/`, and `protocol/`.

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
- [Funding and acknowledgment](FUNDING.md)
- [Contributing](CONTRIBUTING.md)

Runnable examples, each self-contained and using only the canonical API:
[01 tabular](examples/01_tabular_sklearn.py) ·
[02 similarity](examples/02_tabular_similarity.py) ·
[03 text](examples/03_text_explanation.py) ·
[04 image](examples/04_image_explanation.py) ·
[05 strict verbalizer](examples/05_strict_verbalizer.py) ·
[06 custom adapter](examples/06_custom_model_adapter.py) ·
[07 rule-based model](examples/07_rule_based_model.py)

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

Fuzzy/rule-based models are supported through a generic contract: any
adapter can supply an `activated_rules` channel (rule id, antecedent terms
with real `[0, 1]` membership degrees, activation strength, conclusion) —
not tied to any specific ANFIS library. See
[examples/07_rule_based_model.py](examples/07_rule_based_model.py) for a
self-contained Gaussian-membership example (real membership functions, real
product T-norm rule firing, real weighted-average defuzzification).

## Citation

Citation metadata is provided in [CITATION.cff](CITATION.cff).

## License

FuzzyXAI is released under the [MIT License](LICENSE). Third-party dataset,
model, and external-project notices are recorded in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and the corresponding protocol
locks.
