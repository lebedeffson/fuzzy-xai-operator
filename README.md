# FuzzyXAI Research Framework

FuzzyXAI is an evidence-first research framework for tracing a model decision from input data through training history, learned rules or concepts, local evidence, operator diagnostics, and the final action.

The framework implements the mathematical route from dissertation chapters 2-3:

```text
E -> T -> gamma -> Delta -> rDelta -> rho -> D -> chi -> action
```

It does not treat a model score as permission to act. Missing evidence produces `review` or `insufficient_evidence`, never invented `gamma`, `Delta`, `rho`, similarity, or rule importance.

## Install

```bash
python -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
```

## Empirical Validation

Run the small one-thread protocol check locally:

```bash
make empirical-smoke
```

Run the full E1-E8 dissertation candidate in CI or an isolated container:

```bash
make reproduce-dissertation
docker compose run --rm reproduce
```

Controlled 10,000-object datasets validate protocol behavior, not external-domain
generalization. External comprehension and domain reviews remain release blockers
until independent responses are available.

## Chapter 4 v13 practical evaluation

The frozen v13 protocol adds a modern pretrained text contour without changing the
negative v1.3.0 findings. It uses the pinned AG News dataset revision and a frozen
DistilBERT model, produces real Integrated Gradients and token-masking explanations,
compares policies at matched review budgets, evaluates registered/compositional/
held-out route faults, and decomposes end-to-end runtime.

Install the pinned research dependencies into an existing Python 3.12 environment:

```bash
python -m pip install -r config/chapter4_v13_requirements.txt
python -m pip install -e .
```

Run the lightweight contract smoke used by CI:

```bash
make chapter4-v13-smoke
```

Run the complete local experiment once:

```bash
make reproduce-chapter4-v13 CHAPTER4_V13_PYTHON=/path/to/python3.12
```

The full command downloads data and model weights from their pinned upstream
revisions. AG News is not redistributed because its upstream dataset card reports
an unknown license. The pinned model card also does not state a license for the
weights; see `THIRD_PARTY_NOTICES.md`. Raw data, sealed labels and model caches stay under ignored
`artifacts/chapter4_v13/` subdirectories. Every released numeric table cell is
mapped to raw evidence and SHA256 in `artifacts/chapter4_v13/evidence_map.json`.

Known boundaries: the old H3, H5-P and H6-general claims remain unsupported; the
5-million-object benchmark describes only the cached operator layer; the v13 user
study package is a future protocol and contains no participant evidence.

Public reviewer artifacts are committed under
[`dissertation_artifacts/chapter4_v13/final`](dissertation_artifacts/chapter4_v13/final/).
The directory contains the DOCX/PDF, the complete five-budget policy table, full
runtime statistics and raw repetitions, the held-out-fault status, leakage audit,
evidence map, validation report, checksums and the downloadable evidence ZIP.
The code is distributed under the MIT license in [`LICENSE`](LICENSE); upstream
dataset and model license boundaries are recorded in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## Explain a model

```python
from fuzzyxai import ExplainPlan, FuzzyXAI

plan = ExplainPlan.default()
plan.domain_language = {
    "features": {
        "fracture_density": {
            "label": "трещиноватость породы",
            "high_text": "трещин больше, чем в большинстве исследованных участков",
        }
    },
    "classes": {1: {"label": "повышенный риск"}},
    "actions": {"review": {"label": "проверить специалистом"}},
}

fx = FuzzyXAI.wrap(model, adapter="auto", task="auto", explain_plan=plan)
result = fx.explain_one(
    X_test[0],
    object_id="case-202",
    reference_data=X_train,
    reference_labels=y_train,
    feature_names=feature_names,
    include_similar_cases=True,
    include_counterfactuals=True,
)

human = result.explain_for(audience="domain_user", language="ru")
print(human.decision.explanation)
print([reason.explanation for reason in human.main_reasons])
print(human.reliability.explanation)
print(human.recommended_action.explanation)

print(result.summary(audience="domain_user", detail="short"))
print(result.summary(audience="ml_engineer", detail="full"))
print(result.explanation_level)
print(result.overview())
print(result.story())
result.visualize(view="explanation_story", backend="matplotlib", output="explanation.png")
result.visualize(view="decision_evidence", backend="plotly", output="decision.html")
result.inspect("rule:R31").visualize(view="rule_ablation", output="rule_R31.png")
result.export_json("explanation.json")
result.export_html("explanation.html")
```

The selected family, native/derived/surrogate channels, missing evidence, and quality gates are inspectable:

```python
print(fx.capability_report())
print(result.quality_report())
print(result.why_not(target_class=0))

batch = fx.explain_batch(X_test[:10])
global_result = fx.explain_global(X_train, y_train)
comparison = FuzzyXAI.compare_models(
    {"linear": linear_model, "tree": tree_model},
    item=X_test[0],
    reference_data=X_train,
    reference_labels=y_train,
)
```

## Observe training

```python
training = fx.observe_training(history={
    "objects": {"85": epoch_records},
    "global_metric": global_accuracy,
    "subgroup_metrics": {"rare_subtype": rare_recall},
})

training.find_forgotten_objects()
training.find_averaged_subgroups()
training.extract_model_rules()
training.plot_object_trajectory("85", "object_85.png")
```

The executable controlled object-85 protocol is:

```bash
PYTHONPATH=framework/fuzzyxai python examples/object_85_training_trace.py \
  --output-dir release_evidence/object_85
```

## Supported model level

| Model family | Prediction | Local/model evidence | Rules |
| --- | --- | --- | --- |
| Python callable | verified | perturbation/reference channels when supplied | no default |
| sklearn linear | verified | native coefficient terms and margin | no native rules |
| sklearn tree | verified | exact local decision path | native paths |
| sklearn ensembles | verified | votes/disagreement and model-specific paths | family dependent |
| sklearn SVM/KNN/Naive Bayes | verified | family-specific evidence or measured surrogate | no invented rules |
| sklearn Pipeline | verified | transformed and source-feature provenance | estimator dependent |
| `predict_proba` compatible models | verified generic contract | capability dependent | capability dependent |
| ANFIS/fuzzy model with `rules_` | verified contract | native activations/rules | native |
| XGBoost/LightGBM/CatBoost | optional CI | native contribution APIs | native tree runtime |
| PyTorch/TensorFlow | optional CI | native-gradient-derived channels | architecture dependent |
| ONNX Runtime | optional CI | exported runtime outputs | missing unless exported |

`verified` applies to the exact configurations in
`release_evidence/model_universality/support_matrix.csv`. Optional code presence alone is not a support claim.

## Reproducible Chapter 4 candidate

Run the low-load stages separately:

```bash
make model-universality
make external-validation-gates
make chapter4-final-candidate
```

The computed evidence can pass while the final release remains `BLOCKED`. A real six-person A/B pilot and an
independent domain-language review cannot be generated by code and remain mandatory before a final `v1.2.0` tag.

Torch, Keras, and ONNX adapters are not claimed in this release.

## Validation

```bash
python -m pytest
make operator-manifest-check
make doctorate-release-check
python -m build
```

The machine-verifiable implementation map is [framework/fuzzyxai/operators_manifest.yaml](framework/fuzzyxai/operators_manifest.yaml). The canonical transport object is `ExplanationViewModel` schema `2.0`, shared by Matplotlib, HTML, and MATLAB.

The v1.2 explanation surface adds `HumanExplanation` above the claim graph. The first level answers decision, reasons, concerns, reliability, and action without exposing internal identifiers. Every card still links to claims and evidence for inspection. E0-E5 describes available evidence; `audience` controls how the same evidence is communicated. Run the controlled evidence gate with:

```bash
make explanation-experience-evidence
pytest -q tests/test_explanation_experience.py
```

The available views are `explanation_story`, `data_profile`, `training_trace`, `knowledge_atlas`, `decision_evidence`, `similar_cases`, `counterfactual`, `rule_ablation`, `provenance`, and `audit`. Missing channels are disclosed through E0-E5 rather than replaced by an aggregate interpretability score.

## Documentation

- [Architecture](docs/architecture.md)
- [Adapters](docs/adapters.md)
- [Operators](docs/operators.md)
- [Explanation contract](docs/explanation_contract.md)
- [Human Explanation Layer](docs/human_explanation_layer.md)
- [Training observer](docs/training_observer.md)
- [Visualization](docs/visualization.md)
- [Comprehension study protocol](docs/explanation_comprehension_protocol.md)
- [Traceability](docs/traceability.md)
- [Reproducibility](docs/reproducibility.md)
- [Research limitations](docs/research_limitations.md)

## Website status

The generated DubnaXAI website prototype is quarantined in the git branch `archive/site-prototype-cab4018`. It is intentionally excluded from the framework release until the API and explanation evidence are stable.

## Research scope

This software is a research framework. Medical examples are not clinical conclusions, surrogate rules are labelled as surrogate, similarity is not causality, and model quality must be established by a separate benchmark.

## Explanation Experience release boundary

The ten focused Matplotlib and Plotly views consume `ExplanationVisualSpec` schema `1.1`. `result.inspect(...)` returns a typed `InspectionResult`; `result.explanation_graph.validate_reachability()` verifies the evidence-to-action route. Cross-model controlled evidence is generated under `release_evidence/explanation_experience/cross_model/`.

The comprehension study is still `planned_not_run`; the repository therefore claims a testable human-explanation interface, not demonstrated universal human comprehensibility. Release tag `v1.2.0` is blocked until the external pilot and green `main` CI are recorded.

## Empirical validation gate

The controlled object-85 story is now explicitly separated from the measured case `case_real_001`.
Reproduce the 30-checkpoint training run, automatic forgetting selection, native tree-rule ablation,
cross-model capability matrix, and Chapter 4 package with:

```bash
make empirical-validation-check
```

Measured artifacts are written under `release_evidence/empirical_experiments/` and
`release_evidence/chapter4_empirical_validation/`. The benchmark uses the UCI Breast Cancer Wisconsin
(Diagnostic) data only as a methodological research task; it is not clinical validation. The independent
comprehension pilot and regulated-domain dictionary review remain incomplete, so `v1.2.0rc3` is an
untagged computational candidate rather than a completed human-validation release.

## H10 v19 audit integration

The `feat/h10-audit-confirmatory-v19` branch integrates the supplied H10
auditor and frozen one-opening outputs without opening the sealed vault again.
The original handoff files are preserved under
`artifacts/h10_v19/imported_handoff/`.

Repository-level methodology review found that the oracle imports no evaluated
H10 implementation, but source and repair truth are assigned through a static
catalog that semantically duplicates the evaluated auditor taxonomy. The
numerical source/repair differences can be reproduced, but H10-L and H10-R are
therefore marked `invalid_methodology` for scientific release. H10-C remains a
secondary descriptive cut result, H10-U remains descriptive, and H10-T is a
deterministic trace result only.

Use an existing Python environment; no additional virtual environment is
created by these targets:

```bash
make h10-smoke H10_PYTHON=/path/to/python
make reproduce-h10 H10_PYTHON=/path/to/python
```

## Diagnostic framework v21 alpha

The branch `feat/diagnostic-framework-v21` exposes structural route diagnostics
through the canonical `FuzzyXAI` facade:

```python
report = FuzzyXAI().diagnose(route=route, repair_mode="plan")
print(report.summary("user"))
```

The implementation validates a full registered route graph, finds an exact or
explicitly approximate minimal diagnostic cut, proposes provider-bound repair
steps, and recertifies only after explicit external execution. Production code
does not read Gold mutation logs and never copies `expected` values into
`observed` values.

```bash
make diagnostic-v21-check DIAGNOSTIC_PYTHON=/home/lebedeffson/Code/venv/bin/python
```

This is an exploratory alpha implementation. The earlier H10-C result remains
`BLOCKED_PRECONFIRMATORY`; the draft H10-C2 protocol cannot score sealed cases
until power analysis and independent two-reviewer adjudication are complete.
See `docs/DIAGNOSTIC_FRAMEWORK_RU.md`.

`reproduce-h10` rebuilds statistics, replay summaries, tables, figures,
evidence mapping, validation reports, and release archives from committed
frozen outputs. It never calls the confirmatory scoring runner or opens a
label vault.
