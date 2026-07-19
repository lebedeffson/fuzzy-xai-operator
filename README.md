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

## Explain a model

```python
from fuzzyxai import FuzzyXAI

fx = FuzzyXAI.wrap(model, adapter="auto", explain_plan=plan)
result = fx.explain_one(
    X_test[0],
    object_id="case-202",
    reference_data=X_train,
    reference_labels=y_train,
    feature_names=feature_names,
    include_similar_cases=True,
    include_counterfactuals=True,
)

print(result.summary("user"))
print(result.summary("expert"))
print(result.summary("audit"))
print(result.explanation_level)
print(result.overview())
print(result.story())
result.visualize(view="explanation_story", backend="matplotlib", output="explanation.png")
result.visualize(view="decision_evidence", backend="plotly", output="decision.html")
result.inspect("rule:R31").visualize(view="rule_ablation", output="rule_R31.png")
result.export_json("explanation.json")
result.export_html("explanation.html")
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
| Python callable | yes | adapter supplied | no default |
| sklearn linear | yes | native coefficient terms | surrogate rule-like statements |
| sklearn tree | yes | tree structure | native paths |
| RandomForest | yes | global feature importance | native tree paths |
| GradientBoosting | yes | global feature importance | component tree paths |
| HistGradientBoosting | yes | partial explanation | unavailable by default |
| `predict_proba` compatible models | yes | capability dependent | capability dependent |
| ANFIS/fuzzy model with `rules_` | yes | native activations/rules | native |

Torch, Keras, and ONNX adapters are not claimed in this release.

## Validation

```bash
python -m pytest
make operator-manifest-check
make doctorate-release-check
python -m build
```

The machine-verifiable implementation map is [framework/fuzzyxai/operators_manifest.yaml](framework/fuzzyxai/operators_manifest.yaml). The canonical transport object is `ExplanationViewModel` schema `2.0`, shared by Matplotlib, HTML, and MATLAB.

The v1.1 explanation surface is claim-centered: `ExplanationClaim` links every displayed statement to evidence, `ExplanationGraph` is the primary provenance object, and `ExplanationVisualSpec` feeds focused Matplotlib, Plotly, and MATLAB views. Run its controlled evidence gate with:

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

The comprehension study is still `planned_not_run`; the repository therefore claims a testable explanation interface, not demonstrated universal human comprehensibility. Release tag `v1.1.0` is blocked until the external pilot and green `main` CI are recorded.
