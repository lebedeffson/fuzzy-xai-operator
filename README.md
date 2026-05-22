# FuzzyXAI Doctoral Core

Executable research prototype for dissertation chapters 2 and 3.

The project demonstrates a systemic fuzzy-logic operator for compositional explainability and a hierarchy of fuzzy uncertainty representations. It is built as a reproducible software contour: the mathematical objects can be constructed, reduced, compared, composed, calibrated, visualized, and checked by proof scripts.

## What Is Implemented

Chapter 2: systemic explainability operator

- `ExplanationObject`: explanation object `E_k`.
- `ExplainPlan`: reproducible configuration of terms, weights, thresholds, and metadata.
- `SystemOperator`: scalar-risk demo implementation of the operator.
- `compose`: composition of explanation objects with diagnostic state `D_ij`.
- `semantic_disagreement`: semantic disagreement `d_E` and extended disagreement with reduction loss.
- `interpretability_loss` and `interpretability_index`: `L(E)` and `I(E_G)`.
- Beta weight calibration and cross-validation reports.
- Visual composition diagnostics.

Chapter 3: fuzzy representation hierarchy

- `F0`: classical fuzzy representation.
- `IntervalFS`: interval fuzzy representation.
- `HesitantFS`: multiple expert values.
- `NeutrosophicFS`: truth, indeterminacy, falsity with source awareness.
- `MultiLevelFS`: multilevel representation.
- Reductions to `F0` and reduction loss `Delta`.
- Situation profile `P_sit`.
- Pareto selector for minimally sufficient representation class.
- Choice diagnostic `D_choice`.

## Main Demo

Run the focused GUI for a defense-style demonstration:

```bash
pip install -r requirements.txt
python apps/defense_demo.py --port 8085
```

Open:

```text
http://localhost:8085
```

The demo has one workflow:

1. A real sklearn `LogisticRegression` model is trained on `age`, `pressure`, and `marker`.
2. The selected case is passed through the model and receives `risk_score = predict_proba(...)`.
3. The model risk is converted into linguistic terms: low, medium, high.
4. The case is explained as `E_k` with selected representation class `A_k^F`.
5. The system checks whether the risk model and decision module are semantically aligned.

The conflict switch intentionally breaks the interface between components. In that mode the system shows `D_ij` instead of hiding the inconsistency.

The GUI is designed for a non-specialist demonstration: it shows the route `model -> risk_score -> ExplainPlan -> E_k -> A_k^F -> D_ij / I(E_G)`, the current input case, model feature contributions, membership functions, target distribution, selected representation, and the model-to-decision consistency check. Use the presentation switch for projector-friendly sizing, the help button for a short guided tour, and the print button for a browser PDF export.

Key visual explainers in the defense demo:

- Current-case marker on the membership functions: the viewer sees where the selected patient/risk falls on `low`, `medium`, and `high`.
- Model contribution chart: the viewer sees which input features pushed the predicted risk up or down.
- `A_k^F` layer chart: interval uncertainty, expert disagreement, and `T/I/F` conflict are shown as separate visible layers.
- Chapter-3 selection chart: candidate representation classes are plotted by cognitive complexity and expected reduction loss; the selected class is highlighted.
- Composition story graph: model and decision module are connected by a disagreement arrow; conflict mode turns it into diagnostic state `D_ij`.

## Technical Dashboard

The broader NiceGUI dashboard is still available for debugging and extended experiments:

```bash
python apps/nicegui_dashboard.py --port 8080
```

Use it when you need CSV upload, FML synthesis, report inspection, session export, or thesis validation from a GUI. For presentation, prefer `apps/defense_demo.py`.

## Reproducibility Checks

Run all tests:

```bash
PYTHONPATH=. pytest -q
```

Run proof scripts and regenerate reports:

```bash
PYTHONPATH=. python proofs/run_all_proofs.py
```

Run final thesis validation:

```bash
PYTHONPATH=. python proofs/validate_thesis_examples.py
PYTHONPATH=. python examples/thesis_demo.py
```

Current expected status:

```text
24 passed
thesis validation: PASS
thesis demo: PASS
```

## Reports

Generated artifacts are saved in `reports/`.

Important files:

- `reports/thesis_validation.md`: numerical validation for chapter examples.
- `reports/thesis_demo_report.md`: end-to-end route report.
- `reports/thesis_demo_composition_graph.html`: interactive composition graph.
- `reports/chapter2_calibration_report.json`: beta calibration report.
- `reports/breast_cancer_benchmark.md`: minimal medical-like benchmark summary.

## Minimal API Example

```python
from fuzzyxai import FuzzyXAIPipeline

pipe = FuzzyXAIPipeline.from_data(X_train, y_train, mode="audit")
result = pipe.explain_scalar_risk(
    0.72,
    metadata={
        "has_intervals": True,
        "num_experts": 2,
        "source_conflict": True,
        "requires_audit": True,
    },
)

print(result.report)
```

## Project Structure

```text
fuzzyxai/
  core/          explanation objects, operator, composition, distances
  hierarchy/     F0, interval, hesitant, neutrosophic, multilevel classes
  selection/     profile builder, compatibility, Pareto selector
  calibration/   beta calibration and cross-validation
  visual/        Plotly figures for membership functions and composition
  demo/          deterministic synthetic data and demo builders
apps/
  defense_demo.py       main presentation GUI
  nicegui_dashboard.py  extended technical dashboard
proofs/
  chapter2_operator_proof.py
  chapter2_calibration_proof.py
  chapter3_hierarchy_proof.py
  validate_thesis_examples.py
examples/
  thesis_demo.py
benchmarks/
  breast_cancer_benchmark.py
tests/
  pytest checks for core behavior and GUI demo logic
```

## Chapter Mapping

See `CHAPTER_MAPPING.md` for a direct mapping between dissertation objects and implementation files.

For presentation and implementation explanation, see:

- `PRESENTATION.md`: defense script and what to say while showing the GUI.
- `IMPLEMENTATION_SUMMARY.md`: technical summary of what is implemented and where.

## Scientific Scope

This repository is a dissertation reproducibility layer, not a certified clinical system. The medical-like benchmark is a technical proof of concept. Full domain validation, expert review, and regulated deployment are outside this prototype.
