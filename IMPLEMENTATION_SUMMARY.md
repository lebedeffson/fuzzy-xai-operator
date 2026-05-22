# Implementation Summary

This document describes what is implemented in the repository and how it maps to dissertation chapters 2 and 3.

## Repository Goal

The repository provides a compact executable implementation of:

- a systemic fuzzy-logic operator for compositional explainability;
- a hierarchy of fuzzy uncertainty representations;
- a GUI demonstration suitable for presenting the implementation;
- proof scripts and reports for reproducibility.

## Main User-Facing Entry Point

`apps/defense_demo.py`

Purpose:

- show one complete route from data to final explanation consistency check;
- avoid a large multi-page technical interface;
- make the chapter 2 and 3 connection visible on one screen.

Visible route:

```text
Data -> ExplainPlan -> E_k -> A_k^F -> D_ij / I(E_G)
```

Key UI elements:

- membership function plot for linguistic terms;
- input case card;
- selected fuzzy representation class;
- representation visualization;
- active rule table;
- model-to-decision consistency diagram;
- conflict switch that produces `D_ij`;
- presentation mode;
- guided tour;
- print and JSON export.

## Technical Dashboard

`apps/nicegui_dashboard.py`

Purpose:

- technical inspection;
- CSV upload;
- broader page-based exploration;
- reports and session export;
- thesis validation from GUI.

For presentation, prefer `apps/defense_demo.py`.

## Chapter 2 Implementation

### Explanation Object

File:

- `fuzzyxai/core/explanation_object.py`

Implements:

- terms;
- representation `A_k^F`;
- rule base;
- activations;
- uncertainty;
- trace;
- reduction loss.

### ExplainPlan

Files:

- `fuzzyxai/core/explain_plan.py`
- `fuzzyxai/core/plan_builder.py`

Implements:

- beta weights for disagreement;
- lambda weights for interpretability loss;
- eta weights for uncertainty aggregation;
- thresholds;
- metadata;
- automatic plan construction from tabular data.

### System Operator

File:

- `fuzzyxai/core/system_operator.py`

Implements:

- scalar-risk demonstration route;
- fuzzy membership values;
- rule activation;
- uncertainty aggregation;
- construction of `ExplanationObject`.

### Composition And Diagnostics

Files:

- `fuzzyxai/core/composition.py`
- `fuzzyxai/core/diagnostics.py`

Implements:

- composition of explanation objects;
- propagation of uncertainty;
- diagnostic state `D_ij` when linguistic interfaces do not match.

### Semantic Disagreement And Index

File:

- `fuzzyxai/core/trust_evaluator.py`

Implements:

- representation distance;
- rule distance;
- activation distance;
- uncertainty distance;
- trace distance;
- reduction loss contribution;
- semantic disagreement `d_E`;
- interpretability loss `L`;
- index `I(E_G)`.

### Calibration

Files:

- `fuzzyxai/calibration/weight_calibration.py`
- `fuzzyxai/calibration/cross_validation.py`
- `proofs/chapter2_calibration_proof.py`

Implements:

- projected-gradient calibration of beta weights on the simplex;
- cross-validation on expert-like disagreement labels;
- JSON report generation.

## Chapter 3 Implementation

### Representation Classes

Files:

- `fuzzyxai/hierarchy/f0.py`
- `fuzzyxai/hierarchy/interval.py`
- `fuzzyxai/hierarchy/hesitant.py`
- `fuzzyxai/hierarchy/neutrosophic.py`
- `fuzzyxai/hierarchy/multilevel.py`

Implemented classes:

- `F0`;
- `IntervalFS`;
- `HesitantFS`;
- `NeutrosophicFS`;
- `MultiLevelFS`.

### Reductions And Loss

Files:

- `fuzzyxai/hierarchy/reductions.py`
- `fuzzyxai/hierarchy/meta_reducer.py`

Implements:

- reductions to `F0`;
- reduction loss `Delta`;
- policy selection for reduction.

### Profile And Class Selection

Files:

- `fuzzyxai/selection/profile_builder.py`
- `fuzzyxai/selection/pareto_selector.py`
- `fuzzyxai/selection/compatibility.py`
- `fuzzyxai/selection/choice_diagnostic.py`

Implements:

- situation profile `P_sit`;
- coverage checks;
- Pareto front;
- minimally sufficient representation selection;
- diagnostic `D_choice`;
- compatibility checks for multilevel synthesis.

## Cross-Chapter Integration

The main integration point is:

```text
E_k.representation = A_k^F
```

This means the explanation object from chapter 2 can carry any representation class selected by chapter 3.

The reduction loss is then included in semantic disagreement:

```text
d_E_ext = d_E + beta_reduction * Delta
```

In code this is implemented through:

- `ExplanationObject.reduction_loss`;
- `ExplainPlan.with_reduction_weight`;
- `semantic_disagreement(... beta['reduction'])`;
- `compose(...)`.

## Proof And Validation Artifacts

Tests:

- `tests/`

Proof scripts:

- `proofs/chapter2_operator_proof.py`
- `proofs/chapter2_calibration_proof.py`
- `proofs/chapter3_hierarchy_proof.py`
- `proofs/validate_thesis_examples.py`
- `proofs/run_all_proofs.py`

End-to-end demo:

- `examples/thesis_demo.py`

Benchmark:

- `benchmarks/breast_cancer_benchmark.py`

Reports:

- `reports/thesis_validation.md`
- `reports/thesis_demo_report.md`
- `reports/proof_summary.md`
- `reports/breast_cancer_benchmark.md`

## Current Verification Status

Expected local check:

```bash
PYTHONPATH=. pytest -q
```

Expected result:

```text
22 passed
```

Expected validation:

```bash
PYTHONPATH=. python proofs/validate_thesis_examples.py
```

Expected result:

```text
status: PASS
total_checks: 20
failed_checks: 0
```

## What Is Not Claimed

The repository does not claim:

- clinical readiness;
- universal explainability;
- expert validation of medical decisions;
- full production deployment.

It claims:

- executable construction of dissertation objects;
- reproducible numerical checks;
- visual demonstration of the operator and hierarchy;
- diagnostic detection of semantic breaks in an explanation chain.
