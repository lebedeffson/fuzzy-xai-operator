# FuzzyXAI: Presentation Guide

This file is a short script for presenting the implementation of dissertation chapters 2 and 3.

## One-Sentence Description

FuzzyXAI is a reproducible prototype that turns model outputs into fuzzy explanation objects, selects an uncertainty representation, composes explanations across system components, and detects semantic breaks in the explanation chain.

## What To Show First

Run:

```bash
python apps/defense_demo.py --port 8085
```

Open:

```text
http://localhost:8085
```

The visible route is:

```text
Data -> ExplainPlan -> E_k -> A_k^F -> D_ij / I(E_G)
```

This route is the whole story of the implementation.

## Demo Script

1. Start with the first block: data become linguistic terms.

Say:

> Here the system automatically builds linguistic terms from data. A numeric feature such as `risk_score` is converted into fuzzy terms: low, medium, and high. This is the data-driven part of the ExplainPlan.

Show:

- membership functions;
- target distribution;
- quantile table for terms.

2. Move to the second block: one case becomes an explanation object.

Say:

> For one case, the operator builds an explanation object `E_k`. The object contains active rules, uncertainty, trace information, and an uncertainty representation selected from the hierarchy in chapter 3.

Show:

- input case card;
- selected class, usually `FML-audit`;
- uncertainty `u(E_k)`;
- reduction loss `Delta`;
- representation chart;
- active rules.

3. Move to the third block: components are checked for semantic consistency.

Say:

> The important point is that the explanation is checked across components, not only inside one model. The risk model and decision module are composed. The system computes semantic disagreement and either produces an integrated explanation index or a diagnostic state.

Show:

- model-to-decision diagram;
- disagreement value;
- loss and final index `I(E_G)`;
- table with terms on both sides.

4. Toggle `показать конфликт`.

Say:

> Now we intentionally break the interface between components. Instead of hiding the problem, the operator produces diagnostic state `D_ij`. This shows where explainability fails.

Show:

- red status block;
- `D_ij`;
- incompatible terms;
- blocked result.

5. Use presentation mode if showing on a projector.

Say:

> The same implementation can be shown in presentation mode, while the full technical dashboard remains available for deeper inspection.

## What Is Implemented For Chapter 2

Chapter 2 is represented by the systemic fuzzy-logic operator.

Implemented objects:

- `E_k`: explanation object.
- `ExplainPlan`: reproducible configuration of terms, weights, thresholds, and metadata.
- `Omega`: system operator for producing an explanation object.
- `compose(E_i, E_j)`: composition of explanations.
- `d_E`: semantic disagreement.
- `L(E)`: interpretability loss.
- `I(E_G)`: final interpretability index.
- `D_ij`: diagnostic state for incompatible interfaces.
- beta weight calibration from expert-like labels.

Main files:

- `fuzzyxai/core/explanation_object.py`
- `fuzzyxai/core/explain_plan.py`
- `fuzzyxai/core/system_operator.py`
- `fuzzyxai/core/composition.py`
- `fuzzyxai/core/trust_evaluator.py`
- `fuzzyxai/calibration/`
- `proofs/chapter2_operator_proof.py`
- `proofs/chapter2_calibration_proof.py`

## What Is Implemented For Chapter 3

Chapter 3 is represented by the hierarchy of fuzzy uncertainty representations.

Implemented classes:

- `F0`: classical fuzzy representation.
- `IntervalFS`: interval representation.
- `HesitantFS`: several expert values.
- `NeutrosophicFS`: truth, indeterminacy, falsity with source awareness.
- `MultiLevelFS`: multilevel representation.

Implemented mechanisms:

- situation profile `P_sit`;
- Pareto selection of the minimally sufficient class;
- reductions to `F0`;
- reduction loss `Delta`;
- choice diagnostic `D_choice`;
- cross-chapter use of `A_k^F` inside `E_k`.

Main files:

- `fuzzyxai/hierarchy/`
- `fuzzyxai/selection/profile_builder.py`
- `fuzzyxai/selection/pareto_selector.py`
- `fuzzyxai/selection/compatibility.py`
- `fuzzyxai/hierarchy/meta_reducer.py`
- `proofs/chapter3_hierarchy_proof.py`
- `examples/chapter3_end_to_end.py`

## Reproducibility Proof

Run:

```bash
PYTHONPATH=. pytest -q
PYTHONPATH=. python proofs/run_all_proofs.py
PYTHONPATH=. python proofs/validate_thesis_examples.py
PYTHONPATH=. python examples/thesis_demo.py
```

Expected result:

```text
22 passed
thesis validation: PASS
thesis demo: PASS
```

Important reports:

- `reports/thesis_validation.md`
- `reports/thesis_demo_report.md`
- `reports/chapter2_calibration_report.json`
- `reports/breast_cancer_benchmark.md`

## Strong Claims You Can Make

- The code is not only a visualization; it constructs the mathematical objects from chapters 2 and 3.
- The explanation chain is reproducible because the plan, weights, traces, reductions, and reports are saved.
- The operator detects semantic inconsistency between components through `D_ij`.
- The hierarchy in chapter 3 is not decorative: it is used to select `A_k^F`, compute reduction loss, and extend disagreement.
- The implementation is covered by tests and proof scripts.

## Limitations To State Honestly

- This is a dissertation prototype, not a certified clinical decision system.
- The medical-like benchmark is a technical proof of concept.
- Expert validation and regulated deployment are outside this repository.
- The demo data are synthetic unless a user uploads real data in the technical dashboard.

## Short Closing

> The contribution is not only a local explanation for one model. The implementation shows how explanations can be represented, reduced, composed, checked for semantic disagreement, and diagnosed across an AI pipeline.
