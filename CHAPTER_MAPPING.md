# Mapping between dissertation chapters and implementation files

## Chapter 2

| Dissertation object | Implementation | Proof |
|---|---|---|
| Explanation object `E_k` | `fuzzyxai/core/explanation_object.py` | `tests/test_composition.py` |
| ExplainPlan | `fuzzyxai/core/explain_plan.py` | `tests/test_v3_pipeline.py` |
| Plan generation from data | `fuzzyxai/core/plan_builder.py` | `apps/explainplan_builder.py` |
| System operator `Omega` | `fuzzyxai/core/system_operator.py` | `proofs/chapter2_operator_proof.py` |
| Composition `E_i o E_j` | `fuzzyxai/core/composition.py` | `proofs/chapter2_operator_proof.py` |
| Semantic disagreement `d_E` | `fuzzyxai/core/trust_evaluator.py` | `reports/chapter2_proof.json` |
| Loss and index `L(E), I(E)` | `fuzzyxai/core/trust_evaluator.py` | `reports/chapter2_proof.json` |
| Diagnostics `D_ij` | `fuzzyxai/core/diagnostics.py` | `tests/test_composition.py` |
| Beta calibration | `fuzzyxai/calibration/` | `proofs/chapter2_calibration_proof.py` |
| Composition graph | `fuzzyxai/visual/` | `apps/composition_viewer.py` |

## Chapter 3

| Dissertation object | Implementation | Proof |
|---|---|---|
| Classical fuzzy representation `F0` | `fuzzyxai/hierarchy/f0.py` | `tests/test_reductions.py` |
| Interval representation `F_I` | `fuzzyxai/hierarchy/interval.py` | `tests/test_reductions.py` |
| Hesitant representation `F_H` | `fuzzyxai/hierarchy/hesitant.py` | `tests/test_reductions.py` |
| Neutrosophic source-aware representation | `fuzzyxai/hierarchy/neutrosophic.py` | `tests/test_reductions.py` |
| General source decorator | `fuzzyxai/hierarchy/source_annotation.py` | hierarchy API |
| Multilevel configuration `F_ML` | `fuzzyxai/hierarchy/multilevel.py` | `examples/chapter3_end_to_end.py` |
| Reductions and `Delta` | `fuzzyxai/hierarchy/reductions.py`, `meta_reducer.py` | `reports/chapter3_end_to_end_report.json` |
| Profile `P_sit` | `fuzzyxai/selection/profile_builder.py` | `proofs/chapter3_hierarchy_proof.py` |
| Pareto class selection | `fuzzyxai/selection/pareto_selector.py` | `tests/test_selector.py` |
| Choice diagnostic `D_choice` | `fuzzyxai/selection/choice_diagnostic.py` | `reports/chapter3_end_to_end_report.json` |
| Compatibility and FML synthesis | `fuzzyxai/selection/compatibility.py` | `reports/chapter3_end_to_end_report.json` |

## Cross-chapter link

| Link | Implementation | Proof |
|---|---|---|
| Replacement of `mu_k` by `A_k^F` | `ExplanationObject.representation` | `examples/chapter3_end_to_end.py` |
| Reduction loss in `d_E_ext` | `semantic_disagreement(... beta['reduction'])` | `reports/chapter3_end_to_end_report.json` |
| Composition of extended objects | `compose(e_ext, e_dec, beta)` | `examples/chapter3_end_to_end.py` |


## Final validation artifacts

- Chapter 2 numerical route (`mu`, `H`, `d_E`, composition, `L`, `I`, `D_ij`) -> `proofs/validate_thesis_examples.py` -> `reports/thesis_validation.json`.
- Chapter 3 numerical route (`P_sit`, class selection, reductions, `Delta`, `d_E_ext`, `D_choice`) -> `proofs/validate_thesis_examples.py` -> `reports/thesis_validation.json`.
- Combined defense demo route -> `examples/thesis_demo.py` -> `reports/thesis_demo_report.json` and `reports/thesis_demo_composition_graph.html`.
