# Sprint Status

## Git

- branch: main
- commit: 00caff9
- tag: none
- pushed: unknown

## Summary

DubnaXAI/FuzzyXAI has three separated layers: framework computes, applications run scenarios, site displays prepared artifacts.

## Changed Areas

- FuzzyXAI_FINAL_DELIVERY_REPORT.md
- FuzzyXAI_full_delivery_package.zip
- Makefile
- applications
- data
- external_validation
- framework
- fuzzyxai
- fuzzyxai_final_audit_package.zip
- models
- reports
- scripts
- site
- tests
- visual_artifacts_latest.zip

## Checks

| Check | Result |
|---|---|
| fuzzyxai-framework-check | UNKNOWN |
| framework-external-check | PASS |
| operator-route-check | PASS |
| dubnaxai-release-check | UNKNOWN |
| sprint-report | PASS |

## Scenario Matrix

| Scenario | Action | Diagnostic | Route | Proof | Dashboard | Site payload | Verifier | Status |
|---|---|---|---|---|---|---|---|---|
| hybrid_xiris | block | D_quality_source_conflict | yes | yes | yes | yes | passed | PASS |
| medical_ecg_signal | defer_to_human | D_signal_quality | yes | yes | yes | yes | passed | PASS |
| gd_anfis_shap | audit | D_rule_attribution_conflict | yes | yes | yes | yes | passed | PASS |
| beacon_xai | audit | D_counterevidence_conflict | yes | yes | yes | yes | passed | PASS |
| gis_integro | audit_report | D_route_context_limit | yes | yes | yes | yes | passed | PASS |

## Key Values

| Scenario | Key | Expected | Actual | Result |
|---|---|---:|---:|---|
| hybrid_xiris | gamma | 0.351 | 0.351 | PASS |
| hybrid_xiris | delta | 0.106811 | 0.106811 | PASS |
| hybrid_xiris | r_delta | 0.3225 | 0.3225 | PASS |
| hybrid_xiris | rho | 0.8 | 0.8 | PASS |
| hybrid_xiris | chi_crit | 1 | 1 | PASS |
| gd_anfis_shap | alpha_rule | 0.82 | 0.82 | PASS |
| gd_anfis_shap | gamma_rule_shap | 0.685 | 0.685 | PASS |
| gis_integro | p | 0.67 | 0.67 | PASS |
| gis_integro | alpha_mean | 0.72 | 0.72 | PASS |
| gis_integro | s | 0.47 | 0.47 | PASS |
| gis_integro | gamma_route | 0.2 | 0.2 | PASS |

## Artifact Counts

- routes: 5
- proofs: 5
- dashboards: 5
- site payloads: 5
- site route copies: 5
- site dashboard copies: 5

## External Framework Validation

| Task | Result |
|---|---|
| import from /tmp | PASS |
| package path | framework/fuzzyxai |
| external task | sklearn_wine_classification |
| action | accept |
| diagnostic | D_external_tabular_ok |
| route | PASS |
| proof | PASS |
| dashboard | PASS |
| verifier | passed |
| source_commit | 00caff9012de7ad74f81ef367a61d012734b40c2 |

## Site Separation

- site imports fuzzyxai: no
- site computes operator values: no

## Application Separation

- applications choose action directly: no

## Dirty Working Tree

Dirty source files: yes

```text
M Makefile
 M applications/scenarios/beacon_xai/figures/operator_dashboard.png
 M applications/scenarios/beacon_xai/proof/proof_trace.json
 M applications/scenarios/beacon_xai/route/route.json
 M applications/scenarios/beacon_xai/site_payload/scenario.json
 M applications/scenarios/gd_anfis_shap/figures/operator_dashboard.png
 M applications/scenarios/gd_anfis_shap/proof/proof_trace.json
 M applications/scenarios/gd_anfis_shap/route/route.json
 M applications/scenarios/gd_anfis_shap/site_payload/scenario.json
 M applications/scenarios/gis_integro/figures/operator_dashboard.png
 M applications/scenarios/gis_integro/proof/proof_trace.json
 M applications/scenarios/gis_integro/route/route.json
 M applications/scenarios/gis_integro/site_payload/scenario.json
 M applications/scenarios/hybrid_xiris/figures/operator_dashboard.png
 M applications/scenarios/hybrid_xiris/proof/proof_trace.json
 M applications/scenarios/hybrid_xiris/route/route.json
 M applications/scenarios/hybrid_xiris/site_payload/scenario.json
 M applications/scenarios/medical_ecg_signal/figures/operator_dashboard.png
 M applications/scenarios/medical_ecg_signal/proof/proof_trace.json
 M applications/scenarios/medical_ecg_signal/route/route.json
 M applications/scenarios/medical_ecg_signal/site_payload/scenario.json
 M framework/fuzzyxai/fuzzyxai/adapters/__init__.py
 M framework/fuzzyxai/fuzzyxai/core/route.py
 M framework/fuzzyxai/fuzzyxai/core/scenario_registry.py
 M framework/fuzzyxai/fuzzyxai/core/types.py
 M framework/fuzzyxai/fuzzyxai/proof/trace.py
 M fuzzyxai_final_audit_package.zip
R  fuzzyxai/__init__.py -> legacy/fuzzyxai_old/__init__.py
R  fuzzyxai/adapters/__init__.py -> legacy/fuzzyxai_old/adapters/__init__.py
R  fuzzyxai/adapters/beacon_xai.py -> legacy/fuzzyxai_old/adapters/beacon_xai.py
R  fuzzyxai/adapters/gd_anfis_shap.py -> legacy/fuzzyxai_old/adapters/gd_anfis_shap.py
R  fuzzyxai/adapters/gis_integro.py -> legacy/fuzzyxai_old/adapters/gis_integro.py
R  fuzzyxai/adapters/medical_image_to_explanation.py -> legacy/fuzzyxai_old/adapters/medical_image_to_explanation.py
R  fuzzyxai/adapters/tabular_to_explanation.py -> legacy/fuzzyxai_old/adapters/tabular_to_explanation.py
R  fuzzyxai/adapters/text_to_explanation.py -> legacy/fuzzyxai_old/adapters/text_to_explanation.py
R  fuzzyxai/api.py -> legacy/fuzzyxai_old/api.py
R  fuzzyxai/audit/__init__.py -> legacy/fuzzyxai_old/audit/__init__.py
R  fuzzyxai/audit/build_full_delivery.py -> legacy/fuzzyxai_old/audit/build_full_delivery.py
R  fuzzyxai/audit/build_package.py -> legacy/fuzzyxai_old/audit/build_package.py
R  fuzzyxai/audit/common.py -> legacy/fuzzyxai_old/audit/common.py
R  fuzzyxai/audit/dataset_audit.py -> legacy/fuzzyxai_old/audit/dataset_audit.py
R  fuzzyxai/audit/docx_chapters.py -> legacy/fuzzyxai_old/audit/docx_chapters.py
R  fuzzyxai/audit/docx_format.py -> legacy/fuzzyxai_old/audit/docx_format.py
R  fuzzyxai/audit/docx_render_gate.py -> legacy/fuzzyxai_old/audit/docx_render_gate.py
R  fuzzyxai/audit/final_audit.py -> legacy/fuzzyxai_old/audit/final_audit.py
R  fuzzyxai/audit/final_delivery_report.py -> legacy/fuzzyxai_old/audit/final_delivery_report.py
R  fuzzyxai/audit/formula_references.py -> legacy/fuzzyxai_old/audit/formula_references.py
R  fuzzyxai/audit/fresh_clone_gate.py -> legacy/fuzzyxai_old/audit/fresh_clone_gate.py
R  fuzzyxai/audit/grep_stale_terms.py -> legacy/fuzzyxai_old/audit/grep_stale_terms.py
R  fuzzyxai/audit/inventory.py -> legacy/fuzzyxai_old/audit/inventory.py
R  fuzzyxai/audit/package_self_contained.py -> legacy/fuzzyxai_old/audit/package_self_contained.py
R  fuzzyxai/audit/practice_demo.py -> legacy/fuzzyxai_old/audit/practice_demo.py
R  fuzzyxai/audit/proof_qc.py -> legacy/fuzzyxai_old/audit/proof_qc.py
R  fuzzyxai/audit/real_validation.py -> legacy/fuzzyxai_old/audit/real_validation.py
R  fuzzyxai/audit/screenshot_qc.py -> legacy/fuzzyxai_old/audit/screenshot_qc.py
R  fuzzyxai/audit/studio_server_smoke.py -> legacy/fuzzyxai_old/audit/studio_server_smoke.py
R  fuzzyxai/audit/studio_smoke.py -> legacy/fuzzyxai_old/audit/studio_smoke.py
R  fuzzyxai/audit/training_audit.py -> legacy/fuzzyxai_old/audit/training_audit.py
R  fuzzyxai/calibration/__init__.py -> legacy/fuzzyxai_old/calibration/__init__.py
R  fuzzyxai/calibration/cross_validation.py -> legacy/fuzzyxai_old/calibration/cross_validation.py
R  fuzzyxai/calibration/dataset.py -> legacy/fuzzyxai_old/calibration/dataset.py
R  fuzzyxai/calibration/weight_calibration.py -> legacy/fuzzyxai_old/calibration/weight_calibration.py
R  fuzzyxai/category/__init__.py -> legacy/fuzzyxai_old/category/__init__.py
R  fuzzyxai/category/certified_path.py -> legacy/fuzzyxai_old/category/certified_path.py
R  fuzzyxai/category/context_topos.py -> legacy/fuzzyxai_old/category/context_topos.py
R  fuzzyxai/category/diagnostic_completion.py -> legacy/fuzzyxai_old/category/diagnostic_completion.py
R  fuzzyxai/category/expl_category.py -> legacy/fuzzyxai_old/category/expl_category.py
R  fuzzyxai/category/morphism.py -> legacy/fuzzyxai_old/category/morphism.py
R  fuzzyxai/category/presheaf.py -> legacy/fuzzyxai_old/category/presheaf.py
R  fuzzyxai/category/subpresheaf.py -> legacy/fuzzyxai_old/category/subpresheaf.py
R  fuzzyxai/category/yoneda.py -> legacy/fuzzyxai_old/category/yoneda.py
R  fuzzyxai/config/__init__.py -> legacy/fuzzyxai_old/config/__init__.py
R  fuzzyxai/config/defaults.py -> legacy/fuzzyxai_old/config/defaults.py
R  fuzzyxai/core/__init__.py -> legacy/fuzzyxai_old/core/__init__.py
R  fuzzyxai/core/adapters.py -> legacy/fuzzyxai_old/core/adapters.py
R  fuzzyxai/core/alignment.py -> legacy/fuzzyxai_old/core/alignment.py
R  fuzzyxai/core/alignment_synthesis.py -> legacy/fuzzyxai_old/core/alignment_synthesis.py
R  fuzzyxai/core/composition.py -> legacy/fuzzyxai_old/core/composition.py
R  fuzzyxai/core/diagnostics.py -> legacy/fuzzyxai_old/core/diagnostics.py
R  fuzzyxai/core/explain_plan.py -> legacy/fuzzyxai_old/core/explain_plan.py
R  fuzzyxai/core/explanation_object.py -> legacy/fuzzyxai_old/core/explanation_object.py
R  fuzzyxai/core/plan_builder.py -> legacy/fuzzyxai_old/core/plan_builder.py
R  fuzzyxai/core/proof_package.py -> legacy/fuzzyxai_old/core/proof_package.py
R  fuzzyxai/core/reduction.py -> legacy/fuzzyxai_old/core/reduction.py
R  fuzzyxai/core/risk_observer.py -> legacy/fuzzyxai_old/core/risk_observer.py
R  fuzzyxai/core/scenario_engine.py -> legacy/fuzzyxai_old/core/scenario_engine.py
R  fuzzyxai/core/system_operator.py -> legacy/fuzzyxai_old/core/system_operator.py
R  fuzzyxai/core/trust_evaluator.py -> legacy/fuzzyxai_old/core/trust_evaluator.py
R  fuzzyxai/data/__init__.py -> legacy/fuzzyxai_old/data/__init__.py
R  fuzzyxai/data/breast_cancer_adapter.py -> legacy/fuzzyxai_old/data/breast_cancer_adapter.py
R  fuzzyxai/data/cit_registry.py -> legacy/fuzzyxai_old/data/cit_registry.py
R  fuzzyxai/data/citr_loader.py -> legacy/fuzzyxai_old/data/citr_loader.py
R  fuzzyxai/data/dataset_loader.py -> legacy/fuzzyxai_old/data/dataset_loader.py
R  fuzzyxai/data/dataset_record.py -> legacy/fuzzyxai_old/data/dataset_record.py
R  fuzzyxai/data/loader.py -> legacy/fuzzyxai_old/data/loader.py
R  fuzzyxai/data/loaders.py -> legacy/fuzzyxai_old/data/loaders.py
R  fuzzyxai/data/preprocess.py -> legacy/fuzzyxai_old/data/preprocess.py
R  fuzzyxai/data/profile_inference.py -> legacy/fuzzyxai_old/data/profile_inference.py
R  fuzzyxai/data/registry.py -> legacy/fuzzyxai_old/data/registry.py
R  fuzzyxai/data/rikord_loader.py -> legacy/fuzzyxai_old/data/rikord_loader.py
R  fuzzyxai/data/ruccod_loader.py -> legacy/fuzzyxai_old/data/ruccod_loader.py
R  fuzzyxai/data/validators.py -> legacy/fuzzyxai_old/data/validators.py
R  fuzzyxai/datasets/__init__.py -> legacy/fuzzyxai_old/datasets/__init__.py
R  fuzzyxai/datasets/registry.py -> legacy/fuzzyxai_old/datasets/registry.py
R  fuzzyxai/datasets/registry_mosmed_doctor_analysis.py -> legacy/fuzzyxai_old/datasets/registry_mosmed_doctor_analysis.py
R  fuzzyxai/datasets/registry_programs.py -> legacy/fuzzyxai_old/datasets/registry_programs.py
R  fuzzyxai/datasets/registry_steel_ir.py -> legacy/fuzzyxai_old/datasets/registry_steel_ir.py
R  fuzzyxai/demo/__init__.py -> legacy/fuzzyxai_old/demo/__init__.py
R  fuzzyxai/demo/synthetic.py -> legacy/fuzzyxai_old/demo/synthetic.py
R  fuzzyxai/ecosystem/__init__.py -> legacy/fuzzyxai_old/ecosystem/__init__.py
R  fuzzyxai/ecosystem/registry.json -> legacy/fuzzyxai_old/ecosystem/registry.json
R  fuzzyxai/ecosystem/registry.py -> legacy/fuzzyxai_old/ecosystem/registry.py
R  fuzzyxai/evaluate/__init__.py -> legacy/fuzzyxai_old/evaluate/__init__.py
R  fuzzyxai/evaluate/common.py -> legacy/fuzzyxai_old/evaluate/common.py
R  fuzzyxai/evaluate/eval_beacon.py -> legacy/fuzzyxai_old/evaluate/eval_beacon.py
R  fuzzyxai/evaluate/eval_ecg.py -> legacy/fuzzyxai_old/evaluate/eval_ecg.py
R  fuzzyxai/evaluate/eval_gd_anfis_shap.py -> legacy/fuzzyxai_old/evaluate/eval_gd_anfis_shap.py
R  fuzzyxai/evaluate/eval_gis.py -> legacy/fuzzyxai_old/evaluate/eval_gis.py
R  fuzzyxai/evaluate/eval_iris.py -> legacy/fuzzyxai_old/evaluate/eval_iris.py
R  fuzzyxai/evaluate/evaluate_all.py -> legacy/fuzzyxai_old/evaluate/evaluate_all.py
R  fuzzyxai/experiments/__init__.py -> legacy/fuzzyxai_old/experiments/__init__.py
R  fuzzyxai/experiments/chapter2_calibration.py -> legacy/fuzzyxai_old/experiments/chapter2_calibration.py
R  fuzzyxai/experiments/chapter2_equal_raw_structure.py -> legacy/fuzzyxai_old/experiments/chapter2_equal_raw_structure.py
R  fuzzyxai/experiments/chapter2_sample113.py -> legacy/fuzzyxai_old/experiments/chapter2_sample113.py
R  fuzzyxai/experiments/chapter3_critical_ruptures.py -> legacy/fuzzyxai_old/experiments/chapter3_critical_ruptures.py
R  fuzzyxai/export_tables.py -> legacy/fuzzyxai_old/export_tables.py
R  fuzzyxai/hierarchy/__init__.py -> legacy/fuzzyxai_old/hierarchy/__init__.py
R  fuzzyxai/hierarchy/base.py -> legacy/fuzzyxai_old/hierarchy/base.py
R  fuzzyxai/hierarchy/f0.py -> legacy/fuzzyxai_old/hierarchy/f0.py
R  fuzzyxai/hierarchy/hesitant.py -> legacy/fuzzyxai_old/hierarchy/hesitant.py
R  fuzzyxai/hierarchy/interval.py -> legacy/fuzzyxai_old/hierarchy/interval.py
R  fuzzyxai/hierarchy/meta_reducer.py -> legacy/fuzzyxai_old/hierarchy/meta_reducer.py
R  fuzzyxai/hierarchy/multilevel.py -> legacy/fuzzyxai_old/hierarchy/multilevel.py
R  fuzzyxai/hierarchy/neutrosophic.py -> legacy/fuzzyxai_old/hierarchy/neutrosophic.py
R  fuzzyxai/hierarchy/reductions.py -> legacy/fuzzyxai_old/hierarchy/reductions.py
R  fuzzyxai/hierarchy/source_annotation.py -> legacy/fuzzyxai_old/hierarchy/source_annotation.py
R  fuzzyxai/hott/__init__.py -> legacy/fuzzyxai_old/hott/__init__.py
R  fuzzyxai/hott/drift_path.py -> legacy/fuzzyxai_old/hott/drift_path.py
R  fuzzyxai/hott/path_certificates.py -> legacy/fuzzyxai_old/hott/path_certificates.py
R  fuzzyxai/hott/path_type.py -> legacy/fuzzyxai_old/hott/path_type.py
R  fuzzyxai/hott/rupture_type.py -> legacy/fuzzyxai_old/hott/rupture_type.py
R  fuzzyxai/pipeline.py -> legacy/fuzzyxai_old/pipeline.py
R  fuzzyxai/pipelines/__init__.py -> legacy/fuzzyxai_old/pipelines/__init__.py
R  fuzzyxai/pipelines/dataset_observer_pipeline.py -> legacy/fuzzyxai_old/pipelines/dataset_observer_pipeline.py
R  fuzzyxai/practice/__init__.py -> legacy/fuzzyxai_old/practice/__init__.py
R  fuzzyxai/practice/fixtures.py -> legacy/fuzzyxai_old/practice/fixtures.py
R  fuzzyxai/realdata/__init__.py -> legacy/fuzzyxai_old/realdata/__init__.py
R  fuzzyxai/realdata/fetch_real_artifacts.py -> legacy/fuzzyxai_old/realdata/fetch_real_artifacts.py
R  fuzzyxai/reporting/__init__.py -> legacy/fuzzyxai_old/reporting/__init__.py
R  fuzzyxai/reporting/session_report.py -> legacy/fuzzyxai_old/reporting/session_report.py
R  fuzzyxai/risk/__init__.py -> legacy/fuzzyxai_old/risk/__init__.py
R  fuzzyxai/risk/context_acceptance.py -> legacy/fuzzyxai_old/risk/context_acceptance.py
R  fuzzyxai/risk/decision_policy.py -> legacy/fuzzyxai_old/risk/decision_policy.py
R  fuzzyxai/risk/metrics.py -> legacy/fuzzyxai_old/risk/metrics.py
R  fuzzyxai/risk/observer_pipeline.py -> legacy/fuzzyxai_old/risk/observer_pipeline.py
R  fuzzyxai/risk/proxy_policy.py -> legacy/fuzzyxai_old/risk/proxy_policy.py
R  fuzzyxai/risk/reduction_graph.py -> legacy/fuzzyxai_old/risk/reduction_graph.py
R  fuzzyxai/risk/representation_selection.py -> legacy/fuzzyxai_old/risk/representation_selection.py
R  fuzzyxai/risk/risk_aware_model.py -> legacy/fuzzyxai_old/risk/risk_aware_model.py
R  fuzzyxai/risk/risk_function.py -> legacy/fuzzyxai_old/risk/risk_function.py
R  fuzzyxai/risk/risk_observer.py -> legacy/fuzzyxai_old/risk/risk_observer.py
R  fuzzyxai/risk/risk_observer_config.py -> legacy/fuzzyxai_old/risk/risk_observer_config.py
R  fuzzyxai/risk/uncertainty.py -> legacy/fuzzyxai_old/risk/uncertainty.py
R  fuzzyxai/rules/__init__.py -> legacy/fuzzyxai_old/rules/__init__.py
R  fuzzyxai/rules/lofo_f1.py -> legacy/fuzzyxai_old/rules/lofo_f1.py
R  fuzzyxai/run_scenario.py -> legacy/fuzzyxai_old/run_scenario.py
R  fuzzyxai/sdk/__init__.py -> legacy/fuzzyxai_old/sdk/__init__.py
R  fuzzyxai/sdk/base_adapter.py -> legacy/fuzzyxai_old/sdk/base_adapter.py
R  fuzzyxai/sdk/contracts.py -> legacy/fuzzyxai_old/sdk/contracts.py
R  fuzzyxai/sdk/examples/medical_image_adapter.py -> legacy/fuzzyxai_old/sdk/examples/medical_image_adapter.py
R  fuzzyxai/sdk/examples/simple_tabular_adapter.py -> legacy/fuzzyxai_old/sdk/examples/simple_tabular_adapter.py
R  fuzzyxai/selection/__init__.py -> legacy/fuzzyxai_old/selection/__init__.py
R  fuzzyxai/selection/choice_diagnostic.py -> legacy/fuzzyxai_old/selection/choice_diagnostic.py
R  fuzzyxai/selection/compatibility.py -> legacy/fuzzyxai_old/selection/compatibility.py
R  fuzzyxai/selection/pareto_selector.py -> legacy/fuzzyxai_old/selection/pareto_selector.py
R  fuzzyxai/selection/profile_builder.py -> legacy/fuzzyxai_old/selection/profile_builder.py
R  fuzzyxai/studio/__init__.py -> legacy/fuzzyxai_old/studio/__init__.py
R  fuzzyxai/studio/charts.py -> legacy/fuzzyxai_old/studio/charts.py
R  fuzzyxai/studio/demo_runner.py -> legacy/fuzzyxai_old/studio/demo_runner.py
R  fuzzyxai/studio/explainplan_editor.py -> legacy/fuzzyxai_old/studio/explainplan_editor.py
R  fuzzyxai/studio/operator_scenarios.py -> legacy/fuzzyxai_old/studio/operator_scenarios.py
R  fuzzyxai/studio/operators.py -> legacy/fuzzyxai_old/studio/operators.py
R  fuzzyxai/studio/report_export.py -> legacy/fuzzyxai_old/studio/report_export.py
R  fuzzyxai/studio/scenario_presets.py -> legacy/fuzzyxai_old/studio/scenario_presets.py
R  fuzzyxai/studio/state.py -> legacy/fuzzyxai_old/studio/state.py
R  fuzzyxai/studio/widgets.py -> legacy/fuzzyxai_old/studio/widgets.py
R  fuzzyxai/text/__init__.py -> legacy/fuzzyxai_old/text/__init__.py
R  fuzzyxai/text/explanation_generator.py -> legacy/fuzzyxai_old/text/explanation_generator.py
R  fuzzyxai/train/__init__.py -> legacy/fuzzyxai_old/train/__init__.py
R  fuzzyxai/train/common.py -> legacy/fuzzyxai_old/train/common.py
R  fuzzyxai/train/train_all.py -> legacy/fuzzyxai_old/train/train_all.py
R  fuzzyxai/train/train_beacon_counterevidence.py -> legacy/fuzzyxai_old/train/train_beacon_counterevidence.py
R  fuzzyxai/train/train_ecg_quality.py -> legacy/fuzzyxai_old/train/train_ecg_quality.py
R  fuzzyxai/train/train_gd_anfis_shap.py -> legacy/fuzzyxai_old/train/train_gd_anfis_shap.py
R  fuzzyxai/train/train_gis_route.py -> legacy/fuzzyxai_old/train/train_gis_route.py
R  fuzzyxai/train/train_iris_quality.py -> legacy/fuzzyxai_old/train/train_iris_quality.py
R  fuzzyxai/trust/__init__.py -> legacy/fuzzyxai_old/trust/__init__.py
R  fuzzyxai/trust/trust_evaluator.py -> legacy/fuzzyxai_old/trust/trust_evaluator.py
R  fuzzyxai/visual/__init__.py -> legacy/fuzzyxai_old/visual/__init__.py
R  fuzzyxai/visual/composition_graph.py -> legacy/fuzzyxai_old/visual/composition_graph.py
R  fuzzyxai/visual/interactive_graph.py -> legacy/fuzzyxai_old/visual/interactive_graph.py
R  fuzzyxai/visual/representation_plots.py -> legacy/fuzzyxai_old/visual/representation_plots.py
 M reports/audit/artifact_inventory.csv
 M reports/audit/docx_format_report.md
 M reports/audit/docx_render/glava_4_FuzzyXAI_corrected_final.pdf
 M reports/audit/docx_render/glava_5_FuzzyXAI_corrected_final.pdf
 M reports/audit/docx_render_report.md
 M reports/audit/fuzzyxai_final_audit.json
 M reports/audit/fuzzyxai_final_audit.md
 M reports/audit/stale_terms_report.md
 M reports/figures/beacon_xai_operator_dashboard.png
 M reports/figures/gd_anfis_shap_operator_dashboard.png
 M reports/figures/gis_integro_operator_dashboard.png
 M reports/figures/hybrid_xiris_operator_dashboard.png
 M reports/figures/medical_ecg_signal_operator_dashboard.png
 M reports/framework/hybrid_xiris_proof_trace.json
 M reports/framework/hybrid_xiris_route.json
 M reports/release/current/SPRINT_STATUS.md
 M reports/release/current/artifact_manifest.json
 M reports/release/current/check_results.json
 M reports/release/current/git_diff_summary.txt
 M reports/release/current/git_status.txt
 M reports/release/current/release_summary.json
 M reports/routes/beacon_xai_route.json
 M reports/routes/gd_anfis_shap_route.json
 M reports/routes/gis_integro_route.json
 M reports/routes/hybrid_xiris_route.json
 M reports/routes/medical_ecg_signal_route.json
 M reports/studio_batch/hybrid_xiris_proof_package.json
 M reports/validation/repository_inventory/repository_excluded.json
 M reports/validation/repository_inventory/repository_inventory.json
 M reports/validation/repository_inventory/repository_inventory.md
 M scripts/build_sprint_report.py
 M site/dubnaxai/public/figures/beacon_xai_operator_dashboard.png
 M site/dubnaxai/public/figures/gd_anfis_shap_operator_dashboard.png
 M site/dubnaxai/public/figures/gis_integro_operator_dashboard.png
 M site/dubnaxai/public/figures/hybrid_xiris_operator_dashboard.png
 M site/dubnaxai/public/figures/medical_ecg_signal_operator_dashboard.png
 M site/dubnaxai/public/routes/beacon_xai_route.json
 M site/dubnaxai/public/routes/gd_anfis_shap_route.json
 M site/dubnaxai/public/routes/gis_integro_route.json
 M site/dubnaxai/public/routes/hybrid_xiris_route.json
 M site/dubnaxai/public/routes/medical_ecg_signal_route.json
 M visual_artifacts_latest.zip
?? FuzzyXAI_FINAL_DELIVERY_REPORT.md
?? FuzzyXAI_full_delivery_package.zip
?? data/real_public/
?? external_validation/
?? framework/fuzzyxai/fuzzyxai/adapters/tabular_classification.py
?? framework/fuzzyxai/fuzzyxai/core/external_tabular_route.py
?? framework/fuzzyxai/fuzzyxai/core/git_info.py
?? models/
?? reports/dataset_audit/
?? reports/evaluation/
?? reports/framework_audit/
?? reports/practice_demo/
?? reports/real_validation/
?? reports/release/FuzzyXAI_framework_audit_package.zip
?? reports/training/
?? scripts/check_framework_external_usage.py
?? tests/test_framework_external_usage.py
```

Diff summary:

```text
Makefile                                           |   8 +-
 .../beacon_xai/figures/operator_dashboard.png      | Bin 208932 -> 215536 bytes
 .../scenarios/beacon_xai/proof/proof_trace.json    |   5 +-
 applications/scenarios/beacon_xai/route/route.json |   2 +-
 .../beacon_xai/site_payload/scenario.json          |   2 +-
 .../gd_anfis_shap/figures/operator_dashboard.png   | Bin 194342 -> 201861 bytes
 .../scenarios/gd_anfis_shap/proof/proof_trace.json |   5 +-
 .../scenarios/gd_anfis_shap/route/route.json       |   2 +-
 .../gd_anfis_shap/site_payload/scenario.json       |   2 +-
 .../gis_integro/figures/operator_dashboard.png     | Bin 208780 -> 216222 bytes
 .../scenarios/gis_integro/proof/proof_trace.json   |   5 +-
 .../scenarios/gis_integro/route/route.json         |   2 +-
 .../gis_integro/site_payload/scenario.json         |   2 +-
 .../hybrid_xiris/figures/operator_dashboard.png    | Bin 254683 -> 261345 bytes
 .../scenarios/hybrid_xiris/proof/proof_trace.json  |   5 +-
 .../scenarios/hybrid_xiris/route/route.json        |   2 +-
 .../hybrid_xiris/site_payload/scenario.json        |   2 +-
 .../figures/operator_dashboard.png                 | Bin 198816 -> 205968 bytes
 .../medical_ecg_signal/proof/proof_trace.json      |   5 +-
 .../scenarios/medical_ecg_signal/route/route.json  |   2 +-
 .../medical_ecg_signal/site_payload/scenario.json  |   2 +-
 framework/fuzzyxai/fuzzyxai/adapters/__init__.py   |   3 +-
 framework/fuzzyxai/fuzzyxai/core/route.py          |   2 +
 .../fuzzyxai/fuzzyxai/core/scenario_registry.py    |   4 +
 framework/fuzzyxai/fuzzyxai/core/types.py          |   1 +
 framework/fuzzyxai/fuzzyxai/proof/trace.py         |   1 +
 fuzzyxai_final_audit_package.zip                   | Bin 6099067 -> 6129548 bytes
 reports/audit/artifact_inventory.csv               |  11 +-
 reports/audit/docx_format_report.md                |   2 +-
 .../glava_4_FuzzyXAI_corrected_final.pdf           | Bin 674182 -> 674182 bytes
 .../glava_5_FuzzyXAI_corrected_final.pdf           | Bin 782995 -> 782995 bytes
 reports/audit/docx_render_report.md                |   2 +-
 reports/audit/fuzzyxai_final_audit.json            |  19 +-
 reports/audit/fuzzyxai_final_audit.md              |  16 +-
 reports/audit/stale_terms_report.md                |  31 ++-
 reports/figures/beacon_xai_operator_dashboard.png  | Bin 208932 -> 215536 bytes
 .../figures/gd_anfis_shap_operator_dashboard.png   | Bin 194342 -> 201861 bytes
 reports/figures/gis_integro_operator_dashboard.png | Bin 208780 -> 216222 bytes
 .../figures/hybrid_xiris_operator_dashboard.png    | Bin 254683 -> 261345 bytes
 .../medical_ecg_signal_operator_dashboard.png      | Bin 198816 -> 205968 bytes
 reports/framework/hybrid_xiris_proof_trace.json    |   5 +-
 reports/framework/hybrid_xiris_route.json          |   2 +-
 reports/release/current/SPRINT_STATUS.md           | 276 ++++++++++++++++++---
 reports/release/current/artifact_manifest.json     |  24 +-
 reports/release/current/check_results.json         |   6 +-
 reports/release/current/git_diff_summary.txt       |  32 ++-
 reports/release/current/git_status.txt             | 219 ++++++++++++++--
 reports/release/current/release_summary.json       |  12 +-
 reports/routes/beacon_xai_route.json               |   2 +-
 reports/routes/gd_anfis_shap_route.json            |   2 +-
 reports/routes/gis_integro_route.json              |   2 +-
 reports/routes/hybrid_xiris_route.json             |   2 +-
 reports/routes/medical_ecg_signal_route.json       |   2 +-
 .../studio_batch/hybrid_xiris_proof_package.json   |  30 ++-
 .../repository_inventory/repository_excluded.json  |   2 +-
 .../repository_inventory/repository_inventory.json |   6 +-
 .../repository_inventory/repository_inventory.md   |   2 +-
 scripts/build_sprint_report.py                     |  73 +++++-
 .../figures/beacon_xai_operator_dashboard.png      | Bin 208932 -> 215536 bytes
 .../figures/gd_anfis_shap_operator_dashboard.png   | Bin 194342 -> 201861 bytes
 .../figures/gis_integro_operator_dashboard.png     | Bin 208780 -> 216222 bytes
 .../figures/hybrid_xiris_operator_dashboard.png    | Bin 254683 -> 261345 bytes
 .../medical_ecg_signal_operator_dashboard.png      | Bin 198816 -> 205968 bytes
 site/dubnaxai/public/routes/beacon_xai_route.json  |   2 +-
 .../public/routes/gd_anfis_shap_route.json         |   2 +-
 site/dubnaxai/public/routes/gis_integro_route.json |   2 +-
 .../dubnaxai/public/routes/hybrid_xiris_route.json |   2 +-
 .../public/routes/medical_ecg_signal_route.json    |   2 +-
 visual_artifacts_latest.zip                        | Bin 2533274 -> 2533446 bytes
 69 files changed, 696 insertions(+), 158 deletions(-)
```

## Risks and Todos

See `risks_and_todos.md`.

## Next Step

external payload schemas and adapter contracts
