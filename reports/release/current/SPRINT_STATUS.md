# Sprint Status

## Git

- branch: main
- commit: 3aa10fe
- tag: none
- pushed: unknown

## Summary

DubnaXAI/FuzzyXAI has three separated layers: framework computes, applications run scenarios, site displays prepared artifacts.

## Changed Areas

- AGENTS.md
- CLAUDE.md
- README.md
- context
- examples
- external_validation
- framework
- fuzzyxai_experiments
- pyproject.toml
- reports
- study
- tests

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
- site route copies: 0
- site dashboard copies: 0

## External Framework Validation

| Task | Result |
|---|---|
| import from /tmp | PASS |
| package path | framework/fuzzyxai |
| external task | sklearn_wine_classification |
| models | LogisticRegression, GradientBoostingClassifier |
| action | lower_confidence,lower_confidence |
| diagnostic | D_external_tabular_uncertainty,D_external_tabular_uncertainty |
| route | PASS |
| proof | PASS |
| dashboard | PASS |
| verifier | passed |
| source_commit | 3aa10fe778108a32f87829feacb56bb2f9c2e13a |

## Operator Traceability

| Scenario | Nodes traced | Edges traced | Formulas | Components | Verifier report | Dashboard v2 |
|---|---:|---:|---|---|---|---|
| external_logistic_regression | 9 | 9 | yes | yes | passed | yes |
| external_gradient_boosting | 9 | 9 | yes | yes | passed | yes |

## Research Validation

| Metric | Value |
|---|---:|
| status | PASS |
| experiments_total | 20 |
| task_types | 4 |
| model_families | 12 |
| actions_covered | 4 |
| diagnostics_covered | 12 |
| representation_classes_covered | 4 |
| verifier_passed | 20 |
| traceability_passed | 20 |
| research_analysis_status | PASS |
| research_analysis_files | 3 / 3 |

## Site Separation

- site imports fuzzyxai: no
- site computes operator values: no

## Application Separation

- applications choose action directly: no

## Dirty Working Tree

Dirty source files: yes

```text
M AGENTS.md
 M README.md
 M external_validation/outputs/external_wine_blackbox_validation.zip
 M external_validation/outputs/external_wine_blackbox_validation/external_validation_report.md
 M external_validation/outputs/external_wine_blackbox_validation/external_wine_summary.json
 M external_validation/outputs/external_wine_blackbox_validation/gradient_boosting/dashboard_data.json
 M external_validation/outputs/external_wine_blackbox_validation/gradient_boosting/operator_cards/proof.md
 M external_validation/outputs/external_wine_blackbox_validation/gradient_boosting/operator_dashboard.png
 M external_validation/outputs/external_wine_blackbox_validation/gradient_boosting/operator_dashboard_v2.html
 M external_validation/outputs/external_wine_blackbox_validation/gradient_boosting/operator_dashboard_v2.png
 M external_validation/outputs/external_wine_blackbox_validation/gradient_boosting/operator_table.csv
 M external_validation/outputs/external_wine_blackbox_validation/gradient_boosting/operator_trace.json
 M external_validation/outputs/external_wine_blackbox_validation/gradient_boosting/proof_trace.json
 M external_validation/outputs/external_wine_blackbox_validation/gradient_boosting/route.json
 M external_validation/outputs/external_wine_blackbox_validation/gradient_boosting/summary.json
 M external_validation/outputs/external_wine_blackbox_validation/import_provenance.json
 M external_validation/outputs/external_wine_blackbox_validation/logistic_regression/dashboard_data.json
 M external_validation/outputs/external_wine_blackbox_validation/logistic_regression/operator_cards/proof.md
 M external_validation/outputs/external_wine_blackbox_validation/logistic_regression/operator_dashboard.png
 M external_validation/outputs/external_wine_blackbox_validation/logistic_regression/operator_dashboard_v2.html
 M external_validation/outputs/external_wine_blackbox_validation/logistic_regression/operator_dashboard_v2.png
 M external_validation/outputs/external_wine_blackbox_validation/logistic_regression/operator_table.csv
 M external_validation/outputs/external_wine_blackbox_validation/logistic_regression/operator_trace.json
 M external_validation/outputs/external_wine_blackbox_validation/logistic_regression/proof_trace.json
 M external_validation/outputs/external_wine_blackbox_validation/logistic_regression/route.json
 M external_validation/outputs/external_wine_blackbox_validation/logistic_regression/summary.json
 M external_validation/outputs/external_wine_blackbox_validation/manifest.json
 M external_validation/outputs/external_wine_gradient_boosting_operator_dashboard.png
 M external_validation/outputs/external_wine_gradient_boosting_proof_trace.json
 M external_validation/outputs/external_wine_gradient_boosting_route.json
 M external_validation/outputs/external_wine_gradient_boosting_summary.json
 M external_validation/outputs/external_wine_logistic_regression_operator_dashboard.png
 M external_validation/outputs/external_wine_logistic_regression_proof_trace.json
 M external_validation/outputs/external_wine_logistic_regression_route.json
 M external_validation/outputs/external_wine_logistic_regression_summary.json
 M external_validation/outputs/external_wine_summary.json
 M framework/fuzzyxai/fuzzyxai/adapters/model.py
 M framework/fuzzyxai/fuzzyxai/adapters/sklearn_v2.py
 M framework/fuzzyxai/fuzzyxai/evidence/__init__.py
 M framework/fuzzyxai/fuzzyxai/evidence/claims.py
 M framework/fuzzyxai/fuzzyxai/evidence/contracts.py
 M framework/fuzzyxai/fuzzyxai/evidence/graph.py
 M framework/fuzzyxai/fuzzyxai/evidence/human.py
 M framework/fuzzyxai/fuzzyxai/evidence/similarity.py
 M framework/fuzzyxai/fuzzyxai/evidence/validation.py
 M framework/fuzzyxai/fuzzyxai/runtime.py
 M framework/fuzzyxai/fuzzyxai/visualization/matplotlib_renderer.py
 M framework/fuzzyxai/fuzzyxai/visualization/plotly_renderer.py
 M framework/fuzzyxai/fuzzyxai/visualization/spec.py
 M framework/fuzzyxai/fuzzyxai/visualization/view_model.py
 M fuzzyxai_experiments/checksums.sha256
 M fuzzyxai_experiments/manifest_sha256.json
 M fuzzyxai_experiments/reports/ch2_bc_results.json
 M fuzzyxai_experiments/reports/ch2_critical_ruptures.json
 M fuzzyxai_experiments/reports/ch2_synthesis.json
 M fuzzyxai_experiments/reports/ch3_diagnostic_stand.json
 M fuzzyxai_experiments/reports/ch3_reduction.json
 M fuzzyxai_experiments/reports/ch3_selection.json
 M fuzzyxai_experiments/reports/ch4_integration.json
 M fuzzyxai_experiments/reports/ch5_beacon.json
 M fuzzyxai_experiments/reports/ch5_gd_anfis_shap.json
 M fuzzyxai_experiments/reports/ch5_gis.json
 M fuzzyxai_experiments/reports/ch5_hybrid.json
 M fuzzyxai_experiments/reports/ch5_scenario_runs.json
 M fuzzyxai_experiments/reports/chapter4/ch4_integration.json
 M fuzzyxai_experiments/reports/chapter5/beacon_xai_summary.json
 M fuzzyxai_experiments/reports/chapter5/ch5_beacon.json
 M fuzzyxai_experiments/reports/chapter5/ch5_gd_anfis_shap.json
 M fuzzyxai_experiments/reports/chapter5/ch5_gis.json
 M fuzzyxai_experiments/reports/chapter5/ch5_hybrid.json
 M fuzzyxai_experiments/reports/chapter5/ch5_scenario_runs.json
 M fuzzyxai_experiments/reports/chapter5/hybrid_xiris_summary.json
 M fuzzyxai_experiments/reports/gui_screenshots/01_home_dashboard.png
 M fuzzyxai_experiments/reports/gui_screenshots/02_hybrid_xiris_route.png
 M fuzzyxai_experiments/reports/gui_screenshots/03_hybrid_xiris_result.png
 M fuzzyxai_experiments/reports/gui_screenshots/04_beacon_audit_route.png
 M fuzzyxai_experiments/reports/gui_screenshots/05_beacon_audit_result.png
 M fuzzyxai_experiments/reports/gui_screenshots/06_gis_integro_route_report.png
 M fuzzyxai_experiments/reports/gui_screenshots/07_gd_anfis_shap_route_report.png
 M fuzzyxai_experiments/reports/gui_screenshots/08_evidence_center.png
 M fuzzyxai_experiments/reports/gui_screenshots/09_developer_details.png
 M fuzzyxai_experiments/tables/beacon_xai_summary.json
 M fuzzyxai_experiments/tables/ch4_integration.json
 M fuzzyxai_experiments/tables/ch5_beacon.json
 M fuzzyxai_experiments/tables/ch5_gd_anfis_shap.json
 M fuzzyxai_experiments/tables/ch5_gis.json
 M fuzzyxai_experiments/tables/ch5_hybrid.json
 M fuzzyxai_experiments/tables/ch5_scenario_runs.json
 M fuzzyxai_experiments/tables/hybrid_xiris_summary.json
 M pyproject.toml
 M reports/full_demo/01_memberships.html
 M reports/full_demo/02_feature_contributions.html
 M reports/full_demo/03_representation.html
 M reports/full_demo/04_composition_graph.html
 M reports/release/current/SPRINT_STATUS.md
 M reports/release/current/artifact_manifest.json
 M reports/release/current/check_results.json
 M reports/release/current/git_diff_summary.txt
 M reports/release/current/git_status.txt
 M reports/release/current/release_summary.json
 M reports/release/current/risks_and_todos.md
 M reports/release/current/scenario_matrix.json
 M study/strong_confirmatory/manifest.json
?? CLAUDE.md
?? context/
?? examples/01_tabular_sklearn.py
?? examples/02_tabular_similarity.py
?? examples/03_text_explanation.py
?? examples/04_image_explanation.py
?? examples/05_strict_verbalizer.py
?? examples/06_custom_model_adapter.py
?? examples/07_rule_based_model.py
?? examples/text_explanation_with_verbalizer.py
?? framework/fuzzyxai/configs/
?? framework/fuzzyxai/evidence/
?? framework/fuzzyxai/figures/
?? framework/fuzzyxai/fuzzyxai/evidence/fuzzy_rules.py
?? framework/fuzzyxai/fuzzyxai/evidence/image_representation.py
?? framework/fuzzyxai/fuzzyxai/evidence/text_highlighting.py
?? framework/fuzzyxai/fuzzyxai/verbalization/
?? framework/fuzzyxai/fuzzyxai/visualization/text_highlight.py
?? framework/fuzzyxai/reports/
?? tests/test_atomic_claim_provenance.py
?? tests/test_contribution_sign_semantics.py
?? tests/test_cross_model_semantic_validation.py
?? tests/test_explanation_output_layer_smoke.py
?? tests/test_export_detail_levels.py
?? tests/test_fuzzy_rule_evidence.py
?? tests/test_human_explanation_quality.py
?? tests/test_image_object_representation.py
?? tests/test_integration_consistency_p14.py
?? tests/test_object_representation_export.py
?? tests/test_ollama_backend_transport.py
?? tests/test_realistic_image_case.py
?? tests/test_similarity_reference_corpus.py
?? tests/test_slm_verbalizer.py
?? tests/test_text_highlight_evidence.py
?? tests/test_text_highlight_rendering.py
?? tests/test_verbalization_doctor_cli.py
```

Diff summary:

```text
AGENTS.md                                          |   1 +
 README.md                                          | 158 +++++++-
 .../outputs/external_wine_blackbox_validation.zip  | Bin 1244531 -> 1346808 bytes
 .../external_validation_report.md                  |   2 +-
 .../external_wine_summary.json                     |   6 +-
 .../gradient_boosting/dashboard_data.json          |  28 +-
 .../gradient_boosting/operator_cards/proof.md      |   4 +-
 .../gradient_boosting/operator_dashboard.png       | Bin 276160 -> 290128 bytes
 .../gradient_boosting/operator_dashboard_v2.html   |  32 +-
 .../gradient_boosting/operator_dashboard_v2.png    | Bin 396199 -> 437347 bytes
 .../gradient_boosting/operator_table.csv           |   2 +-
 .../gradient_boosting/operator_trace.json          |  26 +-
 .../gradient_boosting/proof_trace.json             |  28 +-
 .../gradient_boosting/route.json                   |  26 +-
 .../gradient_boosting/summary.json                 |   2 +-
 .../import_provenance.json                         |   6 +-
 .../logistic_regression/dashboard_data.json        |  28 +-
 .../logistic_regression/operator_cards/proof.md    |   4 +-
 .../logistic_regression/operator_dashboard.png     | Bin 277277 -> 291661 bytes
 .../logistic_regression/operator_dashboard_v2.html |  32 +-
 .../logistic_regression/operator_dashboard_v2.png  | Bin 397655 -> 438641 bytes
 .../logistic_regression/operator_table.csv         |   2 +-
 .../logistic_regression/operator_trace.json        |  26 +-
 .../logistic_regression/proof_trace.json           |  28 +-
 .../logistic_regression/route.json                 |  26 +-
 .../logistic_regression/summary.json               |   2 +-
 .../manifest.json                                  |  56 +--
 ...l_wine_gradient_boosting_operator_dashboard.png | Bin 276160 -> 290128 bytes
 ...xternal_wine_gradient_boosting_proof_trace.json |  28 +-
 .../external_wine_gradient_boosting_route.json     |  26 +-
 .../external_wine_gradient_boosting_summary.json   |   2 +-
 ...wine_logistic_regression_operator_dashboard.png | Bin 277277 -> 291661 bytes
 ...ernal_wine_logistic_regression_proof_trace.json |  28 +-
 .../external_wine_logistic_regression_route.json   |  26 +-
 .../external_wine_logistic_regression_summary.json |   2 +-
 .../outputs/external_wine_summary.json             |   6 +-
 framework/fuzzyxai/fuzzyxai/adapters/model.py      |  11 +-
 framework/fuzzyxai/fuzzyxai/adapters/sklearn_v2.py |  26 +-
 framework/fuzzyxai/fuzzyxai/evidence/__init__.py   | 107 +++---
 framework/fuzzyxai/fuzzyxai/evidence/claims.py     |  80 +++-
 framework/fuzzyxai/fuzzyxai/evidence/contracts.py  | 201 +++++++++-
 framework/fuzzyxai/fuzzyxai/evidence/graph.py      |  25 +-
 framework/fuzzyxai/fuzzyxai/evidence/human.py      | 254 ++++++++++--
 framework/fuzzyxai/fuzzyxai/evidence/similarity.py |  30 +-
 framework/fuzzyxai/fuzzyxai/evidence/validation.py |  23 +-
 framework/fuzzyxai/fuzzyxai/runtime.py             | 426 ++++++++++++++++++---
 .../fuzzyxai/visualization/matplotlib_renderer.py  | 156 +++++++-
 .../fuzzyxai/visualization/plotly_renderer.py      | 126 +++++-
 framework/fuzzyxai/fuzzyxai/visualization/spec.py  | 254 +++++++++++-
 .../fuzzyxai/fuzzyxai/visualization/view_model.py  |  65 +++-
 fuzzyxai_experiments/checksums.sha256              |  34 +-
 fuzzyxai_experiments/manifest_sha256.json          |  54 +--
 fuzzyxai_experiments/reports/ch2_bc_results.json   |   6 +-
 .../reports/ch2_critical_ruptures.json             |   6 +-
 fuzzyxai_experiments/reports/ch2_synthesis.json    |   6 +-
 .../reports/ch3_diagnostic_stand.json              |   6 +-
 fuzzyxai_experiments/reports/ch3_reduction.json    |   6 +-
 fuzzyxai_experiments/reports/ch3_selection.json    |  16 +-
 fuzzyxai_experiments/reports/ch4_integration.json  |   4 +-
 fuzzyxai_experiments/reports/ch5_beacon.json       |   6 +-
 .../reports/ch5_gd_anfis_shap.json                 |   4 +-
 fuzzyxai_experiments/reports/ch5_gis.json          |   4 +-
 fuzzyxai_experiments/reports/ch5_hybrid.json       |   6 +-
 .../reports/ch5_scenario_runs.json                 |   4 +-
 .../reports/chapter4/ch4_integration.json          |   4 +-
 .../reports/chapter5/beacon_xai_summary.json       |   2 +-
 .../reports/chapter5/ch5_beacon.json               |   6 +-
 .../reports/chapter5/ch5_gd_anfis_shap.json        |   4 +-
 fuzzyxai_experiments/reports/chapter5/ch5_gis.json |   4 +-
 .../reports/chapter5/ch5_hybrid.json               |   6 +-
 .../reports/chapter5/ch5_scenario_runs.json        |   4 +-
 .../reports/chapter5/hybrid_xiris_summary.json     |   2 +-
 .../reports/gui_screenshots/01_home_dashboard.png  | Bin 105657 -> 101411 bytes
 .../gui_screenshots/02_hybrid_xiris_route.png      | Bin 124220 -> 122891 bytes
 .../gui_screenshots/03_hybrid_xiris_result.png     | Bin 108527 -> 101685 bytes
 .../gui_screenshots/04_beacon_audit_route.png      | Bin 119582 -> 114238 bytes
 .../gui_screenshots/05_beacon_audit_result.png     | Bin 80605 -> 73177 bytes
 .../06_gis_integro_route_report.png                | Bin 124686 -> 118026 bytes
 .../07_gd_anfis_shap_route_report.png              | Bin 120150 -> 116186 bytes
 .../reports/gui_screenshots/08_evidence_center.png | Bin 75605 -> 70762 bytes
 .../gui_screenshots/09_developer_details.png       | Bin 101326 -> 93181 bytes
 .../tables/beacon_xai_summary.json                 |   2 +-
 fuzzyxai_experiments/tables/ch4_integration.json   |   4 +-
 fuzzyxai_experiments/tables/ch5_beacon.json        |   6 +-
 fuzzyxai_experiments/tables/ch5_gd_anfis_shap.json |   4 +-
 fuzzyxai_experiments/tables/ch5_gis.json           |   4 +-
 fuzzyxai_experiments/tables/ch5_hybrid.json        |   6 +-
 fuzzyxai_experiments/tables/ch5_scenario_runs.json |   4 +-
 .../tables/hybrid_xiris_summary.json               |   2 +-
 pyproject.toml                                     |  58 ++-
 reports/full_demo/01_memberships.html              |   2 +-
 reports/full_demo/02_feature_contributions.html    |   2 +-
 reports/full_demo/03_representation.html           |   2 +-
 reports/full_demo/04_composition_graph.html        |   2 +-
 reports/release/current/SPRINT_STATUS.md           | 422 ++++++++++----------
 reports/release/current/artifact_manifest.json     | 142 +++----
 reports/release/current/check_results.json         |   2 +-
 reports/release/current/git_diff_summary.txt       | 211 +++++-----
 reports/release/current/git_status.txt             | 181 +++++----
 reports/release/current/release_summary.json       |  14 +-
 reports/release/current/risks_and_todos.md         |   3 +-
 reports/release/current/scenario_matrix.json       |  20 +-
 study/strong_confirmatory/manifest.json            |   2 +-
 103 files changed, 2687 insertions(+), 1029 deletions(-)
```

## Risks and Todos

See `risks_and_todos.md`.

## Next Step

external payload schemas and adapter contracts
