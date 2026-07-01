# Sprint Status

## Git

- branch: main
- commit: c15d5a0
- tag: none
- pushed: unknown

## Summary

DubnaXAI/FuzzyXAI has three separated layers: framework computes, applications run scenarios, site displays prepared artifacts.

## Changed Areas

- DUBNAXAI_FULL_WORK_REPORT.md
- FuzzyXAI_FINAL_DELIVERY_REPORT.md
- FuzzyXAI_full_delivery_package.zip
- applications
- data
- external_validation
- framework
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
| models | LogisticRegression, GradientBoostingClassifier |
| action | lower_confidence,lower_confidence |
| diagnostic | D_external_tabular_uncertainty,D_external_tabular_uncertainty |
| route | PASS |
| proof | PASS |
| dashboard | PASS |
| verifier | passed |
| source_commit | c15d5a0476b909fe9e25d27cc077a9d8fa3be3fd |

## Site Separation

- site imports fuzzyxai: no
- site computes operator values: no

## Application Separation

- applications choose action directly: no

## Dirty Working Tree

Dirty source files: yes

```text
M DUBNAXAI_FULL_WORK_REPORT.md
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
 M external_validation/README.md
 M external_validation/outputs/external_wine_blackbox_validation.zip
 M external_validation/outputs/external_wine_gradient_boosting_operator_dashboard.png
 M external_validation/outputs/external_wine_gradient_boosting_proof_trace.json
 M external_validation/outputs/external_wine_gradient_boosting_route.json
 M external_validation/outputs/external_wine_gradient_boosting_summary.json
 M external_validation/outputs/external_wine_logistic_regression_operator_dashboard.png
 M external_validation/outputs/external_wine_logistic_regression_proof_trace.json
 M external_validation/outputs/external_wine_logistic_regression_route.json
 M external_validation/outputs/external_wine_logistic_regression_summary.json
 M external_validation/outputs/external_wine_summary.json
 M external_validation/run_external_wine_test.py
 M framework/fuzzyxai/fuzzyxai/core/external_tabular_route.py
 M fuzzyxai_final_audit_package.zip
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
 M reports/release/DUBNAXAI_FULL_WORK_REPORT.md
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
 M scripts/check_framework_external_usage.py
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
 M tests/test_framework_external_usage.py
 M visual_artifacts_latest.zip
?? FuzzyXAI_FINAL_DELIVERY_REPORT.md
?? FuzzyXAI_full_delivery_package.zip
?? data/real_public/
?? external_validation/outputs/external_wine_blackbox_validation/
?? models/
?? reports/dataset_audit/
?? reports/evaluation/
?? reports/framework_audit/
?? reports/practice_demo/
?? reports/real_validation/
?? reports/release/FuzzyXAI_framework_audit_package.zip
?? reports/training/
```

Diff summary:

```text
DUBNAXAI_FULL_WORK_REPORT.md                       |   9 +-
 .../beacon_xai/figures/operator_dashboard.png      | Bin 215877 -> 215831 bytes
 .../scenarios/beacon_xai/proof/proof_trace.json    |   4 +-
 applications/scenarios/beacon_xai/route/route.json |   2 +-
 .../beacon_xai/site_payload/scenario.json          |   2 +-
 .../gd_anfis_shap/figures/operator_dashboard.png   | Bin 202228 -> 202172 bytes
 .../scenarios/gd_anfis_shap/proof/proof_trace.json |   4 +-
 .../scenarios/gd_anfis_shap/route/route.json       |   2 +-
 .../gd_anfis_shap/site_payload/scenario.json       |   2 +-
 .../gis_integro/figures/operator_dashboard.png     | Bin 216570 -> 216496 bytes
 .../scenarios/gis_integro/proof/proof_trace.json   |   4 +-
 .../scenarios/gis_integro/route/route.json         |   2 +-
 .../gis_integro/site_payload/scenario.json         |   2 +-
 .../hybrid_xiris/figures/operator_dashboard.png    | Bin 261786 -> 261700 bytes
 .../scenarios/hybrid_xiris/proof/proof_trace.json  |   4 +-
 .../scenarios/hybrid_xiris/route/route.json        |   2 +-
 .../hybrid_xiris/site_payload/scenario.json        |   2 +-
 .../figures/operator_dashboard.png                 | Bin 206309 -> 206245 bytes
 .../medical_ecg_signal/proof/proof_trace.json      |   4 +-
 .../scenarios/medical_ecg_signal/route/route.json  |   2 +-
 .../medical_ecg_signal/site_payload/scenario.json  |   2 +-
 external_validation/README.md                      |  23 +--
 .../outputs/external_wine_blackbox_validation.zip  | Bin 490120 -> 488113 bytes
 ...l_wine_gradient_boosting_operator_dashboard.png | Bin 269393 -> 266872 bytes
 ...xternal_wine_gradient_boosting_proof_trace.json |   6 +-
 .../external_wine_gradient_boosting_route.json     |   4 +-
 .../external_wine_gradient_boosting_summary.json   |   8 +-
 ...wine_logistic_regression_operator_dashboard.png | Bin 270822 -> 267904 bytes
 ...ernal_wine_logistic_regression_proof_trace.json |   6 +-
 .../external_wine_logistic_regression_route.json   |   4 +-
 .../external_wine_logistic_regression_summary.json |   8 +-
 .../outputs/external_wine_summary.json             |  18 +--
 external_validation/run_external_wine_test.py      | 157 +++++++++++++++++++--
 .../fuzzyxai/core/external_tabular_route.py        |   2 +-
 fuzzyxai_final_audit_package.zip                   | Bin 6099067 -> 6129548 bytes
 reports/audit/artifact_inventory.csv               |  11 +-
 reports/audit/docx_format_report.md                |   2 +-
 .../glava_4_FuzzyXAI_corrected_final.pdf           | Bin 674182 -> 674182 bytes
 .../glava_5_FuzzyXAI_corrected_final.pdf           | Bin 782995 -> 782995 bytes
 reports/audit/docx_render_report.md                |   2 +-
 reports/audit/fuzzyxai_final_audit.json            |  19 ++-
 reports/audit/fuzzyxai_final_audit.md              |  16 +--
 reports/audit/stale_terms_report.md                |  31 ++--
 reports/figures/beacon_xai_operator_dashboard.png  | Bin 215877 -> 215831 bytes
 .../figures/gd_anfis_shap_operator_dashboard.png   | Bin 202228 -> 202172 bytes
 reports/figures/gis_integro_operator_dashboard.png | Bin 216570 -> 216496 bytes
 .../figures/hybrid_xiris_operator_dashboard.png    | Bin 261786 -> 261700 bytes
 .../medical_ecg_signal_operator_dashboard.png      | Bin 206309 -> 206245 bytes
 reports/framework/hybrid_xiris_proof_trace.json    |   4 +-
 reports/framework/hybrid_xiris_route.json          |   2 +-
 reports/release/DUBNAXAI_FULL_WORK_REPORT.md       |   9 +-
 reports/release/current/SPRINT_STATUS.md           | 125 ++++++----------
 reports/release/current/artifact_manifest.json     |  14 +-
 reports/release/current/check_results.json         |   2 +-
 reports/release/current/git_diff_summary.txt       |  33 +++--
 reports/release/current/git_status.txt             |  85 +++--------
 reports/release/current/release_summary.json       |   4 +-
 reports/routes/beacon_xai_route.json               |   2 +-
 reports/routes/gd_anfis_shap_route.json            |   2 +-
 reports/routes/gis_integro_route.json              |   2 +-
 reports/routes/hybrid_xiris_route.json             |   2 +-
 reports/routes/medical_ecg_signal_route.json       |   2 +-
 .../studio_batch/hybrid_xiris_proof_package.json   |  30 ++--
 .../repository_inventory/repository_excluded.json  |   2 +-
 .../repository_inventory/repository_inventory.json |   6 +-
 .../repository_inventory/repository_inventory.md   |   2 +-
 scripts/build_sprint_report.py                     |  25 +++-
 scripts/check_framework_external_usage.py          |  48 ++++++-
 .../figures/beacon_xai_operator_dashboard.png      | Bin 215877 -> 215831 bytes
 .../figures/gd_anfis_shap_operator_dashboard.png   | Bin 202228 -> 202172 bytes
 .../figures/gis_integro_operator_dashboard.png     | Bin 216570 -> 216496 bytes
 .../figures/hybrid_xiris_operator_dashboard.png    | Bin 261786 -> 261700 bytes
 .../medical_ecg_signal_operator_dashboard.png      | Bin 206309 -> 206245 bytes
 site/dubnaxai/public/routes/beacon_xai_route.json  |   2 +-
 .../public/routes/gd_anfis_shap_route.json         |   2 +-
 site/dubnaxai/public/routes/gis_integro_route.json |   2 +-
 .../dubnaxai/public/routes/hybrid_xiris_route.json |   2 +-
 .../public/routes/medical_ecg_signal_route.json    |   2 +-
 tests/test_framework_external_usage.py             |  16 +++
 visual_artifacts_latest.zip                        | Bin 2533274 -> 2533446 bytes
 80 files changed, 499 insertions(+), 294 deletions(-)
```

## Risks and Todos

See `risks_and_todos.md`.

## Next Step

external payload schemas and adapter contracts
