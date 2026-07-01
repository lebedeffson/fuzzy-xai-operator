# Sprint Status

## Git

- branch: main
- commit: be2a300
- tag: none
- pushed: unknown

## Summary

DubnaXAI/FuzzyXAI has three separated layers: framework computes, applications run scenarios, site displays prepared artifacts.

## Changed Areas

- DUBNAXAI_FULL_WORK_REPORT.md
- FuzzyXAI_FINAL_DELIVERY_REPORT.md
- FuzzyXAI_full_delivery_package.zip
- Makefile
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
| source_commit | be2a30096b275d3365a92cb69b165e57e99e90de |

## Operator Traceability

| Scenario | Nodes traced | Edges traced | Formulas | Components | Verifier report | Dashboard v2 |
|---|---:|---:|---|---|---|---|
| external_logistic_regression | 9 | 9 | yes | yes | passed | yes |
| external_gradient_boosting | 9 | 9 | yes | yes | passed | yes |

## Site Separation

- site imports fuzzyxai: no
- site computes operator values: no

## Application Separation

- applications choose action directly: no

## Dirty Working Tree

Dirty source files: yes

```text
M DUBNAXAI_FULL_WORK_REPORT.md
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
 M external_validation/README.md
 M external_validation/outputs/external_wine_blackbox_validation.zip
 M external_validation/outputs/external_wine_blackbox_validation/external_validation_report.md
 M external_validation/outputs/external_wine_blackbox_validation/external_wine_summary.json
 M external_validation/outputs/external_wine_blackbox_validation/gradient_boosting/operator_dashboard.png
 M external_validation/outputs/external_wine_blackbox_validation/gradient_boosting/proof_trace.json
 M external_validation/outputs/external_wine_blackbox_validation/gradient_boosting/route.json
 M external_validation/outputs/external_wine_blackbox_validation/gradient_boosting/summary.json
 M external_validation/outputs/external_wine_blackbox_validation/import_provenance.json
 M external_validation/outputs/external_wine_blackbox_validation/logistic_regression/operator_dashboard.png
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
 M external_validation/run_external_wine_test.py
 M framework/fuzzyxai/fuzzyxai/adapters/tabular_classification.py
 M framework/fuzzyxai/fuzzyxai/core/external_tabular_route.py
 M framework/fuzzyxai/fuzzyxai/core/types.py
 M framework/fuzzyxai/fuzzyxai/viz/__init__.py
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
?? external_validation/outputs/external_wine_blackbox_validation/gradient_boosting/dashboard_data.json
?? external_validation/outputs/external_wine_blackbox_validation/gradient_boosting/operator_cards/
?? external_validation/outputs/external_wine_blackbox_validation/gradient_boosting/operator_dashboard_v2.html
?? external_validation/outputs/external_wine_blackbox_validation/gradient_boosting/operator_dashboard_v2.png
?? external_validation/outputs/external_wine_blackbox_validation/gradient_boosting/operator_table.csv
?? external_validation/outputs/external_wine_blackbox_validation/gradient_boosting/operator_trace.json
?? external_validation/outputs/external_wine_blackbox_validation/gradient_boosting/verifier_report.json
?? external_validation/outputs/external_wine_blackbox_validation/logistic_regression/dashboard_data.json
?? external_validation/outputs/external_wine_blackbox_validation/logistic_regression/operator_cards/
?? external_validation/outputs/external_wine_blackbox_validation/logistic_regression/operator_dashboard_v2.html
?? external_validation/outputs/external_wine_blackbox_validation/logistic_regression/operator_dashboard_v2.png
?? external_validation/outputs/external_wine_blackbox_validation/logistic_regression/operator_table.csv
?? external_validation/outputs/external_wine_blackbox_validation/logistic_regression/operator_trace.json
?? external_validation/outputs/external_wine_blackbox_validation/logistic_regression/verifier_report.json
?? framework/fuzzyxai/fuzzyxai/viz/traceability.py
?? models/
?? reports/dataset_audit/
?? reports/evaluation/
?? reports/framework_audit/
?? reports/practice_demo/
?? reports/real_validation/
?? reports/release/FuzzyXAI_framework_audit_package.zip
?? reports/training/
?? scripts/check_operator_traceability.py
```

Diff summary:

```text
DUBNAXAI_FULL_WORK_REPORT.md                       |  24 +
 Makefile                                           |   7 +-
 .../beacon_xai/figures/operator_dashboard.png      | Bin 215831 -> 214951 bytes
 .../scenarios/beacon_xai/proof/proof_trace.json    | 121 ++++-
 applications/scenarios/beacon_xai/route/route.json | 119 ++++-
 .../beacon_xai/site_payload/scenario.json          |   2 +-
 .../gd_anfis_shap/figures/operator_dashboard.png   | Bin 202172 -> 201290 bytes
 .../scenarios/gd_anfis_shap/proof/proof_trace.json | 121 ++++-
 .../scenarios/gd_anfis_shap/route/route.json       | 119 ++++-
 .../gd_anfis_shap/site_payload/scenario.json       |   2 +-
 .../gis_integro/figures/operator_dashboard.png     | Bin 216496 -> 215701 bytes
 .../scenarios/gis_integro/proof/proof_trace.json   | 139 ++++-
 .../scenarios/gis_integro/route/route.json         | 137 ++++-
 .../gis_integro/site_payload/scenario.json         |   2 +-
 .../hybrid_xiris/figures/operator_dashboard.png    | Bin 261700 -> 260987 bytes
 .../scenarios/hybrid_xiris/proof/proof_trace.json  | 175 +++++-
 .../scenarios/hybrid_xiris/route/route.json        | 173 +++++-
 .../hybrid_xiris/site_payload/scenario.json        |   2 +-
 .../figures/operator_dashboard.png                 | Bin 206245 -> 205412 bytes
 .../medical_ecg_signal/proof/proof_trace.json      | 121 ++++-
 .../scenarios/medical_ecg_signal/route/route.json  | 119 ++++-
 .../medical_ecg_signal/site_payload/scenario.json  |   2 +-
 external_validation/README.md                      |  14 +
 .../outputs/external_wine_blackbox_validation.zip  | Bin 488113 -> 1213657 bytes
 .../external_validation_report.md                  |   2 +-
 .../external_wine_summary.json                     |  26 +-
 .../gradient_boosting/operator_dashboard.png       | Bin 266872 -> 265670 bytes
 .../gradient_boosting/proof_trace.json             | 548 ++++++++++++++++++-
 .../gradient_boosting/route.json                   | 545 ++++++++++++++++++-
 .../gradient_boosting/summary.json                 |  12 +-
 .../import_provenance.json                         |   6 +-
 .../logistic_regression/operator_dashboard.png     | Bin 267904 -> 266740 bytes
 .../logistic_regression/proof_trace.json           | 584 ++++++++++++++++++++-
 .../logistic_regression/route.json                 | 581 +++++++++++++++++++-
 .../logistic_regression/summary.json               |  12 +-
 .../manifest.json                                  | 192 ++++++-
 ...l_wine_gradient_boosting_operator_dashboard.png | Bin 266872 -> 265670 bytes
 ...xternal_wine_gradient_boosting_proof_trace.json | 548 ++++++++++++++++++-
 .../external_wine_gradient_boosting_route.json     | 545 ++++++++++++++++++-
 .../external_wine_gradient_boosting_summary.json   |  12 +-
 ...wine_logistic_regression_operator_dashboard.png | Bin 267904 -> 266740 bytes
 ...ernal_wine_logistic_regression_proof_trace.json | 584 ++++++++++++++++++++-
 .../external_wine_logistic_regression_route.json   | 581 +++++++++++++++++++-
 .../external_wine_logistic_regression_summary.json |  12 +-
 .../outputs/external_wine_summary.json             |  26 +-
 external_validation/run_external_wine_test.py      |  10 +-
 .../fuzzyxai/adapters/tabular_classification.py    |   1 +
 .../fuzzyxai/core/external_tabular_route.py        | 296 ++++++++++-
 framework/fuzzyxai/fuzzyxai/core/types.py          |  46 ++
 framework/fuzzyxai/fuzzyxai/viz/__init__.py        |   2 +
 fuzzyxai_final_audit_package.zip                   | Bin 6099067 -> 6129548 bytes
 reports/audit/artifact_inventory.csv               |  11 +-
 reports/audit/docx_format_report.md                |   2 +-
 .../glava_4_FuzzyXAI_corrected_final.pdf           | Bin 674182 -> 674182 bytes
 .../glava_5_FuzzyXAI_corrected_final.pdf           | Bin 782995 -> 782995 bytes
 reports/audit/docx_render_report.md                |   2 +-
 reports/audit/fuzzyxai_final_audit.json            |  19 +-
 reports/audit/fuzzyxai_final_audit.md              |  16 +-
 reports/audit/stale_terms_report.md                |  31 +-
 reports/figures/beacon_xai_operator_dashboard.png  | Bin 215831 -> 214951 bytes
 .../figures/gd_anfis_shap_operator_dashboard.png   | Bin 202172 -> 201290 bytes
 reports/figures/gis_integro_operator_dashboard.png | Bin 216496 -> 215701 bytes
 .../figures/hybrid_xiris_operator_dashboard.png    | Bin 261700 -> 260987 bytes
 .../medical_ecg_signal_operator_dashboard.png      | Bin 206245 -> 205412 bytes
 reports/framework/hybrid_xiris_proof_trace.json    | 175 +++++-
 reports/framework/hybrid_xiris_route.json          | 173 +++++-
 reports/release/DUBNAXAI_FULL_WORK_REPORT.md       |  24 +
 reports/release/current/SPRINT_STATUS.md           | 193 ++++---
 reports/release/current/artifact_manifest.json     | 134 ++---
 reports/release/current/check_results.json         |   6 +-
 reports/release/current/git_diff_summary.txt       | 148 +++---
 reports/release/current/git_status.txt             |  33 +-
 reports/release/current/release_summary.json       |   8 +-
 reports/routes/beacon_xai_route.json               | 119 ++++-
 reports/routes/gd_anfis_shap_route.json            | 119 ++++-
 reports/routes/gis_integro_route.json              | 137 ++++-
 reports/routes/hybrid_xiris_route.json             | 173 +++++-
 reports/routes/medical_ecg_signal_route.json       | 119 ++++-
 .../studio_batch/hybrid_xiris_proof_package.json   |  30 +-
 .../repository_inventory/repository_excluded.json  |   2 +-
 .../repository_inventory/repository_inventory.json |   6 +-
 .../repository_inventory/repository_inventory.md   |   2 +-
 scripts/build_sprint_report.py                     |  74 +++
 scripts/check_framework_external_usage.py          |  18 +
 .../figures/beacon_xai_operator_dashboard.png      | Bin 215831 -> 214951 bytes
 .../figures/gd_anfis_shap_operator_dashboard.png   | Bin 202172 -> 201290 bytes
 .../figures/gis_integro_operator_dashboard.png     | Bin 216496 -> 215701 bytes
 .../figures/hybrid_xiris_operator_dashboard.png    | Bin 261700 -> 260987 bytes
 .../medical_ecg_signal_operator_dashboard.png      | Bin 206245 -> 205412 bytes
 site/dubnaxai/public/routes/beacon_xai_route.json  | 119 ++++-
 .../public/routes/gd_anfis_shap_route.json         | 119 ++++-
 site/dubnaxai/public/routes/gis_integro_route.json | 137 ++++-
 .../dubnaxai/public/routes/hybrid_xiris_route.json | 173 +++++-
 .../public/routes/medical_ecg_signal_route.json    | 119 ++++-
 tests/test_framework_external_usage.py             |  18 +
 visual_artifacts_latest.zip                        | Bin 2533274 -> 2533446 bytes
 96 files changed, 8443 insertions(+), 586 deletions(-)
```

## Risks and Todos

See `risks_and_todos.md`.

## Next Step

external payload schemas and adapter contracts
