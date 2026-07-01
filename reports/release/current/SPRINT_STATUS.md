# Sprint Status

## Git

- branch: main
- commit: 1195585
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
- fuzzyxai_final_audit_package.zip
- models
- reports
- research_validation
- scripts
- site
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
| source_commit | 11955855fb111912239f83500af5349fba895ee5 |

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
 M external_validation/outputs/external_wine_blackbox_validation.zip
 M external_validation/outputs/external_wine_blackbox_validation/external_validation_report.md
 M external_validation/outputs/external_wine_blackbox_validation/external_wine_summary.json
 M external_validation/outputs/external_wine_blackbox_validation/gradient_boosting/dashboard_data.json
 M external_validation/outputs/external_wine_blackbox_validation/gradient_boosting/operator_cards/alignment.md
 M external_validation/outputs/external_wine_blackbox_validation/gradient_boosting/operator_cards/diagnostics.md
 M external_validation/outputs/external_wine_blackbox_validation/gradient_boosting/operator_cards/input_artifact.md
 M external_validation/outputs/external_wine_blackbox_validation/gradient_boosting/operator_cards/proof.md
 M external_validation/outputs/external_wine_blackbox_validation/gradient_boosting/operator_cards/representation.md
 M external_validation/outputs/external_wine_blackbox_validation/gradient_boosting/operator_cards/risk.md
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
 M external_validation/outputs/external_wine_blackbox_validation/logistic_regression/operator_cards/alignment.md
 M external_validation/outputs/external_wine_blackbox_validation/logistic_regression/operator_cards/diagnostics.md
 M external_validation/outputs/external_wine_blackbox_validation/logistic_regression/operator_cards/input_artifact.md
 M external_validation/outputs/external_wine_blackbox_validation/logistic_regression/operator_cards/proof.md
 M external_validation/outputs/external_wine_blackbox_validation/logistic_regression/operator_cards/representation.md
 M external_validation/outputs/external_wine_blackbox_validation/logistic_regression/operator_cards/risk.md
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
 M framework/fuzzyxai/fuzzyxai/adapters/tabular_classification.py
 M framework/fuzzyxai/fuzzyxai/core/external_tabular_route.py
 M framework/fuzzyxai/fuzzyxai/viz/traceability.py
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
?? models/
?? reports/chapter4/chapter4_framework_traceability_insert.md
?? reports/dataset_audit/
?? reports/evaluation/
?? reports/framework_audit/
?? reports/practice_demo/
?? reports/real_validation/
?? reports/release/FuzzyXAI_framework_audit_package.zip
?? reports/training/
?? research_validation/
```

Diff summary:

```text
Makefile                                           |   8 +-
 .../beacon_xai/figures/operator_dashboard.png      | Bin 214951 -> 214634 bytes
 .../scenarios/beacon_xai/proof/proof_trace.json    |   4 +-
 applications/scenarios/beacon_xai/route/route.json |   2 +-
 .../beacon_xai/site_payload/scenario.json          |   2 +-
 .../gd_anfis_shap/figures/operator_dashboard.png   | Bin 201290 -> 200962 bytes
 .../scenarios/gd_anfis_shap/proof/proof_trace.json |   4 +-
 .../scenarios/gd_anfis_shap/route/route.json       |   2 +-
 .../gd_anfis_shap/site_payload/scenario.json       |   2 +-
 .../gis_integro/figures/operator_dashboard.png     | Bin 215701 -> 215246 bytes
 .../scenarios/gis_integro/proof/proof_trace.json   |   4 +-
 .../scenarios/gis_integro/route/route.json         |   2 +-
 .../gis_integro/site_payload/scenario.json         |   2 +-
 .../hybrid_xiris/figures/operator_dashboard.png    | Bin 260987 -> 260458 bytes
 .../scenarios/hybrid_xiris/proof/proof_trace.json  |   4 +-
 .../scenarios/hybrid_xiris/route/route.json        |   2 +-
 .../hybrid_xiris/site_payload/scenario.json        |   2 +-
 .../figures/operator_dashboard.png                 | Bin 205412 -> 204944 bytes
 .../medical_ecg_signal/proof/proof_trace.json      |   4 +-
 .../scenarios/medical_ecg_signal/route/route.json  |   2 +-
 .../medical_ecg_signal/site_payload/scenario.json  |   2 +-
 .../outputs/external_wine_blackbox_validation.zip  | Bin 1213657 -> 1243708 bytes
 .../external_validation_report.md                  |   2 +-
 .../external_wine_summary.json                     |  28 ++-
 .../gradient_boosting/dashboard_data.json          | 168 +++++++++-----
 .../gradient_boosting/operator_cards/alignment.md  |   8 +-
 .../operator_cards/diagnostics.md                  |   6 +-
 .../operator_cards/input_artifact.md               |   6 +-
 .../gradient_boosting/operator_cards/proof.md      |   6 +-
 .../operator_cards/representation.md               |  15 +-
 .../gradient_boosting/operator_cards/risk.md       |  10 +-
 .../gradient_boosting/operator_dashboard.png       | Bin 265670 -> 276160 bytes
 .../gradient_boosting/operator_dashboard_v2.html   | 203 +++++++++++------
 .../gradient_boosting/operator_dashboard_v2.png    | Bin 389401 -> 396660 bytes
 .../gradient_boosting/operator_table.csv           |  12 +-
 .../gradient_boosting/operator_trace.json          | 164 ++++++++++----
 .../gradient_boosting/proof_trace.json             | 171 +++++++++++----
 .../gradient_boosting/route.json                   | 158 ++++++++++----
 .../gradient_boosting/summary.json                 |  13 +-
 .../import_provenance.json                         |   6 +-
 .../logistic_regression/dashboard_data.json        | 168 +++++++++-----
 .../operator_cards/alignment.md                    |   8 +-
 .../operator_cards/diagnostics.md                  |   6 +-
 .../operator_cards/input_artifact.md               |   6 +-
 .../logistic_regression/operator_cards/proof.md    |   6 +-
 .../operator_cards/representation.md               |  15 +-
 .../logistic_regression/operator_cards/risk.md     |  10 +-
 .../logistic_regression/operator_dashboard.png     | Bin 266740 -> 277277 bytes
 .../logistic_regression/operator_dashboard_v2.html | 203 +++++++++++------
 .../logistic_regression/operator_dashboard_v2.png  | Bin 394306 -> 396564 bytes
 .../logistic_regression/operator_table.csv         |  12 +-
 .../logistic_regression/operator_trace.json        | 164 ++++++++++----
 .../logistic_regression/proof_trace.json           | 171 +++++++++++----
 .../logistic_regression/route.json                 | 158 ++++++++++----
 .../logistic_regression/summary.json               |  13 +-
 .../manifest.json                                  | 130 +++++------
 ...l_wine_gradient_boosting_operator_dashboard.png | Bin 265670 -> 276160 bytes
 ...xternal_wine_gradient_boosting_proof_trace.json | 171 +++++++++++----
 .../external_wine_gradient_boosting_route.json     | 158 ++++++++++----
 .../external_wine_gradient_boosting_summary.json   |  13 +-
 ...wine_logistic_regression_operator_dashboard.png | Bin 266740 -> 277277 bytes
 ...ernal_wine_logistic_regression_proof_trace.json | 171 +++++++++++----
 .../external_wine_logistic_regression_route.json   | 158 ++++++++++----
 .../external_wine_logistic_regression_summary.json |  13 +-
 .../outputs/external_wine_summary.json             |  28 ++-
 .../fuzzyxai/adapters/tabular_classification.py    |   1 +
 .../fuzzyxai/core/external_tabular_route.py        | 195 +++++++++++------
 framework/fuzzyxai/fuzzyxai/viz/traceability.py    |  25 ++-
 fuzzyxai_final_audit_package.zip                   | Bin 6099067 -> 6129548 bytes
 reports/audit/artifact_inventory.csv               |  11 +-
 reports/audit/docx_format_report.md                |   2 +-
 .../glava_4_FuzzyXAI_corrected_final.pdf           | Bin 674182 -> 674182 bytes
 .../glava_5_FuzzyXAI_corrected_final.pdf           | Bin 782995 -> 782995 bytes
 reports/audit/docx_render_report.md                |   2 +-
 reports/audit/fuzzyxai_final_audit.json            |  19 +-
 reports/audit/fuzzyxai_final_audit.md              |  16 +-
 reports/audit/stale_terms_report.md                |  31 ++-
 reports/figures/beacon_xai_operator_dashboard.png  | Bin 214951 -> 214634 bytes
 .../figures/gd_anfis_shap_operator_dashboard.png   | Bin 201290 -> 200962 bytes
 reports/figures/gis_integro_operator_dashboard.png | Bin 215701 -> 215246 bytes
 .../figures/hybrid_xiris_operator_dashboard.png    | Bin 260987 -> 260458 bytes
 .../medical_ecg_signal_operator_dashboard.png      | Bin 205412 -> 204944 bytes
 reports/framework/hybrid_xiris_proof_trace.json    |   4 +-
 reports/framework/hybrid_xiris_route.json          |   2 +-
 reports/release/current/SPRINT_STATUS.md           | 241 ++++++++++++---------
 reports/release/current/artifact_manifest.json     | 102 ++++-----
 reports/release/current/check_results.json         |   6 +-
 reports/release/current/git_diff_summary.txt       | 169 ++++++++-------
 reports/release/current/git_status.txt             |  51 ++---
 reports/release/current/release_summary.json       |  19 +-
 reports/routes/beacon_xai_route.json               |   2 +-
 reports/routes/gd_anfis_shap_route.json            |   2 +-
 reports/routes/gis_integro_route.json              |   2 +-
 reports/routes/hybrid_xiris_route.json             |   2 +-
 reports/routes/medical_ecg_signal_route.json       |   2 +-
 .../studio_batch/hybrid_xiris_proof_package.json   |  30 ++-
 .../repository_inventory/repository_excluded.json  |   2 +-
 .../repository_inventory/repository_inventory.json |   6 +-
 .../repository_inventory/repository_inventory.md   |   2 +-
 scripts/build_sprint_report.py                     |  64 +++++-
 .../figures/beacon_xai_operator_dashboard.png      | Bin 214951 -> 214634 bytes
 .../figures/gd_anfis_shap_operator_dashboard.png   | Bin 201290 -> 200962 bytes
 .../figures/gis_integro_operator_dashboard.png     | Bin 215701 -> 215246 bytes
 .../figures/hybrid_xiris_operator_dashboard.png    | Bin 260987 -> 260458 bytes
 .../medical_ecg_signal_operator_dashboard.png      | Bin 205412 -> 204944 bytes
 site/dubnaxai/public/routes/beacon_xai_route.json  |   2 +-
 .../public/routes/gd_anfis_shap_route.json         |   2 +-
 site/dubnaxai/public/routes/gis_integro_route.json |   2 +-
 .../dubnaxai/public/routes/hybrid_xiris_route.json |   2 +-
 .../public/routes/medical_ecg_signal_route.json    |   2 +-
 visual_artifacts_latest.zip                        | Bin 2533274 -> 2533446 bytes
 111 files changed, 2615 insertions(+), 1211 deletions(-)
```

## Risks and Todos

See `risks_and_todos.md`.

## Next Step

external payload schemas and adapter contracts
