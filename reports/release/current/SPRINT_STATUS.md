# Sprint Status

## Git

- branch: main
- commit: ed1fe54
- tag: v0.4-fuzzyxai-framework-all-scenarios
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
- fuzzyxai_final_audit_package.zip
- models
- reports
- scripts
- tests
- visual_artifacts_latest.zip

## Checks

| Check | Result |
|---|---|
| fuzzyxai-framework-check | UNKNOWN |
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

## Site Separation

- site imports fuzzyxai: no
- site computes operator values: no

## Application Separation

- applications choose action directly: no

## Dirty Working Tree

Dirty source files: yes

```text
M  DUBNAXAI_FULL_WORK_REPORT.md
M  Makefile
M  applications/export_operator_routes.py
M  applications/run_framework_scenario.py
M  applications/scenarios/beacon_xai/site_payload/scenario.json
M  applications/scenarios/gd_anfis_shap/site_payload/scenario.json
M  applications/scenarios/gis_integro/site_payload/scenario.json
M  applications/scenarios/hybrid_xiris/site_payload/scenario.json
M  applications/scenarios/medical_ecg_signal/site_payload/scenario.json
 M fuzzyxai_final_audit_package.zip
 M reports/audit/artifact_inventory.csv
 M reports/audit/docx_format_report.md
 M reports/audit/docx_render/glava_4_FuzzyXAI_corrected_final.pdf
 M reports/audit/docx_render/glava_5_FuzzyXAI_corrected_final.pdf
 M reports/audit/docx_render_report.md
 M reports/audit/fuzzyxai_final_audit.json
 M reports/audit/fuzzyxai_final_audit.md
 M reports/audit/stale_terms_report.md
M  reports/release/DUBNAXAI_FULL_WORK_REPORT.md
A  reports/release/current/SPRINT_STATUS.md
A  reports/release/current/artifact_manifest.json
A  reports/release/current/check_results.json
A  reports/release/current/git_diff_summary.txt
A  reports/release/current/git_status.txt
A  reports/release/current/release_summary.json
A  reports/release/current/risks_and_todos.md
A  reports/release/current/scenario_matrix.json
 M reports/studio_batch/hybrid_xiris_proof_package.json
 M reports/validation/repository_inventory/repository_excluded.json
 M reports/validation/repository_inventory/repository_inventory.json
 M reports/validation/repository_inventory/repository_inventory.md
A  scripts/build_sprint_report.py
A  tests/test_sprint_report.py
 M visual_artifacts_latest.zip
?? FuzzyXAI_FINAL_DELIVERY_REPORT.md
?? FuzzyXAI_full_delivery_package.zip
?? data/real_public/
?? models/
?? reports/dataset_audit/
?? reports/evaluation/
?? reports/practice_demo/
?? reports/real_validation/
?? reports/training/
```

Diff summary:

```text
fuzzyxai_final_audit_package.zip                   | Bin 6099067 -> 6129548 bytes
 reports/audit/artifact_inventory.csv               |  11 ++++----
 reports/audit/docx_format_report.md                |   2 +-
 .../glava_4_FuzzyXAI_corrected_final.pdf           | Bin 674182 -> 674182 bytes
 .../glava_5_FuzzyXAI_corrected_final.pdf           | Bin 782995 -> 782995 bytes
 reports/audit/docx_render_report.md                |   2 +-
 reports/audit/fuzzyxai_final_audit.json            |  19 ++++++++++---
 reports/audit/fuzzyxai_final_audit.md              |  16 +++++------
 reports/audit/stale_terms_report.md                |  31 +++++++++++++--------
 .../studio_batch/hybrid_xiris_proof_package.json   |  30 ++++++++++++++------
 .../repository_inventory/repository_excluded.json  |   2 +-
 .../repository_inventory/repository_inventory.json |   6 ++--
 .../repository_inventory/repository_inventory.md   |   2 +-
 visual_artifacts_latest.zip                        | Bin 2533274 -> 2533446 bytes
 14 files changed, 77 insertions(+), 44 deletions(-)
```

## Risks and Todos

See `risks_and_todos.md`.

## Next Step

external payload schemas and adapter contracts
