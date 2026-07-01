# DubnaXAI / FuzzyXAI Work Report

## Status

The repository was reorganized into three independent layers:

- `framework/fuzzyxai` - installable Python framework with core FuzzyXAI operators, proof trace, verification and route visualization.
- `applications/scenarios` - reproducible application scenarios that use the framework and produce proof packages, tables and screenshots.
- `site/dubnaxai` - static DubnaXAI ecosystem site that displays prepared data, repositories, routes and scenario results without importing the framework internals.

The site is intentionally separated from the framework. FuzzyXAI computes; DubnaXAI displays.

## What Was Done

### Repository Restructure

- Created the monorepo layout: `framework/`, `site/`, `applications/`, `data/`, `docs/`, `reports/`.
- Moved the FuzzyXAI Python package into `framework/fuzzyxai`.
- Added application scenario folders for:
  - `hybrid_xiris`
  - `medical_ecg_signal`
  - `gd_anfis_shap`
  - `beacon_xai`
  - `gis_integro`
- Added a static DubnaXAI site under `site/dubnaxai`.

### Framework Layer

- Added public framework API in `framework/fuzzyxai/fuzzyxai/__init__.py`.
- Cleaned the package boundary: the old root-level `./fuzzyxai` package was moved to
  `legacy/fuzzyxai_old`, so `import fuzzyxai` resolves to the installable framework
  package after `pip install -e framework/fuzzyxai`.
- Added v0.3 framework-core API:
  - `build_explainable_object`
  - `build_route`
  - `build_proof_trace`
  - `verify_proof_trace`
  - `render_dashboard`
  - `save_route_json`
- Added framework dataclasses:
  - `AdaptedInput`
  - `ExplainableObject`
  - `OperatorNode`
  - `OperatorRoute`
  - `ProofTrace`
- Added HYBRID-XIRIS adapter and operator modules for alignment, reduction, risk, diagnostics and actions.
- Extended the framework route API to all five scenarios in v0.4:
  - `hybrid_xiris`
  - `medical_ecg_signal`
  - `gd_anfis_shap`
  - `beacon_xai`
  - `gis_integro`
- Added scenario adapters and examples:
  - `signal_quality.py`
  - `rule_attribution.py`
  - `counterevidence.py`
  - `geospatial_route.py`
  - `fuzzyxai.examples.load_example(...)`
- Added scenario registry dispatch in `fuzzyxai.core.scenario_registry`.
- Added generic external tabular support:
  - `fuzzyxai.adapters.tabular_classification.TabularClassificationAdapter`
  - `external_wine_classification` route builder
  - non-empty `source_commit` in `OperatorRoute` and `ProofTrace`
- Added framework hardening layer:
  - `fuzzyxai.runtime.FuzzyXAI`
  - expanded `ExplainPlan` thresholds and policies
  - adapter SDK validation and adapter registry
  - operator registry
  - JSON payload/route/proof/operator trace schemas
  - CLI entrypoint `fuzzyxai`
- Added framework documentation:
  - `framework/fuzzyxai/docs/API.md`
  - `framework/fuzzyxai/docs/CLI.md`
  - `framework/fuzzyxai/docs/ADAPTER_SDK.md`
  - `framework/fuzzyxai/docs/EXPLAIN_PLAN.md`
  - `framework/fuzzyxai/docs/OPERATOR_REGISTRY.md`
- Added route visualization layer:
  - `framework/fuzzyxai/fuzzyxai/viz/operator_state.py`
  - `framework/fuzzyxai/fuzzyxai/viz/route_builder.py`
  - `framework/fuzzyxai/fuzzyxai/viz/matplotlib_dashboard.py`
- Added example:
  - `framework/fuzzyxai/examples/show_hybrid_xiris_dashboard.py`
- Generated:
  - `reports/framework/hybrid_xiris_route.json`
  - `reports/framework/hybrid_xiris_proof_trace.json`
  - `reports/routes/hybrid_xiris_route.json`
  - `reports/figures/hybrid_xiris_operator_dashboard.png`

### Application Layer

Each scenario now has a stable structure:

- `README.md`
- `input/`
- `config/`
- `model_card/`
- `proof/`
- `tables/`
- `screenshots/`
- `run.py`

The scenario runner verifies expected actions and proof statuses:

- HYBRID-XIRIS: `block`
- ECG: `defer_to_human`
- GD-ANFIS/SHAP: `audit`
- BEACON-XAI: `audit`
- GIS INTEGRO: `audit_report`

Application `run.py` files are thin wrappers. They call the FuzzyXAI public API through
`applications/run_framework_scenario.py`; they do not choose diagnostics or actions themselves.

### Operator Route Pipeline

The HYBRID-only operator dashboard was extended to all five application scenarios.
Each scenario now exports the same artifact set:

- `route/route.json`
- `proof/proof_trace.json`
- `figures/operator_dashboard.png`
- `site_payload/scenario.json`

The shared exporter is:

- `applications/export_operator_routes.py`

Generated site-facing artifacts are copied to:

- `site/dubnaxai/public/routes/*_route.json`
- `site/dubnaxai/public/figures/*_operator_dashboard.png`

The site still does not compute operator values. It only displays prepared routes.

Route QA was added as a release gate:

- `applications/check_operator_routes.py`
- `make operator-route-check`

It verifies route/proof consistency, expected actions, diagnostics, dashboard files,
site payload links and the rule that the site does not import or compute FuzzyXAI.
It also checks that application scenario runners stay thin and do not reintroduce local
action-selection logic.

### Sprint Reporting Layer

Added a control reporting layer for sprint and release state:

- `scripts/build_sprint_report.py`
- `make sprint-report`
- `reports/release/current/SPRINT_STATUS.md`
- `reports/release/current/release_summary.json`
- `reports/release/current/check_results.json`
- `reports/release/current/scenario_matrix.json`
- `reports/release/current/artifact_manifest.json`
- `reports/release/current/git_status.txt`
- `reports/release/current/git_diff_summary.txt`
- `reports/release/current/risks_and_todos.md`

The report validates all five scenario artifacts, expected actions, diagnostics,
verifier statuses, key control values, site/framework separation, thin application
runners, external framework validation and current git state. It is called by
`dubnaxai-release-check`.

### External Framework Validation

Added black-box validation for FuzzyXAI as an installed library:

- `external_validation/run_external_wine_test.py`
- `external_validation/outputs/external_wine_blackbox_validation.zip`
- `external_validation/outputs/external_wine_blackbox_validation/manifest.json`
- `external_validation/outputs/external_wine_blackbox_validation/external_validation_report.md`
- `external_validation/outputs/external_wine_blackbox_validation/import_provenance.json`
- `external_validation/outputs/external_wine_blackbox_validation/logistic_regression/route.json`
- `external_validation/outputs/external_wine_blackbox_validation/gradient_boosting/route.json`
- `external_validation/outputs/external_wine_summary.json`
- `scripts/check_framework_external_usage.py`
- `make framework-external-check`

The external task uses `sklearn.datasets.load_wine` with two external model
families: `LogisticRegression` and `GradientBoostingClassifier`. It selects
moderate-confidence objects and passes only top-k feature importances, so
`gamma`, `delta` and `rho` are non-zero and the framework selects
`lower_confidence`. It runs outside the repository root, imports only the
installed `fuzzyxai` package, uses the generic tabular adapter and does not
import `applications/scenarios`.

### Research Validation Suite

Added a framework-level research validation suite under `research_validation/`.
It uses controlled external payloads through the installed FuzzyXAI public API and
does not import `applications/scenarios` or the DubnaXAI site.

Current coverage:

- 20 experiments
- 4 task classes
- 12 model families
- 4 action classes
- 12 diagnostic states
- 4 representation classes

Generated artifacts:

- `research_validation/reports/research_validation_results.csv`
- `research_validation/reports/action_distribution.csv`
- `research_validation/reports/diagnostic_distribution.csv`
- `research_validation/reports/representation_class_coverage.csv`
- `research_validation/reports/risk_component_summary.csv`
- `research_validation/reports/manifest.json` with `manifest_self_hash_policy = excluded`
- `research_validation/reports/fuzzyxai_research_validation_package.zip`
- `docs/chapter_4_framework/research_validation_section.md`
- `docs/chapter_4_framework/research_assets/README.md`

Every experiment exports route, operator trace, operator table, proof trace,
verifier report, dashboard data, dashboard v2 and operator cards.

Additional research analysis layer:

- `research_validation/sensitivity/sensitivity_results.csv`
- `research_validation/sensitivity/rho_surface.png`
- `research_validation/sensitivity/action_transition_heatmap.png`
- `research_validation/sensitivity/gamma_delta_action_map.png`
- `research_validation/ablation/ablation_results.csv`
- `research_validation/ablation/ablation_action_changes.csv`
- `research_validation/ablation/ablation_summary.csv`
- `research_validation/ablation/ablation_changed_actions.png`
- `research_validation/cross_model/cross_model_summary.csv`
- `research_validation/cross_model/cross_model_mean_rho.png`

### Framework Release Candidate Acceptance

Added a release-candidate acceptance audit for FuzzyXAI as an installed
framework:

- `scripts/build_fuzzyxai_framework_rc.py`
- `make fuzzyxai-framework-rc-check`
- `make fuzzyxai-framework-rc-package`
- `reports/release/fuzzyxai_framework_rc/`
- `reports/release/fuzzyxai_framework_rc_package.zip`

The RC audit creates a temporary local git clone, checks out the current source
commit, builds a clean virtual environment, installs `framework/fuzzyxai` in
editable mode and verifies that `import fuzzyxai` resolves to the framework
package. It then runs CLI validation, CLI route/proof/dashboard/package export,
an SDK smoke test through `FuzzyXAI` and `ExplainPlan`, schema/registry checks,
documentation presence checks and research-analysis artifact checks.

The acceptance payload is intentionally external to `applications/scenarios`.
It produces non-zero `gamma`, `delta` and `rho`, and the framework selects
`lower_confidence`.

### Operator Traceability v2

Added framework-level operator traceability for the external validation package:

- expanded `OperatorNode` with input/output values, formula text, components,
  thresholds, status reason, interpretation and next-node links;
- added `OperatorEdge` and route edge export;
- added `framework/fuzzyxai/fuzzyxai/viz/traceability.py`;
- added per-model trace artifacts:
  - `operator_trace.json`
  - `operator_table.csv`
  - `verifier_report.json`
  - `dashboard_data.json`
  - `operator_dashboard_v2.png`
  - `operator_dashboard_v2.html`
  - `operator_cards/*.md`
- added `scripts/check_operator_traceability.py`;
- added `make operator-traceability-check`.

The v2 dashboard and HTML are exported by the framework from route/dashboard
data. They do not compute `gamma`, `delta`, `rho` and do not depend on the
DubnaXAI site.

### Site Layer

The DubnaXAI site reads prepared JSON and images:

- `site/dubnaxai/src/data/models.json`
- `site/dubnaxai/src/data/methods.json`
- `site/dubnaxai/src/data/researchers.json`
- `site/dubnaxai/src/data/publications.json`
- `site/dubnaxai/src/data/repositories.json`
- `site/dubnaxai/public/routes/hybrid_xiris_route.json`

The site does not compute `gamma`, `delta`, `rho`, diagnostics or actions.

### Research Repository Inventory

Added public GitHub repository inventory for:

- `https://github.com/fims9000`
- `https://github.com/lebedeffson`

Important correction: not every public repository is included in the DubnaXAI research layer.

Selected repositories are only public research/application repositories with a clear role. Profile repositories, mobile utilities, assistant skills, web utility projects and duplicates are kept in the audit inventory only.

Generated files:

- `registry/repositories.yaml` - selected repositories only.
- `site/dubnaxai/src/data/repositories.json` - selected repositories only.
- `reports/validation/repository_inventory/repository_inventory.md`
- `reports/validation/repository_inventory/repository_inventory.json`
- `reports/validation/repository_inventory/repository_excluded.json`

Current inventory result:

- Discovered public repositories: `24`
- Selected for DubnaXAI: `16`
- Excluded/catalog-only: `8`

Excluded examples:

- `lebedeffson/MobilePhotoSensor` - mobile utility/project repo.
- `lebedeffson/Yandex_skill` - assistant skill/utility repo.
- `lebedeffson/TMPKWebApp` - web/hackathon utility repo.
- profile repositories and duplicate listings.

## Verification

Command:

```bash
make dubnaxai-release-check PYTHON=.venv/bin/python
```

Result:

```text
dubnaxai-release-check: PASS
```

The check runs:

- research repository inventory
- editable framework install
- public framework import/API smoke
- external framework black-box check
- FuzzyXAI CLI check
- FuzzyXAI schema check
- FuzzyXAI adapter SDK check
- FuzzyXAI framework RC acceptance check/package
- operator traceability check
- research validation is available through `make research-validation` and
  `make research-validation-check`; research analysis is available through
  `make fuzzyxai-research-analysis` and `make fuzzyxai-research-analysis-check`
- HYBRID-XIRIS framework example
- framework-core and all-scenario tests
- all application scenarios
- HYBRID-XIRIS operator dashboard export
- all scenario operator route export
- operator route QA
- DubnaXAI static site build
- sprint report

## Key Outputs

- Framework: `framework/fuzzyxai`
- Applications: `applications/scenarios`
- Site: `site/dubnaxai`
- Repository registry: `registry/repositories.yaml`
- Repository report: `reports/validation/repository_inventory/repository_inventory.md`
- Operator route JSON: `reports/routes/hybrid_xiris_route.json`
- Operator dashboard PNG: `reports/figures/hybrid_xiris_operator_dashboard.png`
- Framework route JSON: `reports/framework/hybrid_xiris_route.json`
- Framework proof trace JSON: `reports/framework/hybrid_xiris_proof_trace.json`
- Framework RC package: `reports/release/fuzzyxai_framework_rc_package.zip`

## Chapter Mapping

- Chapter 4: `framework/fuzzyxai` - FuzzyXAI as Python framework.
- Chapter 5: `site/dubnaxai` - DubnaXAI as site/ecosystem.
- Chapter 6: `applications/scenarios` - applied scenarios and proof traces.

## Notes

- Generated audit/practice artifacts may keep the working tree dirty after release checks.
- The committed source layer is separated from generated reports and archives.
- External public repositories are not blindly vendored; they require license review and pinned commit review before mirroring under `external/research_repos`.
