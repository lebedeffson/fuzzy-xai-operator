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
- Added route visualization layer:
  - `framework/fuzzyxai/fuzzyxai/viz/operator_state.py`
  - `framework/fuzzyxai/fuzzyxai/viz/route_builder.py`
  - `framework/fuzzyxai/fuzzyxai/viz/matplotlib_dashboard.py`
- Added example:
  - `framework/fuzzyxai/examples/show_hybrid_xiris_dashboard.py`
- Generated:
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
- all application scenarios
- HYBRID-XIRIS operator dashboard export
- DubnaXAI static site build

## Key Outputs

- Framework: `framework/fuzzyxai`
- Applications: `applications/scenarios`
- Site: `site/dubnaxai`
- Repository registry: `registry/repositories.yaml`
- Repository report: `reports/validation/repository_inventory/repository_inventory.md`
- Operator route JSON: `reports/routes/hybrid_xiris_route.json`
- Operator dashboard PNG: `reports/figures/hybrid_xiris_operator_dashboard.png`

## Chapter Mapping

- Chapter 4: `framework/fuzzyxai` - FuzzyXAI as Python framework.
- Chapter 5: `site/dubnaxai` - DubnaXAI as site/ecosystem.
- Chapter 6: `applications/scenarios` - applied scenarios and proof traces.

## Notes

- Generated audit/practice artifacts may keep the working tree dirty after release checks.
- The committed source layer is separated from generated reports and archives.
- External public repositories are not blindly vendored; they require license review and pinned commit review before mirroring under `external/research_repos`.
