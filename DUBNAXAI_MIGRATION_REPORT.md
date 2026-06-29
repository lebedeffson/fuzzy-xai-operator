# DubnaXAI Migration Report

## Status

`framework / site / applications` structure created and checked.

```text
dubnaxai-release-check: PASS
framework install/import: PASS
applications scenario runs: PASS
site build: PASS
```

## New Structure

```text
framework/fuzzyxai/          installable FuzzyXAI package copy
site/dubnaxai/               static DubnaXAI site
applications/scenarios/      five reproducible application scenarios
applications/real_public/    real public iris/ECG artifacts copy
docs/assets/screenshots/     screenshots for chapters
docs/assets/tables/          chapter-ready tables
docs/assets/figures/         figure assets
reports/validation/          QA and validation reports
```

## Framework

Path:

```text
framework/fuzzyxai/
```

Added public API to both current root package and framework copy:

```python
build_explanation
compute_alignment
compute_reduction_loss
observe_risk
diagnose
make_action
build_proof_trace
verify_proof_trace
show_operator_route
```

Check:

```bash
make framework-check
```

## Applications

Created:

```text
applications/scenarios/hybrid_xiris/
applications/scenarios/medical_ecg_signal/
applications/scenarios/gd_anfis_shap/
applications/scenarios/beacon_xai/
applications/scenarios/gis_integro/
```

Each scenario contains:

```text
README.md
input/
config/
model_card/
proof/
tables/
screenshots/
run.py
```

Check:

```bash
python applications/run_all_scenarios.py
```

Expected actions:

```text
hybrid_xiris          block
medical_ecg_signal    defer_to_human
gd_anfis_shap         audit
beacon_xai            audit
gis_integro           audit_report
```

## Site

Path:

```text
site/dubnaxai/
```

Data files:

```text
site/dubnaxai/src/data/models.json
site/dubnaxai/src/data/methods.json
site/dubnaxai/src/data/researchers.json
site/dubnaxai/src/data/publications.json
site/dubnaxai/src/data/demos.json
```

Build:

```bash
python site/dubnaxai/build.py
```

Output:

```text
site/dubnaxai/dist/index.html
```

## Validation

New combined check:

```bash
make dubnaxai-release-check
```

Observed result:

```text
dubnaxai-release-check: PASS
```

## Important Note

The old root layout is still kept as source-compatible legacy during migration. Nothing was deleted. The new structure is a working separated layer and can now be used as the basis for chapters:

```text
Chapter 4 -> framework/fuzzyxai
Chapter 5 -> site/dubnaxai
Chapter 6 -> applications/scenarios
```

