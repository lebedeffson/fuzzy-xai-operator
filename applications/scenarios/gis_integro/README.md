# GIS INTEGRO

Scenario id: `gis_integro`.

Evidence level: `gis_integro` / see `config/` and proof package.

## Contents

- `input/` — scenario input and visual/input artifacts.
- `model_card/` — model or rule card.
- `proof/` — FuzzyXAIProofPackage.
- `tables/` — exported calculation tables.
- `screenshots/` — chapter/demo screenshots.
- `run.py` — local reproducibility check.

## Run

```bash
python applications/scenarios/gis_integro/run.py
```

Expected action: `audit_report`.
Verifier status: `PASS`.

## Claim boundary

Control geolayer, not a production GIS validation.
