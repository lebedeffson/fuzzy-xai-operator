# Medical ECG Signal

Scenario id: `medical_ecg_signal`.

Evidence level: `medical_ecg_signal` / see `config/` and proof package.

## Contents

- `input/` — scenario input and visual/input artifacts.
- `model_card/` — model or rule card.
- `proof/` — FuzzyXAIProofPackage.
- `tables/` — exported calculation tables.
- `screenshots/` — chapter/demo screenshots.
- `run.py` — local reproducibility check.

## Run

```bash
python applications/scenarios/medical_ecg_signal/run.py
```

Expected action: `defer_to_human`.
Verifier status: `PASS`.

## Claim boundary

Not a clinical diagnostic validation.
