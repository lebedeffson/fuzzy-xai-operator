# ML Pipeline v2 comparative evaluation

| Mode | Detect | Stage | Contract | Component | Action | False cert | Evidence | Repair | Recert |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | 0.222 | 0.222 | 0.167 | 0.222 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| B1 | 0.167 | 0.167 | 0.167 | 0.167 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| B2 | 0.667 | 0.667 | 0.667 | 0.667 | 0.667 | 0.333 | 1.000 | 0.000 | 0.000 |
| B3 | 0.667 | 0.667 | 0.667 | 0.667 | 0.667 | 0.333 | 1.000 | 0.000 | 0.000 |
| A0 | 0.667 | 0.667 | 0.667 | 0.667 | 0.667 | 0.333 | 1.000 | 0.000 | 0.000 |
| A1 | 0.667 | 0.667 | 0.667 | 0.667 | 0.667 | 0.333 | 1.000 | 0.000 | 0.000 |
| A2 | 1.000 | 1.000 | 1.000 | 1.000 | 0.944 | 0.056 | 1.000 | 0.000 | 0.000 |
| A3 | 1.000 | 1.000 | 1.000 | 1.000 | 0.944 | 0.056 | 1.000 | 0.000 | 0.000 |
| A4 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 |

Final status: `FUZZYXAI_ML_PIPELINE_V2_COMPARATIVE_SUPPORTED`. A4 is the full FuzzyXAI mode O. MLflow is treated as run registration rather than a competing diagnostic system.

## Primary comparison

The strong local-check baseline B2 reached `0.6667` for violation detection,
stage localization, contract identification, component localization and action
selection. It falsely certified six scenarios whose local component checks
passed while a registered cross-stage relation was violated.

The full A4/O mode reached `1.0000` on all five diagnostic metrics, produced
zero false certifications, and attached all seven required evidence fields.
Its paired improvement over B2 was `0.3333`; the scenario-paired bootstrap 95%
interval was `[0.1111, 0.5556]` for detection, stage, contract and action.

Among the six preregistered cross-stage cases, B2 already detected the local
violations in S11, S16 and S17. FuzzyXAI added three diagnoses that B2 could
not determine: split overlap (S12), preprocessor fit scope (S13), and the
model-explainer version relation (S2). This exceeds the locked two-case
acceptance threshold without weakening B2.

## Functional comparison

B1 recorded normal run parameters, metrics and registered artifacts but did
not infer a contract diagnosis. This is not treated as an MLflow defect:
MLflow preserves run information, while FuzzyXAI evaluates registered
relations over that information. Diagnostic artifacts produced by FuzzyXAI
were explicitly excluded from B1 input.

The raw McNemar p-value for O versus B2 was `0.03125` for each primary binary
diagnostic metric. After the preregistered Holm correction across all main
tests it was `0.5`; the fixed 18-scenario result is therefore interpreted
descriptively rather than as broad statistical generalization.
