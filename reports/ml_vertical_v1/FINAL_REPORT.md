# FuzzyXAI ML Vertical v1 final report

## Result

Status: `FUZZYXAI_ML_VERTICAL_V1_IMPLEMENTED`

The implementation connects a real deterministic sklearn LogisticRegression and real SHAP explanation to the existing FuzzyXAI evidence, uncertainty-representation, RouteGraph, diagnostic, repair, recertification, and audience-view layers.

## Acceptance

- Registered scenarios: `10/10 PASS`
- Route executability: `1.0`
- False certification: `0`
- Representation selection accuracy: `1.0`
- Observer action accuracy: `1.0`
- Repair-plan success: `1.0`
- Full recertification success: `1.0`
- Audience consistency: `1.0`
- Artifact integrity: `1.0`
- MLflow runs: `10`, each with `9` required JSON artifacts

S1-S10 cover `F0`, `F_int`, `NAS`, `F_ML` and actions `ACCEPT`, `WARN`, `REQUEST_DATA`, `REVIEW`, and `BLOCK`. S9 executes a registered repair and full recertification; S10 verifies deterministic canonical output.

## Interfaces

- REST: `/predict`, `/explain`, `/diagnose`, `/repair/plan`, `/repair/execute`, `/recertify`, run retrieval and audience views.
- UI: an integrated ML Vertical tab plus a Compose-safe `--ml-vertical-only` mode in `apps/layered_demo.py`.
- Tracking: local or HTTP MLflow backend with parameters, metrics, tags and nine named artifacts.
- Deployment: API, UI, MLflow, and persistent artifact storage through `docker compose up`.

## Boundary

This is a reproducible engineering demonstration, not a new scientific hypothesis or clinical evaluation. It does not establish human-time reduction, user benefit, universal practical utility, or natural-incident repair. Historical Chapter 4 results and statuses remain unchanged. DOCX and PDF were not modified.
