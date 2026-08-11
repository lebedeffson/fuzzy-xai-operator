# FuzzyXAI Release Status

## Current Release

- Package: `fuzzyxai-operator`
- Version: `1.4.0a2`
- Default branch: `main`
- Runtime status: framework available
- Public API: `FuzzyXAI.wrap(...).explain(...)`
- Canonical visualization namespace: `fuzzyxai.visualization`
- License: MIT

## Verified Surfaces

The current release provides:

- capability-based adapters for common Python and machine-learning models;
- evidence and provenance objects with deterministic canonical serialization;
- RouteGraph construction for executable ML/XAI pipelines;
- local and inter-stage contract auditing;
- minimal diagnostic cuts and structured failure localization;
- registered repair planning, guarded execution, rollback, and recertification;
- Python, REST, UI, MLflow, and visualization integration surfaces;
- observation-only adapters for independently implemented external pipelines.

The operator manifest maps 41 registered operators to callables, schemas,
tests, evidence, and visualization policies.

## Validation Boundary

The registered external validation covers four pinned public ML/XAI pipeline
fixtures, 40 controlled cases, and 200 comparison decisions. Within that
registered scope, the full framework mode reports complete fault detection,
stage and contract localization, evidence binding, repair, rollback, and route
recertification without false certification or false blocking.

These results apply only to the registered contracts, controlled mutations,
fixed fixtures, and recorded environments. They do not establish universal
defect detection, prediction correctness, clinical safety, industrial
readiness, or measured human benefit.

Detailed evidence and limitations are stored alongside the corresponding
machine-readable files under `protocol/`, `results/`, and `reports/`.

## Reproducibility

Use the committed source tree as the source of truth. Generated archives are
built with:

```bash
python scripts/build_framework_release.py
```

Missing evidence must remain explicitly missing or insufficient. Historical
negative and blocked outcomes are retained in Git and in their registered
result files; they are not rewritten by a later software release.

## Funding

The research was carried out under the state assignment of the Ministry of
Science and Higher Education of the Russian Federation, theme No.
124112200072-2.
