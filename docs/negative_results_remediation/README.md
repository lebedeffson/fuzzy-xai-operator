# FXAI negative-results remediation

This package implements the architecture registered by `FXAI-NEGATIVE-RESULTS-REMEDIATION` without changing the frozen `v1.3.0` results:

- `H3-original = not_supported`
- `H5-P-original = not_supported`
- `H6-general = not_supported`

## Reproduce

Use an existing project Python environment; no additional virtual environment is required.

```bash
make reproduce-negative-results-remediation \
  REMEDIATION_PYTHON=/home/lebedeffson/Code/fuzzy-xai-operator/.venv-confirmatory/bin/python
```

The full profile uses one process with BLAS thread counts fixed to one. It performs three and only three formative iterations, freezes the resulting code lineage, runs the controlled fault library, two exploratory real tabular rule-effect evaluations, and a 500,000-event controlled temporal replay.

## Claim boundary

The current package is a technical research candidate, not independent confirmation:

- H3-R1 through H3-R3 were not evaluated because no new independent sealed dataset was registered before the lock; no test labels were opened.
- H3-R4 is supported only in the preregistered controlled replay and is not a production-stream claim.
- H5-A2 is supported only for the registered controlled/compositional fault library.
- H5-P2 and H5-P3 were not evaluated on independent model-error data.
- H6-R1 through H6-R3 are exploratory measurements; H6-R4 is not confirmatory and H6-R5 was not evaluated.
- The registered H6 detectability target was not met.

The evidence map is `artifacts/negative_results_remediation/evidence_map.json`. The first pre-final controlled run is preserved as invalid evidence because it exposed three implementation defects that were corrected before the final lock.
