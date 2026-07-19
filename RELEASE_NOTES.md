# FuzzyXAI v1.2.0rc3 Empirical Validation Candidate

## Scope

This candidate moves the framework from controlled explanation stories to measured checkpoint evidence.
It does not complete human validation and is intentionally untagged.

## Reproduction

```bash
pip install -e ".[dev]"
make empirical-validation-check
python -m pytest -q
```

## Canonical Evidence

- `release_evidence/empirical_experiments/breast_cancer_checkpoint/empirical_summary.json`
- `release_evidence/chapter4_empirical_validation/report.md`
- `release_evidence/chapter4_empirical_validation/manifest_sha256.json`
- `release_evidence/user_study/comprehension_pilot/scoring_report.json`

## Claim Boundary

The BCDW run is a measured methodological benchmark, not clinical validation. The controlled object-85 story
is not a training result. User comprehensibility remains unproven until the independent A/B pilot is run.
The domain dictionary cannot produce categorical regulated-domain statements before external review.
