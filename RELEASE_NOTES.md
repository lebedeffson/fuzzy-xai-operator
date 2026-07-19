# FuzzyXAI v1.3.0rc1 Universal Model Integration Candidate

This candidate adds a capability-based adapter registry, family-specific sklearn evidence, optional boosting/neural
adapters, explanation planning, quality gates, conformance reports, and batch/global/why-not/model-comparison APIs.
The deterministic core benchmark records 34 verified configurations. Optional runtimes are not claimed until their
dedicated public CI jobs pass.

The candidate does not close the `v1.2.0` external human gates. The independent comprehension pilot remains
`planned_not_run` and the regulated-domain dictionary remains `pending_external_review`; therefore no final tag is
created.

Canonical evidence:

- `release_evidence/model_universality/summary.json`;
- `release_evidence/model_universality/support_matrix.csv`;
- `release_evidence/chapter4_final_candidate/report.md`;
- `release_evidence/user_study/comprehension_pilot/scoring_report.json`;
- `release_evidence/domain_language_review/review_record.json`.

---

## FuzzyXAI v1.2.0rc3 Empirical Validation Candidate

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
