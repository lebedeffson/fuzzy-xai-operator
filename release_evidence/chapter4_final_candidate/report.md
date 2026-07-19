# FuzzyXAI Chapter 4 final candidate evidence

## Computed evidence

- Measured checkpoint experiment: PASS; selected case `case_real_001`.
- Measured native rule ablation: PASS; rule `tree_leaf_11`.
- Universal-model evidence: 40 verified configurations
  (30 classification, 10 regression).
- Core deterministic matrix: 34 configurations.
- Optional runtime integrations: 6 libraries, each verified on Python 3.11 and 3.12.
- Prediction parity: 1.000.
- Adapter conformance: 1.000.
- Explanation graph validation: 1.000.
- Explanation quality gate: 40 / 40.

## External release gates

- Independent A/B comprehension pilot: `planned_not_run`.
- Independent regulated-domain language review: `pending_external_review`.
- Release gate: `BLOCKED`.

The package proves the current computational contracts and measured benchmark only. It does not claim
demonstrated human comprehensibility, clinical validity, or support beyond the explicitly listed model families.
