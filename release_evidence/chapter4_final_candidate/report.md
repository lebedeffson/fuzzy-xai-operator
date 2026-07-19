# FuzzyXAI Chapter 4 final candidate evidence

## Computed evidence

- Measured checkpoint experiment: PASS; selected case `case_real_001`.
- Measured native rule ablation: PASS; rule `tree_leaf_11`.
- Universal-model benchmark: 34 verified configurations
  (24 classification, 10 regression).
- Prediction parity: 1.000.
- Adapter conformance: 1.000.
- Explanation graph validation: 1.000.

## External release gates

- Independent A/B comprehension pilot: `planned_not_run`.
- Independent regulated-domain language review: `pending_external_review`.
- Release gate: `BLOCKED`.

The package proves the current computational contracts and measured benchmark only. It does not claim
demonstrated human comprehensibility, clinical validity, or verified support for optional runtimes recorded as
`not_installed_not_verified`.
