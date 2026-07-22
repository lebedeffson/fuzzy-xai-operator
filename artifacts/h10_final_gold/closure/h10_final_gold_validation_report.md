# H10 final Gold validation report

- Study: `FXAI-H10-FINAL-GOLD`
- Repository HEAD at report generation: `1145c395ced391cce6ea60ba3a8131121d03cfd3`
- Gold cases generated: `4500` across `6` pipelines
- Composite cases: `1800`
- Oracle independence: `PASS`
- Manual adjudication: `PENDING_TWO_REAL_REVIEWERS`
- Power gate: `BLOCKED`
- Protocol-validation primary effect: `0.0` for both endpoints
- Sealed opening count: `0`
- Confirmatory scoring: `NOT_RUN`
- Scientific release: `BLOCKED`

## Blocking reasons

- `two_real_manual_adjudication_files_absent`
- `development_primary_effect_below_registered_margin`

## Development-only observation

After strengthening both baselines to ignore the explicitly derived status
field, Full H10 and the best baseline both reached 1.0 source and repair F1 on
the composite development subset. The expected primary effect is 0.0, below
the registered 0.04 margin. Increasing sample size cannot create a missing
effect. Minimal-cut exact match remains exploratory and secondary.
