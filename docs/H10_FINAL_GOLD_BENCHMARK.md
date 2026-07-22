# H10 final Gold benchmark

This cycle evaluates source localization, diagnostic cuts, and repair sets on
transaction-derived route mutations. It does not evaluate prediction quality,
H3, H5-P, H6-general, user comprehension, or production readiness.

## Independence boundary

- Gold source elements come from nodes and edges actually changed by executed
  low-level transactions.
- Gold repair actions are normalized inverse transactions.
- Broken paths come from clean/mutated graph differences.
- All equal-cost optimal cuts are enumerated by `gold_oracle`, which does not
  import the evaluated H10 package.
- Methods receive registered and observed graphs only. They never receive the
  transaction log, source truth, repair truth, or optimal cuts.

## Existing environment

No dedicated virtual environment is required. Select an existing Python:

```bash
make reproduce-h10-gold H10_GOLD_PYTHON=/home/lebedeffson/Code/venv/bin/python
```

Ordinary reproduction runs tests, deterministic generation, development
evaluation, power analysis, machine tables, figures, and the validation report.
It does not open sealed truth.

## Manual gate

Two real reviewers must independently complete the templates described in
`artifacts/h10_final_gold/adjudication/README.md`. The pipeline never fabricates
or prefills their answers. `make h10-gold-freeze` fails until adjudication and
the development power gate both pass.

## Current boundary

The generated benchmark has 4,500 cases across six pipelines, including 1,800
composite cases. After a formative correction that made the independent
baselines ignore the explicitly derived status field, Full H10 and the strongest
baseline both achieved development-only source and repair F1 of 1.0 on the
composite subset. The expected primary effect is therefore 0.0, below the
registered 0.04 margin. Sealed scoring remains unopened and scientific release
is blocked. Minimal-cut results remain secondary exploratory evidence only.
