# FuzzyXAI Research Validation

This layer validates the installable FuzzyXAI framework on controlled external payloads.
It does not import `applications/scenarios` and is not part of the DubnaXAI site.

Run:

```bash
make research-validation PYTHON=.venv/bin/python
make research-validation-check PYTHON=.venv/bin/python
```

Outputs:

- `research_validation/reports/research_validation_results.csv`
- `research_validation/reports/action_distribution.csv`
- `research_validation/reports/diagnostic_distribution.csv`
- `research_validation/reports/representation_class_coverage.csv`
- `research_validation/reports/risk_component_summary.csv`
- `research_validation/reports/fuzzyxai_research_validation_package.zip`

Each experiment also has a full trace package in `research_validation/outputs/<experiment_id>/`.
