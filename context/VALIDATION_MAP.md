# Validation map

Use a focused test for each P19 contract. Do not run full pytest per loop.

- alignment/Gamma/risk: `tests/test_p15_automatic_operator_layer.py`
- membership endpoints: `tests/test_p16_membership_policy.py`
- reports/provenance: `tests/test_p15_full_report.py`, `tests/test_p15_provenance_graph.py`
- public runtime behavior: `tests/test_public_framework_api.py`

At P19 completion run full pytest, mypy, ruff, compileall, manifest validation,
wheel build, and wheel-only smoke outside the source checkout.
