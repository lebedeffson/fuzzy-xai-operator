# H10-C7R reproduction boundary

The official scoring has already been opened exactly once. Do not rerun the
scoring target against the disclosed held-out Gold.

Safe verification commands:

```bash
export PYTHONPATH=experiments/h10_c3/src:framework/fuzzyxai:.
export PY=/home/lebedeffson/Code/venv/bin/python

$PY -m pytest -q tests/h10_c7r
$PY -m pytest -q
$PY -m ruff check scripts/ch4_revision tests/h10_c7r
$PY scripts/ch4_revision/claim_lint.py \
  --root . \
  --output reports/h10_c7r/H10_C7R_CLAIM_LINT.json
$PY scripts/ch4_revision/verify_parent_result_immutability.py
```

The runtime evidence, selection locks, sealed Gold, authorization, opening
ledger, per-incident results, and bootstrap output are packaged for audit.
Reproduction of file hashes and report generation does not authorize a second
scientific scoring.
