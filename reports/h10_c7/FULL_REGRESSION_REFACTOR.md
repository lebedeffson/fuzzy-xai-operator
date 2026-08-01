# H10-C7 refactor acceptance

## Full regression

```text
PYTHONPATH=experiments/h10_c3/src:framework/fuzzyxai:.
/home/lebedeffson/Code/venv/bin/python -m pytest -q
```

Result:

```text
747 passed, 6 skipped, 491 warnings in 224.36s
```

Warnings are the existing scikit-learn feature-name warnings.

## Focused acceptance

```text
H10-C7 tests: 34 passed
Ruff: PASS
Compileall: PASS
Claim lint: PASS
Operator manifest: PASS, 41 operators
Open replay SHA256SUMS: PASS, 7 files
```

The full suite regenerated historical demonstration outputs. They were restored
byte-for-byte from `HEAD`; only the H10-C7 refactor code, tests, reports, and
new `open_replay_refactor` result directory remain changed.
