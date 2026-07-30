# H10-C7R Prospective Lock Validation

## Scope

- Protocol: `H10-C7R-v1`
- Frozen method commit: `358ed40a0fb7f5adc1291695ff15affa39cae485`
- Method: `R5`, budget 20
- Baseline: `B_BM25`, budget 160
- Minimum recall: 0.80
- Scientific result: `NOT_EVALUATED`
- Held-out created/scored: `false/false`

Contract inference is a descriptive secondary metric and is not a support
gate for this retrieval-only protocol.

## Reproduced Development Boundary

- Incidents: 40
- Repositories: 16
- R5 top-20 signature:
  `f3ad88d430e922ba6ecfeff0de2e32031675610f800c539b0cfa3d33625623e1`
- Frozen-prefix mismatches: 0
- Source release SHA256:
  `cd33dd6689623c043dfb74485d1618d10b2922f6a9934047db5ce43293f56a4b`

## Validation

- H10-C7 focused tests: `64 passed`
- H10-C7R focused tests: `6 passed`
- Full regression: `783 passed, 6 skipped`
- Ruff: `PASS`
- Compileall: `PASS`
- Claim lint: `PASS`
- Parent-result immutability: `114 files, PASS`
- Clean-worktree parent checksum inventory: `880 files, PASS`

The first unconstrained full-regression invocation was not used because it
omitted the registered H10-C3 import root. The canonical invocation was:

```bash
PYTHONPATH=experiments/h10_c3/src:framework/fuzzyxai:. \
  /home/lebedeffson/Code/venv/bin/python -m pytest -q
```

No held-out manifest or Gold was opened while constructing this lock.
