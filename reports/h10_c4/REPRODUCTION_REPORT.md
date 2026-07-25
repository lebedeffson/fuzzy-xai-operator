# H10-C4 Reproduction

```bash
make h10-c4-test
make h10-c4-verify
```

The published result tables are the output of the valid frozen run. Re-running
`make h10-c4-run` is not required to verify the package and may change measured
wall-clock fields. Use `make h10-c4-verify` for deterministic verification.

Verified on 2026-07-25:

- H10-C4 focused tests: `28 passed`;
- Ruff changed scope: `PASS`;
- parent H10-C3 immutability: `54/54 PASS`;
- H10-C4 output and overlap gates: `PASS`;
- manuscript claim lint: `PASS`;
- result manifest checksums: `PASS`.

The protocol lock predates result generation. See
`reports/h10_c4/FULL_REGRESSION_REPORT.md` for the isolated legacy-suite
resource limitation observed on this workstation.
