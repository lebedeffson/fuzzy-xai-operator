# H10-C7-R5V audit reproduction

The audit uses the autonomous open replay bundle. It requires no network,
project dependency installation, or failing-test execution.

```bash
make h10-c7-r5v-audit \
  H10_C3_PYTHON=/home/lebedeffson/Code/venv/bin/python \
  H10_C7_REPLAY_BUNDLE=/home/lebedeffson/.local/share/fuzzyxai/h10-c7/h10-c7-open-replay-bundle \
  H10_C7_VERIFICATION_OUTPUT=results/h10_c7/verification
```

The command intentionally exits non-zero while
`TARGET_ALIGNMENT_AUDIT_FAILED` remains present. The machine-readable audit
files and `R5V_STATUS.json` are written before the fail-closed exit.

Expected status:

```text
H10_C7_R5V_BLOCKED_AUDIT
Scientific result: NOT_EVALUATED
V0 executed: false
V1 executed: false
Held-out created/scored: false/false
```
