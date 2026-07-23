# H10-C3 R4 validation report

- Implementation commit: `e729834c077ecb5c0011d9fb85d5f00b10129f18`
- Full regression: `533 passed, 4 skipped`
- Focused R4/diagnostics: `71 passed`
- Ruff: `PASS`
- Gate: `READY_FOR_SEALED_GENERATION`
- Sealed status: `SEALED_SCORED`
- Sealed opening count: `1`
- Scientific status: `SCIENTIFIC_PASS`
- Sealed private plaintext distributed before scoring: `false`
- Sealed payload: `AES-256-GCM encrypted`

## Open R4 results

### H10-C3a

- Development effect: `0.475308641975`, 95% CI `[0.416666666667; 0.533950617284]`, Holm p `0.000299970003`.
- Protocol-validation effect: `0.473765432099`, 95% CI `[0.416666666667; 0.529320987654]`, Holm p `0.000299970003`.
- Positive pipeline families: `6/6`.
- Power: `1.000000`, lower bound `0.999615987522`, `10000` simulations.
- Confirmatory status: `CONFIRMATORY_PASS`.

### H10-C3b

- Development effect: `0.375000000000`, 95% CI `[0.322530864198; 0.427469135802]`, Holm p `0.000299970003`.
- Protocol-validation effect: `0.362654320988`, 95% CI `[0.316358024691; 0.412037037037]`, Holm p `0.000299970003`.
- Positive pipeline families: `6/6`.
- Power: `1.000000`, lower bound `0.999615987522`, `10000` simulations.
- Confirmatory status: `CONFIRMATORY_PASS`.

## Sealed confirmatory results

The encrypted sealed payload was opened once at
`2026-07-23T23:33:34.585247+00:00`. The opening counter changed from `0` to
`1` before decryption. The official scorer exited with code `0`.

### H10-C3a

- Effect over the locked `typed_without_optimization` baseline:
  `0.527426160338`.
- Hierarchical 95% CI: `[0.459915611814; 0.596638655462]`.
- Holm-adjusted p: `0.000299970003`.
- Positive pipeline families: `6/6`.
- Normalized cost-regret effect: `0.244175299790`.
- Cost-regret 95% CI: `[0.210077175940; 0.278760963797]`.
- Cost-regret Holm-adjusted p: `0.000299970003`.
- Automatic classification: `CONFIRMATORY_PASS`.

### H10-C3b

- Effect over the locked `weighted_greedy` baseline:
  `0.333333333333`.
- Hierarchical 95% CI: `[0.255230125523; 0.419491525424]`.
- Holm-adjusted p: `0.000299970003`.
- Positive pipeline families: `6/6`.
- Automatic classification: `CONFIRMATORY_PASS`.

### Safety and reproduction

- False certification: `0.0`.
- New critical violations: `0.0`.
- Runtime p95: `1.047843999913 ms`.
- Official result SHA256:
  `3fb4e2ba43c16552e1d52083d1ffec2da312e1c72df5d33079d28e4ac32fe0c8`.
- Independent postopen reproduction: `PASS`; 240 private templates, 1,920
  method rows, zero non-runtime scientific-field mismatches, exact registered
  bootstrap statistics, and the same `SCIENTIFIC_PASS` classification.

## Boundaries

- Development and protocol-validation results remain open evidence and are
  reported separately from the sealed result.
- The sealed result supports controlled algorithmic route diagnostics,
  optimal-cut membership, and full route recertification under the locked
  H10-C3 R4 design.
- Human factors and expert usefulness were not evaluated.
- Production safety, natural-fault generalization, clinical validity, and
  organizational effectiveness were not evaluated.
