# H10 v19 repository integration validation

## Repository state

- Branch: `feat/h10-audit-confirmatory-v19`.
- Base: `origin/main@e678c047896b81ed796ec6be9cdc4370cf12ddca`.
- Input handoff SHA256: `c7060ab1f88bb853cd73435bdd5a0e95e4ce9f0cbab2554f77c2a79d854443c4`.
- Frozen raw-results SHA256 before and after reproduction:
  `51a2bf9a73ad2fcd691ca327dbe9c3d0d59c298d21811c3746d097d142495574`.
- Sealed scoring repeated: `false`.
- Opening count in imported evidence: `1`.
- Post-lock tuning: `false`.
- Old v16 changed: `false`.
- Old v18 changed: `false`.

## Checks

- Full repository pytest: `465 passed, 4 skipped`.
- H10 focused pytest: `14 passed`.
- H10 focused Ruff: `PASS`.
- Full-repository Ruff: `FAIL`, 315 inherited findings outside the H10
  integration scope. These findings are disclosed and were not suppressed or
  bulk-edited as part of this branch.
- `make reproduce-h10`: `PASS` from frozen outputs without vault access.
- Evidence entries: `188`; locator, source-file, value, and SHA256 failures:
  `0`.
- Release payload filename scan: no vault, key, `.h10_private`, or
  `raw_labels` entries.

## Scientific status

`TECHNICAL_PASS_SCIENTIFIC_INVALID`

The supplied H10-L and H10-R numbers reproduce, but their source and repair
truth use a static oracle catalog that semantically duplicates the evaluated
H10 taxonomy. The repository therefore blocks scientific release of those two
primary claims. H10-C remains secondary descriptive, H10-U descriptive, and
H10-T is supported only as byte-identical deterministic trace generation.
