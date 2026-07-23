# Reproduce H10-C3 R4 postopen

Use a clean checkout of the signed preopen tag and the disclosed postopen
folder. This verifier decrypts the released payload directly and never calls
the official opening API.

```bash
git checkout h10-c3-r4-v23.3-preopen-final
git verify-tag h10-c3-r4-v23.3-preopen-final

PYTHONPATH=experiments/h10_c3_r4/src:framework/fuzzyxai:. \
python scripts/reproduce_h10_c3_r4_postopen.py \
  --repo . \
  --official-dir /path/to/postopen \
  --output-dir /tmp/h10-c3-r4-reproduction
```

Expected invariants:

- disclosed seed commitment: `PASS`;
- encrypted and plaintext commitments: `PASS`;
- private templates: `240`;
- overlap with open template banks: `0`;
- reconstructed cases: `240`;
- reconstructed method rows: `1920`;
- non-runtime scientific-field mismatches: `0`;
- registered statistics: `PASS`;
- classification: `SCIENTIFIC_PASS`;
- reproduction opening record created: `false`.

Runtime measurements are intentionally remeasured and are not required to be
byte-identical. The registered scientific fields, bootstrap statistics, and
classification must be identical.
