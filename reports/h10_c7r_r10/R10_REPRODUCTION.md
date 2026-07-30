# H10-C7R R10 reproduction

## Lightweight checks

```bash
make h10-c7r-r10-test \
  H10_C3_PYTHON=/home/lebedeffson/Code/venv/bin/python
```

## Runtime readiness

The old runtime evidence must not be silently upgraded. Audit raw events:

```bash
make h10-c7r-r10-runtime-audit \
  H10_C3_PYTHON=python3.11 \
  H10_C7R_R10_MANIFEST=/work/r10/HELD_OUT_MANIFEST.jsonl \
  H10_C7R_R10_READINESS=/work/r10/R10_RUNTIME_READINESS.json
```

Exit code `2` and status `R10_RUNTIME_RECOLLECTION_REQUIRED` mean that
development scoring is prohibited.

## Development scoring

Run only after every disclosed development incident passes readiness:

```bash
make h10-c7r-r10-development \
  H10_C3_PYTHON=python3.11 \
  H10_C7R_R10_MANIFEST=/work/r10/HELD_OUT_MANIFEST.jsonl \
  H10_C7R_R10_GOLD=/work/r10/HELD_OUT_GOLD.jsonl \
  H10_C7R_R10_READINESS=/work/r10/R10_RUNTIME_READINESS.json \
  H10_C7R_R10_OUTPUT=/work/r10/development
```

The scorer uses the disclosed H10-C7R-v1 incidents as development data and
keeps `scientific_result=NOT_EVALUATED`. It cannot create or score a new
held-out set.

R10B, R10C and R10D remain unavailable until a local source-aware model lock is
created. There is no network fallback.

## Provenance

`R10_RELEASE_PROVENANCE.json` distinguishes the implementation verification
from the final release verification. Run the offline audit with:

```bash
python scripts/ch4_revision/verify_h10_c7r_r10_provenance.py
```
