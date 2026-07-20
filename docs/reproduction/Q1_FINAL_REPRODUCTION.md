# Q1 Final Reproduction

This protocol reproduces the final technical evidence without manufacturing external human records.

## Local smoke

```bash
python -m pip install -r requirements.lock
python -m pip install --no-deps -e .
make reproduce-q1-final-smoke
```

The smoke profile checks contracts, report generation, claims, artifact identity, forbidden wording and fail-closed external gates. It does not support real-data claims.

## Full technical reproduction

```bash
make reproduce-q1-final
make verify-q1-final
make q1-final-archives
make q1-final-archive-check
```

The full command downloads public datasets into `.cache/q1-final`, trains native multiclass models with five seeds, evaluates the frozen explainer cohorts, executes H1-H6 and scalability, and rebuilds reports and dissertation artifacts. Raw public datasets are not copied into release archives.

## Docker

```bash
docker compose -f docker-compose.q1-final.yml run --rm reproduce-q1-final
```

Use `smoke-q1-final` instead of `reproduce-q1-final` for the contract-only container check.

## External studies

`study/q1_final` contains frozen stimuli, consent materials, blank schemas and scorers. When genuine anonymized records are absent, the external gates remain `open`; no script generates participant or reviewer responses. Stable release verification must fail until approval or exemption, signed records, raw anonymized responses and scorer outputs satisfy the preregistered contracts.

## Outputs

- `release_evidence/q1_final/`: technical and external gate evidence;
- `reports/q1_final/`: final and reviewer-facing reports;
- `dissertation_artifacts/q1_final/`: generated chapter inserts and tables;
- `release_artifacts/q1_final/`: verified ZIP archives and SHA256 files.

All orchestration is single-threaded at the process level. Optional numerical libraries are also constrained through `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` and `NUMEXPR_NUM_THREADS`.
