# H10-C7 implementation report

## Boundary

- Parent commit: `1fed9ac074295bc4b6cad33841b73a4022bcafcd`
- Protocol: `H10-C7-v1`
- Scientific result: `NOT_EVALUATED`
- Development data collected/scored: no/no
- Held-out created/scored: no/no
- H10-C5b/H10-C5c results changed: no

## Implemented

- observable-manifest and separate development-Gold channels;
- R0-R8 development tournament and repository-fold assignment;
- B_TRACE, B_BM25, B_DENSE, B_RRF, B_GREEDY, B_REPOGRAPH,
  B_AGENTLESS_LOC and O_ROUTE baselines;
- deterministic BM25, RRF and personalized RepoGraph;
- pinned local-only dense and cross-encoder interfaces;
- structural reranking and multi-obligation global ordering;
- hierarchical contract inference with `UNKNOWN_CONTRACT`;
- entropy/cost active evidence requests and registered-probe reranking;
- incident routing and a bounded twelve-action read-only explorer;
- strict repair verification over Fail-to-Pass, regression, recertification,
  new critical violations and rollback;
- fail-closed development and held-out guards.

## Verification

- Focused H10-C7: `22 passed`.
- Ruff changed scope: `PASS`.
- H10-C5b/H10-C5c/H9 compatibility: `126 passed`.
- Full repository regression: `735 passed, 6 skipped`.
- Parent result SHA256: `PASS`.
- Canonical operator manifest SHA256: `PASS`.

Hashing encoders are smoke-test backends only. Real development remains blocked
until GraphCodeBERT, UniXcoder and the registered cross-encoder weights are
available locally and their SHA256 values replace the pending registry fields.
