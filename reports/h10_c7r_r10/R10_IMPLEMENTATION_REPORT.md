# H10-C7R R10 implementation report

R10 fixes the runtime chronology representation and replaces direct
repository-wide symbol compression with a file-first retrieval path.

Implemented components:

- monotonic runtime `sequence_id`, timestamp, thread, depth and occurrence
  counts;
- a bounded full event tail plus aggregated prefix;
- typed argument, return, exception, assertion, last-writer and value-flow
  observations;
- fail-closed raw-runtime readiness auditing;
- top-20 file retrieval followed by a maximum 200-symbol pool;
- source-aware reranking with candidate-specific causal bonuses;
- a bounded schema and planner for at most two read-only targeted probes;
- contract inference only after localization.

The disclosed 40 incidents have not been recollected or rescored. Their old
runtime files do not contain recoverable execution chronology, so they cannot
be used to claim R10 causal performance.

Status:
`H10_C7R_R10_IMPLEMENTED_AWAITING_CAUSAL_DEVELOPMENT_RECOLLECTION`.

Scientific result: `NOT_EVALUATED`.

At release commit `4642278...`, no container execution, neural inference, new
held-out creation, or held-out scoring had been performed. The subsequent
technical recollection run `30566975780` executed the locked ten development
containers and produced five ready and five fail-closed incidents. It did not
perform development scoring; details are preserved in
`R10_RECOLLECTION_RUN_1_REPORT.md`.

Targeted probes were not executed. R10D remains unavailable until both a
source-aware model lock and candidate-specific probe observations exist.

## Verification

- Local focused compatibility: `176 passed`.
- GitHub focused R10 gate: `54 passed`.
- GitHub full regression: `819 passed, 6 skipped`.
- Ruff, compileall, claim lint and operator manifest: `PASS`.
- Parent H10-C7R-v1 and R9 protected hashes: `PASS`.

## Commit and CI provenance

- Implementation commit: `e74570a9272527bde92e7a089d1d27a722c3e378`.
- Implementation CI run: `30563208205`.
- Release commit: `4642278de6a8d60aa6bb3d8b301e48398737020f`.
- Release CI run: `30563604221`.
- The six files changed between those commits are confined to
  `reports/h10_c7r_r10/` and `results/h10_c7r_r10/`.
- No framework code, collector, scoring logic, tests, workflow, protocol lock,
  or parent result changed between the implementation and release commits.
