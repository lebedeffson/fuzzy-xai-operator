# H10-C5c Practical Diagnostic Loop

Status: `IMPLEMENTED_AWAITING_DEVELOPMENT_DATA`

This implementation is a prospective technical cycle. It does not rescore the
24 official H10-C5b incidents and does not establish a scientific effect.

## Implemented path

```text
typed runtime events
-> per-test executed slice
-> high-recall candidate retrieval
-> independent contract inference
-> evidence calibration
-> greedy or global diagnostic selection
-> three-level diagnosis
-> minimal evidence request
-> registered repair plan
-> isolated FAIL_TO_PASS execution
-> regression execution
-> route recertification
```

The runtime event schema rejects Gold fields. Traceback, dynamic calls,
coverage, artifact access, configuration reads, loaded dependencies and
imports remain associated with the failing test that produced them.

Candidate retrieval and contract inference are separate. Unsupported contract
evidence yields `UNREGISTERED_CONTRACT`; it is not coerced to the nearest
registered family. `B_GREEDY` ranks individual candidates, while `O_ROUTE`
orders candidates using jointly selected obligation-covering cuts.

The three report states are:

- `DIAGNOSIS_CONFIRMED`
- `DIAGNOSIS_CANDIDATES`
- `INSUFFICIENT_EVIDENCE`

The latter two may generate a read-only `EvidenceRequest`. A repair is
executable only when its operation has an explicitly registered provider.
Semi-automatic source or adapter changes require approval. Execution occurs in
an isolated copy and succeeds only after FAIL_TO_PASS, regression and
recertification all pass.

## Prospective boundary

Development requires at least 30 incidents from at least eight repositories,
complete traceback evidence and typed runtime event streams. The official
H10-C5b held-out repositories are rejected. The locked development gates are
candidate Recall@10, contract accuracy, coverage, false localization,
strategy distinction and zero Gold leakage.

No H10-C5c held-out set exists and no H10-C5c scientific scoring has occurred.
