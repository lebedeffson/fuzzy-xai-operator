# H10-C5c Post-hoc Oracle Decomposition

- Status: `POSTHOC_DECOMPOSITION_COMPLETE`
- Analysis set: 30 disclosed development incidents from 8 repositories.
- Scientific result: `NOT_EVALUATED`.
- Held-out created/scored: `false/false`.
- Official development status modified: `false`.
- Interpretation: `RETRIEVAL_CONTRACT_INTERACTION`.

| Variant | Coverage | Joint Hit@1 | Joint Hit@3 | Cut contains Gold |
|---|---:|---:|---:|---:|
| Baseline | 0.7667 | 0.0000 | 0.0000 | 0.0000 |
| Oracle Candidate | 0.8667 | 0.0667 | 0.1333 | 0.0667 |
| Oracle Contract | 0.3000 | 0.0333 | 0.1000 | 0.0333 |
| Oracle Candidate + Contract | 0.3667 | 0.1000 | 0.1667 | 0.1000 |

Oracle interventions use disclosed development Gold. The values are diagnostic
upper bounds and must not be reported as method performance. No hypothesis test
was performed and no new prospective cycle or held-out set was opened.
