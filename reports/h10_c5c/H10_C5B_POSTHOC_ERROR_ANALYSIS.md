# H10-C5b post-hoc error analysis

This report does not rescore H10-C5b and does not modify its final negative
result. It summarizes the immutable official per-incident output.

- O_ROUTE abstained in 19 of 24 incidents.
- O_ROUTE issued five diagnoses; all five were false localizations.
- Twelve incidents reported `contract_family_not_supported_by_evidence`.
- Six incidents reported `no_structural_candidate`.
- O_ROUTE and B_GREEDY exposed identical top-3 candidate lists in all 24
  incidents.

The non-rescoring incident table classifies the primary observable boundary as:

- `RETRIEVAL_MISS`: 14;
- `INSUFFICIENT_RUNTIME_EVIDENCE`: 7;
- `CONTRACT_INFERENCE_MISS`: 1;
- `GRAPH_CONSTRUCTION_MISS`: 1;
- `REPAIR_NOT_EXPRESSIBLE`: 1.

The frozen result retained only three candidates, so `retrieved_top10` is
reported as `NOT_ESTIMABLE_FROZEN_TOP3_ONLY`, not reconstructed.

The observed boundary precedes global cut optimization: set cover cannot
recover a Gold atom absent from the retrieved candidate pool. The 24 official
held-out incidents are restricted to post-hoc error analysis and must not be
rescored.
