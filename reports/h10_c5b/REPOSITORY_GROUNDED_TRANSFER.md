# H10-C5b Repository-Grounded Transfer

- protocol_id: `h10-c5b-repository-grounded-v1`
- status: `H10_C5B_NOT_SUPPORTED`
- parent_result: `H10_C5_NOT_SUPPORTED`
- parent_result_modified: `False`
- primary_endpoint: `joint_file_symbol_contract_hit_at_3`
- primary_comparison: `O_ROUTE_vs_B_GREEDY`
- held_out_incidents: `24`
- held_out_repositories: `12`
- development_incidents: `0`
- coverage_min: `0.7`
- gold_leakage_audit: `PASS`
- runtime_evidence_complete: `True`
- runtime_evidence_status_counts: `{'BUG_REPRODUCED_WITH_TRACE': 24}`
- recovery_claim_enabled: `False`
- sealed_scoring_enabled: `False`
- held_out_scored: `True`
- input_manifest_sha256: `a22a62380249e7744f7387ec0d65b91d6717889b56de8af8e7dcc034a6e59f46`
- bootstrap: `{'comparison': 'O_ROUTE_vs_B_GREEDY', 'repository_count': 12, 'mean_difference': 0.0, 'ci_lower': 0.0, 'ci_upper': 0.0, 'bootstrap_p_two_sided': 1.0, 'iterations': 10000}`

The method consumed only buggy-revision repository structure and runtime evidence. Future patches were disclosed only to the independent Gold scorer after prediction. Natural recovery remains disabled unless project execution evidence is separately registered.
