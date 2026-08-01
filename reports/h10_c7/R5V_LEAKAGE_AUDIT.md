# H10-C7-R5V leakage audit

Status: `PASS_FOR_AUDIT_ONLY`

- Gold is loaded from the physically separate `gold.jsonl`.
- Gold is used only by `pair_target` and target-alignment reporting.
- Candidate retrieval and candidate-contract inference run before Gold is
  consulted.
- Candidate-specific features use frozen candidate outputs and observable
  runtime events.
- `incident_id` and repository names are identifiers for grouping and
  reporting, not verifier features.
- No pair model was fitted or evaluated because the audit stop conditions
  were reached.

This audit does not authorize V0/V1 execution or held-out scoring.
