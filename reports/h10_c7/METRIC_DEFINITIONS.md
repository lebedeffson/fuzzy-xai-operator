# H10-C7 replay metric definitions

- `retrieval_coverage`: at least one candidate was returned.
- `contract_coverage`: the leading candidate has a registered contract rather
  than `UNKNOWN_CONTRACT`.
- `confirmed_diagnosis_coverage`: the system emitted
  `DIAGNOSIS_CONFIRMED`.
- `repair_coverage`: a registered executable repair was available. Repair was
  not executed in this replay and this value remains zero.
- `false_localization`: confirmed but incorrect joint localization divided by
  all incidents. This population-normalized compatibility metric is not a
  precision estimate.
- `selective_precision`: correct confirmed joint localizations divided by all
  confirmed diagnoses.
- `conditional_confirmation_error`: one minus selective precision.
- `context_lines`: sum of AST `lineno`/`end_lineno` spans for returned
  candidates.
- `runtime_ms`: measured controller time for the variant, not project runtime
  and not engineer time.

Two empty or unavailable states are never treated as positive evidence.
