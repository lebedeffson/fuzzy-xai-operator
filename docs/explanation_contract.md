# Explanation contract

`ExplanationViewModel` schema `2.0` remains the backwards-compatible JSON envelope. Its scientific center is now:

```text
ExplanationResult
|- evidence
|- ExplanationClaim[]
|- ExplanationGraph
|- diagnostics
|- action
`- provenance
```

Every `ExplanationClaim` has an explicit status, evidence references, limitations, applicability, and optional metric semantics. Claims are construction material, not user prose. `HumanExplanation` groups, ranks, deduplicates, and translates supported claims into decision, reasons, concerns, reliability, action, and tested changes. Claim IDs remain inside card provenance and are hidden from domain-user text. A missing channel produces an explicit limitation; it cannot silently generate a sentence.

The result discloses the highest evidenced level E0-E5 plus `available_channels`, `missing_channels`, `native_channels`, and `surrogate_channels`. Callable, ANFIS, tree, linear, and future neural adapters can therefore expose different depths without pretending to be equally transparent.

Evidence level and reader profile are independent. E0-E5 states what the model run can prove. `domain_user`, `ml_engineer`, `researcher`, and `auditor` determine how those proofs are communicated. The verified public methods are `result.explain_for(...)` and `result.summary(audience=..., detail=...)`.

`ExplanationVisualSpec` is a separate typed presentation contract. Matplotlib, Plotly, MATLAB, and a future web client consume it and do not infer scientific semantics from arbitrary dictionary keys.

Required provenance includes adapter, model fingerprint, input checksum, dataset version, ExplainPlan checksum, object IDs, run parameters, generation time, missing evidence, claims, and graph edges. Backwards-compatible operator fields (`model`, `fuzzy`, `route`, `disagreement`, `risk`, `layers`) remain available.
