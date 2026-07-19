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

Every `ExplanationClaim` has an explicit status, evidence references, limitations, applicability, and optional metric semantics. User, expert, and audit prose is assembled only from claims and includes visible claim IDs. A missing channel produces `insufficient_evidence`; it cannot silently generate a sentence.

The result discloses the highest evidenced level E0-E5 plus `available_channels`, `missing_channels`, `native_channels`, and `surrogate_channels`. Callable, ANFIS, tree, linear, and future neural adapters can therefore expose different depths without pretending to be equally transparent.

`ExplanationVisualSpec` is a separate typed presentation contract. Matplotlib, Plotly, MATLAB, and a future web client consume it and do not infer scientific semantics from arbitrary dictionary keys.

Required provenance includes adapter, model fingerprint, input checksum, dataset version, ExplainPlan checksum, object IDs, run parameters, generation time, missing evidence, claims, and graph edges. Backwards-compatible operator fields (`model`, `fuzzy`, `route`, `disagreement`, `risk`, `layers`) remain available.
