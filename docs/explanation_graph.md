# Explanation graph

`ExplanationGraph` is the primary provenance object of FuzzyXAI. It connects measured evidence to claims, diagnostics, and the recommended action.

```text
evidence -> claim -> diagnostic -> action
prediction --------------------------^
```

Supported claims must have an incoming `supports_claim`, `derived_from`, `observed_during`, `lost_during`, or `changed_by` edge. `validate_reachability()` reports dangling edges, unsupported supported-claims, and actions that cannot be traced to both prediction and a claim or diagnostic.

```python
errors = result.explanation_graph.validate_reachability()
claim_graph = result.explanation_graph.trace_claim("C-004")
action_graph = result.explanation_graph.trace_action()
object_graph = result.explanation_graph.subgraph(subject_id="85")
```

`evidence_status` describes whether a claim is supported. `effect` describes whether its meaning is favorable or adverse. `severity` controls escalation. These fields are never interchangeable.
