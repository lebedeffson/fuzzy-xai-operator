# Traceability

Every explanation is a directed `ExplanationGraph`. Nodes represent data, anomalies, training events, rules, concepts, similar cases, counterfactuals, prediction, diagnostics, and action. Edges use explicit relations such as `supported_by`, `derived_from`, `similar_to`, `changed_by`, and `validated_by`.

Evidence references point to model attributes, controlled interventions, or source profiles. Input, model, and ExplainPlan fingerprints prevent a report from being silently reused with another run.

The audit summary lists every graph node and missing channel. Human text may not introduce facts absent from the graph.
