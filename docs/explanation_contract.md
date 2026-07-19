# Explanation contract

`ExplanationViewModel` schema `2.0` is the canonical JSON boundary. Its schema is `fuzzyxai/schemas/explanation_view_model.schema.json`.

The `layers` section contains `DataEvidence`, `TrainingObjectTrace`, subgroup evidence, `LearnedRule`, `ClassConcept`, `SimilarCaseEvidence`, and `CounterfactualEvidence`. `explanation_graph` connects those facts to prediction, diagnostics, and action. `human_explanations` contains user, expert, and audit views generated only from graph facts.

Required provenance includes adapter, model fingerprint, input checksum, dataset version, ExplainPlan checksum, object IDs, run parameters, generation time, and missing evidence.

Backward-compatible operator fields (`model`, `fuzzy`, `route`, `disagreement`, `risk`) remain available.
