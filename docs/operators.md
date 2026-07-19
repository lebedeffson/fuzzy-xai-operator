# Operators

The complete formula/function/test/artifact map is `framework/fuzzyxai/operators_manifest.yaml`.

The manifest covers 15 mathematical constructions from chapters 2-3 and 15 evidence/framework operators: data quality, training dynamics, forgetting, subgroup averaging, rule extraction, rule complexity, measured rule significance, class concepts, local explanation, similar cases, counterfactuals, explanation graph, human explanation, quality metrics, and action.

Rules:

- no public operator contains scenario constants;
- missing inputs produce diagnostics;
- rule importance requires measured split metrics for ablation claims;
- class descriptions report uncovered fractions;
- a similarity score always names the metric and compared representation.
