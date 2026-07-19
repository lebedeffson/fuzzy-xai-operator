# Explanation planning and quality

`ExplanationPlanner` selects evidence channels from the model capability descriptor, requested task, evidence
budget, and expected explanation quality. It does not run every available explainer.

## Quality dimensions

- prediction parity: adapter output equals the model output;
- faithfulness: local perturbations agree with the reported direction where measurable;
- surrogate fidelity: approximation agreement on the evaluated neighborhood;
- reconstruction error: contribution sum versus the model score or margin;
- stability: explanation variation under small controlled perturbations;
- completeness: requested channels with valid evidence;
- sparsity: number of user-facing factors retained.

A surrogate channel is blocked when fidelity is absent or below the configured threshold. A failed channel is not
silently replaced by a fabricated value. The result records the blocked channel, diagnostic reason, and available
fallbacks.

## Public result operations

```python
result = FuzzyXAI.wrap(model, task="auto").explain_one(item, reference_data=X_train)
result.capability_report()
result.quality_report()
result.why_not(target=other_class)

batch = fx.explain_batch(items)
global_result = fx.explain_global(reference_data=X_train)
comparison = fx.compare_models(other_fx, item)
```

`why_not()` reports missing evidence rather than inventing a counterfactual. Batch/global/model comparison methods
retain the same claim and provenance boundaries as `explain_one()`.
