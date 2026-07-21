# Practical FuzzyXAI controller

The practical controller turns prediction, explanation and route artifacts into a budget-aware operational action.
It does not replace the model prediction and does not invent unavailable evidence.

## API

```python
from fuzzyxai import assess_action, cost_profile
from fuzzyxai.practical_controller import (
    DeploymentContext,
    ExplanationArtifact,
    PracticalPolicy,
    PredictionArtifact,
    ReviewBudget,
    RouteArtifacts,
)

assessment = assess_action(
    prediction_artifact,
    explanation_artifact,
    route_artifacts,
    deployment_context,
    ReviewBudget(0.20),
    cost_profile("unsafe_accept_sensitive"),
    policy=frozen_policy,
)

print(assessment.action)
print(assessment.operational_risk)
print(assessment.reason_codes)
print(assessment.trace_id)
```

Use `assess_batch()` when a review budget must be allocated across a population. Use `assess_stream()` for bounded
memory processing. A replay can be checked with `verify_replay()`.

## Action boundary

- `accept`: the route is certified and the object was not selected for review.
- `short_review`: a certified case has elevated risk and receives a bounded review.
- `full_review`: mandatory evidence is incomplete or route risk requires complete review.
- `block`: a formal frozen contract is violated. Low confidence alone never blocks.

If mandatory reviews exceed the available budget, the result records `budget_feasible=false`; it does not silently
accept those objects.

## Evidence boundary

The current tracked measurements are formative development evidence. The independent confirmatory test remains
sealed until all dataset manifests, OOF split declarations and the real blinded AI formative run 2 are present.

```bash
make PYTHON=.venv/bin/python practical-release-check
make PYTHON=.venv/bin/python practical-controller-freeze
```

The first command validates the technical formative candidate. The second must return `BLOCKED` until real external
inputs are supplied. A formative advantage must not be described as confirmatory evidence.

