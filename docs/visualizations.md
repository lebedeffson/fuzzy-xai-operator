# Explanation views

| View | Question | Required channels | Main limitation |
|---|---|---|---|
| `explanation_story` | How did data become an action? | claims and graph | absent stages remain missing |
| `data_profile` | Is the object unusual relative to reference/subgroup? | reference data | deviation is not automatically an error |
| `training_trace` | When was the object learned or forgotten? | epoch history | unavailable for a model without checkpoints |
| `knowledge_atlas` | Which class concepts and rules exist? | rules/concepts | surrogate rules are labelled |
| `decision_evidence` | What supports, contradicts, and limits the decision? | claims | association is not causation |
| `similar_cases` | Which cases are similar and in what representation? | reference cases | score is not probability |
| `counterfactuals` | What measured/plausible change alters output? | reference/counterfactual channel | feasibility is domain-dependent |
| `rule_ablation` | What changes with and without a rule? | measured before/after metrics | no chart is generated from guessed effects |
| `provenance` | Where did every claim and action come from? | graph | full audit may be large |
| `audit` | Are schema, claims, and graph valid? | trace and graph | audit does not certify deployment |

```python
result.visualize(view="decision_evidence", backend="plotly", output="decision.html")
result.visualize(view="training_trace", backend="matplotlib", output="training.png")
```
