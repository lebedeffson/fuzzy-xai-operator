# ExplanationVisualSpec

`ExplanationVisualSpec` schema `1.1` is the strict presentation contract shared by Matplotlib and Plotly. Renderers do not inspect arbitrary scientific objects or recompute metrics.

The contract contains typed sections for overview, story stages, feature profiles, training points, class concepts, rules, decision claims, similar cases, counterfactuals, rule ablation, provenance, and audit.

```python
spec = result.view_model.visual_spec
typed = ExplanationVisualSpec.from_dict(spec)
assert typed.to_dict() == spec
```

The JSON schema is `fuzzyxai/schemas/explanation_visual_spec.schema.json`. Legacy `1.0` payloads can be passed through `migrate_visual_spec`; unavailable fields remain explicit limitations instead of receiving invented values.
