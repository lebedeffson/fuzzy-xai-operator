# Visualization

FuzzyXAI does not treat a collection of unrelated charts as an explanation. The primary visual object is the claim-centered route:

```text
Data -> Training -> Model knowledge -> Decision -> Action
```

`ExplanationVisualSpec` is the typed presentation boundary shared by renderers. It contains percentile-based feature profiles, separate training tracks, a class/rule atlas, supporting and contradicting claims, similar cases with named representations, counterfactuals, and provenance. Renderers never substitute rule coverage for missing importance and never combine `gamma`, `Delta`, and `rho` into one homogeneous score.

```python
result.visualize(view="explanation_story", backend="matplotlib")
result.visualize(view="data_profile", backend="plotly", output="profile.html")
result.visualize(view="training_trace", output="training.png")
result.inspect("claim:C004").visualize(view="provenance")
```

The legacy `result.plot(kind="dashboard")` call remains an alias for `explanation_story` during the compatibility cycle. MATLAB reads the same JSON via `fuzzyxai.loadResult` and renders `fuzzyxai.explanationStory` or `fuzzyxai.dataProfile`; it does not recompute scientific evidence.
