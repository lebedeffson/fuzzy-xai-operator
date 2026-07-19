# Canonical visualization API

`fuzzyxai.visualization` is the canonical visualization namespace.

All new renderers consume `ExplanationViewModel`, which is serializable to JSON
and can be rendered without access to Python core objects:

```python
result = fuzzyxai.FuzzyXAI.wrap(model).explain(X, evidence=evidence)
result.export_json("explanation.json")
result.plot("dashboard.png")
```

MATLAB consumes the same file:

```matlab
addpath("framework/fuzzyxai/matlab");
result = fuzzyxai.loadResult("explanation.json");
fuzzyxai.dashboard(result);
```

The historical `fuzzyxai.visual` and `fuzzyxai.viz` modules remain compatibility
namespaces for composition and proof-route artifacts. They must not define a
second explanation data contract.
