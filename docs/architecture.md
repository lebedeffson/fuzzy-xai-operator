# Architecture

```text
Model / dataset / training run
  -> ModelAdapter (facts and capabilities)
  -> evidence collectors
  -> chapter 2-3 core operators
  -> ExplanationEvidence
  -> ExplanationGraph
  -> ExplanationViewModel
  -> Matplotlib / HTML / MATLAB
```

`fuzzyxai.core` owns the mathematical implementation. `fuzzyxai.operators` is a typed public facade and must only delegate to core. `fuzzyxai.evidence` observes data, training, model knowledge, similar cases, and counterfactual tests. `fuzzyxai.visualization` never computes scientific metrics.

The invariant operator route is `E -> T -> gamma -> Delta -> rDelta -> rho -> D -> chi -> action`. Prediction, evidence quality, diagnostics, and action are separate values.

Legacy `fuzzyxai.visual` and `fuzzyxai.viz` are compatibility shims for one release. New code imports `fuzzyxai.visualization`.
