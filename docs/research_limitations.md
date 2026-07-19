# Research limitations

- The framework is model-agnostic through adapters, not universally model-complete.
- Torch, Keras, and ONNX adapters are not implemented in this release.
- XGBoost and LightGBM use the generic `predict_proba` contract when installed; internal evidence depends on a custom adapter.
- Linear rule-like statements and neural concepts are surrogate evidence and are labelled accordingly.
- Feature importance and similarity are associations, not causal explanations.
- Mask overlap, embedding similarity, and feature similarity are different metrics and cannot be interchanged.
- The object-85 protocol is controlled synthetic evidence, not a production benchmark.
- Medical examples are research-only and are not clinical conclusions.
- Missing evidence produces review/insufficient evidence rather than inferred scientific metrics.
- MATLAB files require an external MATLAB or Octave runner; absence of that runner is reported, not hidden.
