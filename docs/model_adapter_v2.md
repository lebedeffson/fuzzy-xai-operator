# ModelAdapterV2 contract

`ModelAdapterV2` is the canonical capability-based boundary between a predictive model and FuzzyXAI. A model
family is not considered supported merely because prediction works. The adapter must disclose task semantics,
input/output schemas, available evidence channels, channel origin, limitations, and deterministic provenance.

## Resolution order

1. Explicit adapter instance or registered adapter name.
2. Exact native family adapter.
3. Installed plugin from the `fuzzyxai.adapters` entry-point group.
4. Generic `predict_proba` or `decision_function` contract.
5. Callable black-box fallback.

The selected adapter and every matched/rejected candidate are available through `capability_report()`.

## Evidence origin

- `native`: emitted directly by the model runtime, such as a tree path or ANFIS rule.
- `derived_from_native`: computed from native differentiable state, such as integrated gradients.
- `surrogate`: produced by a separately fitted approximation and accepted only with measured fidelity.
- `external`: supplied by an audited external method.

Missing channels remain missing. The framework must not replace a missing local contribution with global feature
importance or describe a surrogate rule as native.

## Third-party adapters

Plugins register an adapter through the `fuzzyxai.adapters` Python entry-point group. A plugin must expose a
`supports_model(model)` predicate and pass `run_adapter_conformance(...)`. The conformance suite checks prediction,
parity, deterministic fingerprinting, serializability, and capability truthfulness.

## Optional runtimes

XGBoost, LightGBM, CatBoost, PyTorch, TensorFlow/Keras, and ONNX Runtime are lazy optional dependencies. Importing
`fuzzyxai` does not import them. Runtime evidence uses typed statuses: `pass`, `implemented_not_executed`,
`dependency_unavailable`, `unsupported`, or `failed`. A successful source-code inspection is never promoted to
`pass`; that status requires a checksummed runtime report. The release candidate records each optional family on
Python 3.11 and Python 3.12 and rejects cross-version inconsistencies.
