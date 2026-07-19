# Model adapters

Every adapter implements `predict()` and declares `capabilities()`. Optional channels include probabilities, feature importance, rules, embeddings, gradients, training history, and checkpoints.

The pipeline never calls an unsupported channel. It returns a partial graph and lists missing evidence.

Built-in level-1 adapters:

- `CallableAdapter` for Python callables;
- `PredictProbaAdapter` for XGBoost/LightGBM-like APIs when installed;
- `SklearnAdapter` for linear, tree, forest, and boosting estimators;
- `NativeRuleAdapter` for ANFIS/fuzzy models exposing auditable `rules_`;
- `CustomAdapter` as the extension point.

Tree paths are model-native. Linear rule-like statements are marked `surrogate=True`. Neural rules must never be called native unless the model exposes a genuine rule structure.
