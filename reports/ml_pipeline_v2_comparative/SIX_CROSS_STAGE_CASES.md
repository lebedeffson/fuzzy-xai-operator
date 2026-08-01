# Six cross-stage cases

## S11_TARGET_LEAKAGE

- B0: detected `False`; available standard log fields do not include RouteGraph contracts.
- B1: detected `False`; MLflow preserves registered run data without inferring a diagnosis.
- B2: detected `True`; contract `TARGET_NOT_IN_FEATURES`.
- O: `DATA_PREPARATION` / `TARGET_NOT_IN_FEATURES` / `breast_cancer_dataset`; action `BLOCK`.
- Repair and recertification: `False` / `False`.

## S12_SPLIT_OVERLAP

- B0: detected `False`; available standard log fields do not include RouteGraph contracts.
- B1: detected `False`; MLflow preserves registered run data without inferring a diagnosis.
- B2: detected `False`; contract `None`.
- O: `DATA_SPLIT` / `TRAIN_VALIDATION_TEST_DISJOINTNESS` / `deterministic_splitter`; action `BLOCK`.
- Repair and recertification: `False` / `False`.

## S13_PREPROCESSOR_FULL_FIT

- B0: detected `False`; available standard log fields do not include RouteGraph contracts.
- B1: detected `False`; MLflow preserves registered run data without inferring a diagnosis.
- B2: detected `False`; contract `None`.
- O: `PREPROCESSING` / `PREPROCESSOR_FIT_SCOPE` / `standard_scaler`; action `BLOCK`.
- Repair and recertification: `True` / `True`.

## S16_MODEL_ARTIFACT_TAMPER

- B0: detected `False`; available standard log fields do not include RouteGraph contracts.
- B1: detected `False`; MLflow preserves registered run data without inferring a diagnosis.
- B2: detected `True`; contract `MODEL_ARTIFACT_HASH`.
- O: `MODEL_ARTIFACT` / `MODEL_ARTIFACT_HASH` / `serialized_model`; action `BLOCK`.
- Repair and recertification: `True` / `True`.

## S2_EXPLAINER_VERSION_MISMATCH

- B0: detected `False`; available standard log fields do not include RouteGraph contracts.
- B1: detected `False`; MLflow preserves registered run data without inferring a diagnosis.
- B2: detected `False`; contract `None`.
- O: `POST_HOC_EXPLANATION` / `MODEL_EXPLAINER_VERSION` / `logistic_regression`; action `BLOCK`.
- Repair and recertification: `False` / `False`.

## S17_SHAP_INCONSISTENCY

- B0: detected `False`; available standard log fields do not include RouteGraph contracts.
- B1: detected `False`; MLflow preserves registered run data without inferring a diagnosis.
- B2: detected `True`; contract `EXPLANATION_OUTPUT_CONSISTENCY`.
- O: `POST_HOC_EXPLANATION` / `EXPLANATION_OUTPUT_CONSISTENCY` / `shap_linear_explainer`; action `BLOCK`.
- Repair and recertification: `False` / `False`.
