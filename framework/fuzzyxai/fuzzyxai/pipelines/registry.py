from __future__ import annotations

from dataclasses import asdict, dataclass

from fuzzyxai.diagnostics.contracts import canonical_sha256


@dataclass(frozen=True)
class PipelineRegistration:
    pipeline_id: str
    pipeline_version: str
    task_type: str
    dataset_id: str
    preprocessor_id: str
    model_id: str
    explainer_id: str
    random_state: int
    model_parameters: dict[str, object]
    supported_contracts: tuple[str, ...]
    supported_repairs: tuple[str, ...]

    @property
    def sha256(self) -> str:
        return canonical_sha256(asdict(self))


COMMON_CONTRACTS = (
    "DATASET_IDENTITY",
    "DATASET_SCHEMA",
    "TARGET_NOT_IN_FEATURES",
    "CLASS_MAPPING",
    "TRAIN_VALIDATION_TEST_DISJOINTNESS",
    "SPLIT_REPRODUCIBILITY",
    "PREPROCESSOR_VERSION",
    "PREPROCESSOR_FIT_SCOPE",
    "FEATURE_ORDER",
    "FEATURE_COUNT",
    "TRANSFORM_OUTPUT_SCHEMA",
    "FINITE_TRANSFORMED_VALUES",
    "TRAINING_CONFIGURATION",
    "MODEL_CONVERGENCE",
    "MODEL_FEATURE_SCHEMA",
    "TRAINING_DATA_HASH",
    "MODEL_ARTIFACT_HASH",
    "METRIC_SANITY",
    "MODEL_INPUT_SCHEMA",
    "PREDICTION_OBJECT_BINDING",
    "PREDICTION_OUTPUT_SANITY",
    "MODEL_EXPLAINER_VERSION",
    "EXPLANATION_OBJECT_ID",
    "EXPLANATION_OUTPUT_CONSISTENCY",
    "EXPLANATION_FEATURE_SCHEMA",
    "REQUIRED_PROVENANCE",
    "REDUCTION_LOSS_LIMIT",
    "USER_CLAIM_EVIDENCE_BINDING",
)

COMMON_REPAIRS = (
    "restore_clean_transform",
    "restore_split_manifest",
    "refit_preprocessor_on_train",
    "restore_feature_order",
    "restore_registered_model",
    "rerun_registered_explainer",
    "restore_output_semantics",
)


PIPELINE_REGISTRY: dict[str, PipelineRegistration] = {
    "breast-cancer-logreg-linearshap": PipelineRegistration(
        "breast-cancer-logreg-linearshap",
        "1.0.0",
        "BINARY_CLASSIFICATION",
        "sklearn_breast_cancer",
        "sklearn.StandardScaler",
        "sklearn.LogisticRegression",
        "shap.LinearExplainer",
        1729,
        {"solver": "lbfgs", "max_iter": 500, "random_state": 1729},
        COMMON_CONTRACTS,
        COMMON_REPAIRS,
    ),
    "wine-logreg-linearshap": PipelineRegistration(
        "wine-logreg-linearshap",
        "1.0.0",
        "MULTICLASS_CLASSIFICATION",
        "sklearn_wine",
        "sklearn.StandardScaler",
        "sklearn.LogisticRegression",
        "shap.LinearExplainer",
        1729,
        {"solver": "lbfgs", "max_iter": 500, "random_state": 1729},
        COMMON_CONTRACTS,
        COMMON_REPAIRS,
    ),
    "diabetes-ridge-linearshap": PipelineRegistration(
        "diabetes-ridge-linearshap",
        "1.0.0",
        "REGRESSION",
        "sklearn_diabetes",
        "sklearn.StandardScaler",
        "sklearn.Ridge",
        "shap.LinearExplainer",
        1729,
        {"alpha": 1.0},
        COMMON_CONTRACTS,
        COMMON_REPAIRS,
    ),
    "digits-random-forest-treeshap": PipelineRegistration(
        "digits-random-forest-treeshap",
        "1.0.0",
        "MULTICLASS_CLASSIFICATION",
        "sklearn_digits",
        "sklearn.StandardScaler",
        "sklearn.RandomForestClassifier",
        "shap.TreeExplainer",
        1729,
        {"n_estimators": 40, "max_depth": 8, "min_samples_leaf": 2, "n_jobs": 1, "random_state": 1729},
        COMMON_CONTRACTS,
        COMMON_REPAIRS,
    ),
    "mixed-logreg-linearshap": PipelineRegistration(
        "mixed-logreg-linearshap",
        "1.0.0",
        "BINARY_CLASSIFICATION",
        "fuzzyxai_mixed_features_v1",
        "sklearn.ColumnTransformer(StandardScaler+OneHotEncoder)",
        "sklearn.LogisticRegression",
        "shap.LinearExplainer",
        1729,
        {"solver": "lbfgs", "max_iter": 500, "random_state": 1729},
        COMMON_CONTRACTS,
        COMMON_REPAIRS,
    ),
}


def get_pipeline_registration(pipeline_id: str) -> PipelineRegistration:
    try:
        return PIPELINE_REGISTRY[pipeline_id]
    except KeyError as exc:
        raise KeyError(f"unknown registered pipeline: {pipeline_id}") from exc


def list_pipeline_registrations() -> tuple[PipelineRegistration, ...]:
    return tuple(PIPELINE_REGISTRY.values())


__all__ = [
    "COMMON_CONTRACTS",
    "COMMON_REPAIRS",
    "PIPELINE_REGISTRY",
    "PipelineRegistration",
    "get_pipeline_registration",
    "list_pipeline_registrations",
]
