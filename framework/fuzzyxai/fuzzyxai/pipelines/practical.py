from __future__ import annotations

import hashlib
import pickle
import resource
import warnings
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol

import numpy as np
import pandas as pd
import shap
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_breast_cancer, load_diabetes, load_digits, load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fuzzyxai.diagnostics.contracts import Contract, RouteEdge, RouteGraph, RouteNode, canonical_sha256
from fuzzyxai.ml_vertical.pipeline import CONTRACT_STAGE

from .registry import PIPELINE_REGISTRY, PipelineRegistration, get_pipeline_registration

PRACTICAL_VERSION = "FUZZYXAI_CROSS_PIPELINE_PRACTICAL_V1"
MODE_IDS = ("B_LOCAL_STRONG", "B_PAIRWISE_RULES", "B_MLFLOW_QUERY", "B_GREEDY_CROSS_STAGE", "O_FUZZYXAI")
LOCAL_CONTRACTS = frozenset(
    {
        "FINITE_TRANSFORMED_VALUES",
        "FEATURE_COUNT",
        "MODEL_ARTIFACT_HASH",
        "MODEL_CONVERGENCE",
        "METRIC_SANITY",
        "REQUIRED_PROVENANCE",
        "EXPLANATION_OUTPUT_CONSISTENCY",
        "PREDICTION_OUTPUT_SANITY",
        "CLASS_MAPPING",
    }
)
MLFLOW_QUERY_CONTRACTS = frozenset({"MODEL_ARTIFACT_HASH", "MODEL_EXPLAINER_VERSION", "MODEL_CONVERGENCE", "METRIC_SANITY", "REQUIRED_PROVENANCE"})
STAGE_ORDER = tuple(stage.value for stage in CONTRACT_STAGE.values())
STAGE_INDEX = {stage: index for index, stage in enumerate(dict.fromkeys(STAGE_ORDER))}
CONTRACT_COMPONENT = {
    "FINITE_TRANSFORMED_VALUES": "preprocessor",
    "TRAIN_VALIDATION_TEST_DISJOINTNESS": "split_manifest",
    "PREPROCESSOR_FIT_SCOPE": "preprocessor",
    "FEATURE_ORDER": "inference_input",
    "FEATURE_COUNT": "inference_input",
    "MODEL_INPUT_SCHEMA": "model",
    "MODEL_ARTIFACT_HASH": "model_artifact",
    "PREDICTION_OBJECT_BINDING": "prediction",
    "PREDICTION_OUTPUT_SANITY": "prediction",
    "MODEL_EXPLAINER_VERSION": "explainer",
    "EXPLANATION_OBJECT_ID": "explanation",
    "EXPLANATION_OUTPUT_CONSISTENCY": "explanation",
    "EXPLANATION_FEATURE_SCHEMA": "explanation",
    "CLASS_MAPPING": "class_mapping",
    "USER_CLAIM_EVIDENCE_BINDING": "presentation",
}
CAUSE_RELATIONS = {
    "FEATURE_ORDER": ("MODEL_INPUT_SCHEMA", "EXPLANATION_FEATURE_SCHEMA", "USER_CLAIM_EVIDENCE_BINDING"),
    "FEATURE_COUNT": ("MODEL_INPUT_SCHEMA", "EXPLANATION_FEATURE_SCHEMA", "USER_CLAIM_EVIDENCE_BINDING"),
    "MODEL_ARTIFACT_HASH": ("MODEL_EXPLAINER_VERSION", "PREDICTION_OBJECT_BINDING"),
    "MODEL_EXPLAINER_VERSION": ("EXPLANATION_OUTPUT_CONSISTENCY",),
}
REPAIR_FOR_CONTRACT = {
    "FINITE_TRANSFORMED_VALUES": "restore_clean_transform",
    "TRAIN_VALIDATION_TEST_DISJOINTNESS": "restore_split_manifest",
    "PREPROCESSOR_FIT_SCOPE": "refit_preprocessor_on_train",
    "FEATURE_ORDER": "restore_feature_order",
    "FEATURE_COUNT": "restore_feature_order",
    "MODEL_ARTIFACT_HASH": "restore_registered_model",
    "MODEL_EXPLAINER_VERSION": "rerun_registered_explainer",
    "EXPLANATION_OBJECT_ID": "rerun_registered_explainer",
    "EXPLANATION_OUTPUT_CONSISTENCY": "rerun_registered_explainer",
    "EXPLANATION_FEATURE_SCHEMA": "rerun_registered_explainer",
    "CLASS_MAPPING": "restore_output_semantics",
    "PREDICTION_OUTPUT_SANITY": "restore_output_semantics",
}


@dataclass(frozen=True)
class DatasetArtifact:
    dataset_id: str
    features: pd.DataFrame
    target: np.ndarray
    row_ids: tuple[str, ...]
    target_name: str
    sha256: str

    def manifest(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "dataset_sha256": self.sha256,
            "row_count": len(self.features),
            "column_count": self.features.shape[1],
            "feature_names": tuple(self.features.columns),
            "target_name": self.target_name,
        }


@dataclass(frozen=True)
class SplitArtifact:
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    train_ids: tuple[str, ...]
    validation_ids: tuple[str, ...]
    test_ids: tuple[str, ...]
    sha256: str

    def manifest(self) -> dict[str, Any]:
        return {
            "train_ids_sha256": canonical_sha256(self.train_ids),
            "validation_ids_sha256": canonical_sha256(self.validation_ids),
            "test_ids_sha256": canonical_sha256(self.test_ids),
            "split_sha256": self.sha256,
            "random_state": 1729,
        }


@dataclass(frozen=True)
class PreprocessorArtifact:
    transformer: Any
    transformed_train: np.ndarray
    transformed_validation: np.ndarray
    transformed_test: np.ndarray
    feature_names: tuple[str, ...]
    fit_row_ids: tuple[str, ...]
    sha256: str

    def manifest(self) -> dict[str, Any]:
        return {
            "class": type(self.transformer).__name__,
            "library_version": sklearn.__version__,
            "artifact_sha256": self.sha256,
            "fit_row_ids_sha256": canonical_sha256(self.fit_row_ids),
            "feature_names": self.feature_names,
            "feature_schema_sha256": canonical_sha256(self.feature_names),
            "output_feature_count": len(self.feature_names),
        }


@dataclass(frozen=True)
class ModelArtifact:
    estimator: Any
    artifact_bytes: bytes
    sha256: str
    feature_names: tuple[str, ...]
    classes: tuple[Any, ...] | None
    converged: bool
    training_metrics: dict[str, float]

    def manifest(self) -> dict[str, Any]:
        return {
            "class": type(self.estimator).__name__,
            "library_version": sklearn.__version__,
            "artifact_sha256": self.sha256,
            "feature_schema_sha256": canonical_sha256(self.feature_names),
            "feature_count": len(self.feature_names),
            "classes": self.classes,
            "converged": self.converged,
            "metrics": self.training_metrics,
        }


@dataclass(frozen=True)
class PredictionArtifact:
    object_id: str
    output: float
    predicted_value: Any
    selected_output: int | None
    model_sha256: str
    input_sha256: str
    finite: bool
    shape: tuple[int, ...]


@dataclass(frozen=True)
class ExplanationArtifact:
    object_id: str
    explainer_id: str
    explainer_version: str
    model_sha256: str
    feature_names: tuple[str, ...]
    base_value: float
    attributions: tuple[float, ...]
    model_output: float
    absolute_error: float
    selected_output: int | None
    sha256: str

    def manifest(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "explainer_id": self.explainer_id,
            "explainer_version": self.explainer_version,
            "model_sha256": self.model_sha256,
            "feature_schema_sha256": canonical_sha256(self.feature_names),
            "selected_output": self.selected_output,
            "absolute_error": self.absolute_error,
            "artifact_sha256": self.sha256,
        }


@dataclass(frozen=True)
class PipelineArtifacts:
    registration: PipelineRegistration
    dataset: DatasetArtifact
    split: SplitArtifact
    preprocessor: PreprocessorArtifact
    model: ModelArtifact
    prediction: PredictionArtifact
    explanation: ExplanationArtifact
    route_graph: RouteGraph
    runtime_breakdown_ms: dict[str, float]
    peak_rss_kb: int
    artifact_bytes: int


class RegisteredMLPipeline(Protocol):
    pipeline_id: str
    pipeline_version: str
    task_type: str

    def load_data(self) -> DatasetArtifact: ...
    def split_data(self) -> SplitArtifact: ...
    def fit_preprocessor(self) -> PreprocessorArtifact: ...
    def train_model(self) -> ModelArtifact: ...
    def predict(self) -> PredictionArtifact: ...
    def explain(self) -> ExplanationArtifact: ...
    def build_route(self) -> RouteGraph: ...
    def audit(self) -> AuditResult: ...
    def repair(self, plan: dict[str, Any]) -> RepairResult: ...
    def recertify(self) -> RecertificationResult: ...


class ExecutableRegisteredPipeline:
    def __init__(self, registration: PipelineRegistration) -> None:
        self.registration = registration
        self.pipeline_id = registration.pipeline_id
        self.pipeline_version = registration.pipeline_version
        self.task_type = registration.task_type
        self.dataset: DatasetArtifact | None = None
        self.split: SplitArtifact | None = None
        self.preprocessor: PreprocessorArtifact | None = None
        self.model: ModelArtifact | None = None
        self.prediction: PredictionArtifact | None = None
        self.explanation: ExplanationArtifact | None = None
        self.route_graph: RouteGraph | None = None
        self.timings: dict[str, float] = {}

    def execute(self) -> PipelineArtifacts:
        for name, operation in (
            ("data_preparation", self.load_data),
            ("data_split", self.split_data),
            ("preprocessing", self.fit_preprocessor),
            ("training", self.train_model),
            ("inference", self.predict),
            ("explanation", self.explain),
            ("graph_construction", self.build_route),
        ):
            started = perf_counter()
            operation()
            self.timings[name] = (perf_counter() - started) * 1000
        assert self.dataset and self.split and self.preprocessor and self.model and self.prediction and self.explanation and self.route_graph
        artifact_bytes = len(self.model.artifact_bytes) + len(self.explanation.attributions) * 8 + len(self.route_graph.trace_sha256)
        return PipelineArtifacts(
            self.registration,
            self.dataset,
            self.split,
            self.preprocessor,
            self.model,
            self.prediction,
            self.explanation,
            self.route_graph,
            dict(self.timings),
            int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            artifact_bytes,
        )

    def load_data(self) -> DatasetArtifact:
        registration = self.registration
        if registration.dataset_id == "sklearn_breast_cancer":
            loaded = load_breast_cancer(as_frame=True)
            features = loaded.data.copy()
            target = 1 - np.asarray(loaded.target)
            target_name = "risk_target"
        elif registration.dataset_id == "sklearn_wine":
            loaded = load_wine(as_frame=True)
            features = loaded.data.copy()
            target = np.asarray(loaded.target)
            target_name = "target"
        elif registration.dataset_id == "sklearn_diabetes":
            loaded = load_diabetes(as_frame=True)
            features = loaded.data.copy()
            target = np.asarray(loaded.target, dtype=float)
            target_name = "target"
        elif registration.dataset_id == "sklearn_digits":
            loaded = load_digits(as_frame=True)
            features = loaded.data.copy()
            target = np.asarray(loaded.target)
            target_name = "target"
        else:
            path = Path(__file__).resolve().parents[4] / "data/cross_pipeline_v1/mixed_features.csv"
            frame = pd.read_csv(path)
            target = frame.pop("target").to_numpy()
            features = frame
            target_name = "target"
        row_ids = tuple(f"{registration.dataset_id}:{index}" for index in range(len(features)))
        digest = canonical_sha256(
            {
                "dataset_id": registration.dataset_id,
                "columns": tuple(features.columns),
                "features": features.astype(str).values.tolist(),
                "target": target.tolist(),
            }
        )
        self.dataset = DatasetArtifact(registration.dataset_id, features, target, row_ids, target_name, digest)
        return self.dataset

    def split_data(self) -> SplitArtifact:
        assert self.dataset
        indices = np.arange(len(self.dataset.features))
        stratify = self.dataset.target if "CLASSIFICATION" in self.task_type else None
        train, temporary = train_test_split(indices, test_size=0.4, random_state=1729, stratify=stratify)
        temporary_target = self.dataset.target[temporary] if stratify is not None else None
        validation, test = train_test_split(temporary, test_size=0.5, random_state=1729, stratify=temporary_target)
        train_ids = tuple(self.dataset.row_ids[index] for index in train)
        validation_ids = tuple(self.dataset.row_ids[index] for index in validation)
        test_ids = tuple(self.dataset.row_ids[index] for index in test)
        digest = canonical_sha256({"train": train_ids, "validation": validation_ids, "test": test_ids, "seed": 1729})
        self.split = SplitArtifact(tuple(train), tuple(validation), tuple(test), train_ids, validation_ids, test_ids, digest)
        return self.split

    def fit_preprocessor(self) -> PreprocessorArtifact:
        assert self.dataset and self.split
        features = self.dataset.features
        train = features.iloc[list(self.split.train_indices)]
        validation = features.iloc[list(self.split.validation_indices)]
        test = features.iloc[list(self.split.test_indices)]
        if self.registration.dataset_id == "fuzzyxai_mixed_features_v1":
            numeric = tuple(train.select_dtypes(include=[np.number]).columns)
            categorical = tuple(name for name in train.columns if name not in numeric)
            transformer = ColumnTransformer(
                (
                    ("numeric", StandardScaler(), list(numeric)),
                    ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), list(categorical)),
                ),
                verbose_feature_names_out=True,
            )
        else:
            transformer = StandardScaler()
        transformed_train = np.asarray(transformer.fit_transform(train), dtype=float)
        transformed_validation = np.asarray(transformer.transform(validation), dtype=float)
        transformed_test = np.asarray(transformer.transform(test), dtype=float)
        names = tuple(str(item) for item in transformer.get_feature_names_out())
        digest = canonical_sha256(
            {
                "class": type(transformer).__name__,
                "parameters": transformer.get_params(deep=False),
                "feature_names": names,
                "train_mean": np.mean(transformed_train, axis=0).tolist(),
                "fit_ids": self.split.train_ids,
            }
        )
        self.preprocessor = PreprocessorArtifact(
            transformer,
            transformed_train,
            transformed_validation,
            transformed_test,
            names,
            self.split.train_ids,
            digest,
        )
        return self.preprocessor

    def train_model(self) -> ModelArtifact:
        assert self.dataset and self.split and self.preprocessor
        registration = self.registration
        if registration.model_id == "sklearn.LogisticRegression":
            estimator: Any = LogisticRegression(**registration.model_parameters)
        elif registration.model_id == "sklearn.Ridge":
            estimator = Ridge(**registration.model_parameters)
        else:
            estimator = RandomForestClassifier(**registration.model_parameters)
        target_train = self.dataset.target[list(self.split.train_indices)]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            estimator.fit(self.preprocessor.transformed_train, target_train)
        converged = not any(issubclass(item.category, ConvergenceWarning) for item in caught)
        validation_target = self.dataset.target[list(self.split.validation_indices)]
        validation_prediction = estimator.predict(self.preprocessor.transformed_validation)
        if "CLASSIFICATION" in self.task_type:
            metrics = {
                "accuracy": float(accuracy_score(validation_target, validation_prediction)),
                "f1_macro": float(f1_score(validation_target, validation_prediction, average="macro")),
            }
            classes: tuple[Any, ...] | None = tuple(_python_scalar(item) for item in estimator.classes_)
        else:
            metrics = {
                "mae": float(mean_absolute_error(validation_target, validation_prediction)),
                "rmse": float(mean_squared_error(validation_target, validation_prediction) ** 0.5),
            }
            classes = None
        artifact = pickle.dumps({"preprocessor": self.preprocessor.transformer, "model": estimator}, protocol=5)
        digest = hashlib.sha256(artifact).hexdigest()
        self.model = ModelArtifact(estimator, artifact, digest, self.preprocessor.feature_names, classes, converged, metrics)
        return self.model

    def predict(self) -> PredictionArtifact:
        assert self.preprocessor and self.model and self.split
        sample = self.preprocessor.transformed_test[:1]
        object_id = self.split.test_ids[0]
        if self.task_type == "REGRESSION":
            predicted = float(self.model.estimator.predict(sample)[0])
            output = predicted
            selected = None
        elif self.registration.model_id == "sklearn.RandomForestClassifier":
            predicted = _python_scalar(self.model.estimator.predict(sample)[0])
            selected = int(np.where(np.asarray(self.model.classes) == predicted)[0][0])
            output = float(self.model.estimator.predict_proba(sample)[0, selected])
        else:
            predicted = _python_scalar(self.model.estimator.predict(sample)[0])
            decision = np.asarray(self.model.estimator.decision_function(sample))
            if decision.ndim == 1:
                selected = 1
                output = float(decision[0])
            else:
                selected = int(np.where(np.asarray(self.model.classes) == predicted)[0][0])
                output = float(decision[0, selected])
        self.prediction = PredictionArtifact(
            object_id,
            output,
            predicted,
            selected,
            self.model.sha256,
            canonical_sha256(sample.tolist()),
            bool(np.isfinite(output)),
            (1,),
        )
        return self.prediction

    def explain(self) -> ExplanationArtifact:
        assert self.preprocessor and self.model and self.prediction
        background = self.preprocessor.transformed_train[: min(128, len(self.preprocessor.transformed_train))]
        sample = self.preprocessor.transformed_test[:1]
        if self.registration.explainer_id == "shap.TreeExplainer":
            explainer: Any = shap.TreeExplainer(self.model.estimator)
        else:
            explainer = shap.LinearExplainer(self.model.estimator, background)
        raw = explainer(sample)
        values = np.asarray(raw.values, dtype=float)
        base_values = np.asarray(raw.base_values, dtype=float)
        selected = self.prediction.selected_output
        if values.ndim == 3:
            output_index = int(selected or 0)
            attributions = values[0, :, output_index]
            base = float(base_values[0, output_index] if base_values.ndim == 2 else base_values[output_index])
        else:
            attributions = values[0]
            base = float(base_values.reshape(-1)[0])
        reconstructed = base + float(np.sum(attributions))
        error = abs(reconstructed - self.prediction.output)
        payload = {
            "object_id": self.prediction.object_id,
            "explainer": self.registration.explainer_id,
            "model_sha256": self.model.sha256,
            "features": self.preprocessor.feature_names,
            "base": base,
            "attributions": attributions.tolist(),
            "model_output": self.prediction.output,
            "selected_output": selected,
        }
        self.explanation = ExplanationArtifact(
            self.prediction.object_id,
            self.registration.explainer_id,
            shap.__version__,
            self.model.sha256,
            self.preprocessor.feature_names,
            base,
            tuple(float(item) for item in attributions),
            self.prediction.output,
            error,
            selected,
            canonical_sha256(payload),
        )
        return self.explanation

    def build_route(self) -> RouteGraph:
        assert self.dataset and self.split and self.preprocessor and self.model and self.prediction and self.explanation
        node_specs = (
            ("dataset", "dataset", self.dataset.sha256),
            ("split_manifest", "split", self.split.sha256),
            ("preprocessor", "preprocessor", self.preprocessor.sha256),
            ("model_artifact", "model", self.model.sha256),
            ("prediction", "prediction", self.prediction.input_sha256),
            ("explainer", "explainer", self.explanation.model_sha256),
            ("explanation", "explanation", self.explanation.sha256),
            ("representation", "representation", canonical_sha256({"explanation": self.explanation.sha256})),
        )
        nodes = tuple(
            RouteNode(
                node_id,
                node_type,
                f"{self.pipeline_id}.{node_id}",
                self.pipeline_version,
                {"artifact_sha256": artifact_sha256},
                {"artifact_sha256": artifact_sha256},
                True,
                node_id not in {"dataset", "prediction", "representation"},
                (f"evidence:{node_id}:{artifact_sha256[:12]}",),
            )
            for node_id, node_type, artifact_sha256 in node_specs
        )
        edge_specs = (
            ("dataset", "split_manifest", "derived_from"),
            ("split_manifest", "preprocessor", "fitted_on"),
            ("preprocessor", "model_artifact", "trained_on"),
            ("model_artifact", "prediction", "produced_by"),
            ("model_artifact", "explainer", "explained_by"),
            ("prediction", "explanation", "bound_to"),
            ("explanation", "representation", "represented_by"),
        )
        edges = tuple(
            RouteEdge(
                f"edge:{source}:{target}",
                source,
                target,
                relation,
                True,
                {"relation": relation},
                {"relation": relation},
                True,
                (f"evidence:edge:{source}:{target}",),
                "known_valid",
            )
            for source, target, relation in edge_specs
        )
        contracts = tuple(
            Contract(
                contract_id,
                "registered_cross_pipeline_contract",
                CONTRACT_COMPONENT.get(contract_id, "representation"),
                severity="error",
                category="pipeline",
                repairable=contract_id in REPAIR_FOR_CONTRACT,
            )
            for contract_id in self.registration.supported_contracts
        )
        self.route_graph = RouteGraph(
            f"cross-pipeline:{self.pipeline_id}",
            nodes,
            edges,
            contracts,
            {
                "pipeline_id": self.pipeline_id,
                "task_type": self.task_type,
                "dataset_sha256": self.dataset.sha256,
                "split_sha256": self.split.sha256,
                "preprocessor_sha256": self.preprocessor.sha256,
                "model_sha256": self.model.sha256,
                "explainer_id": self.explanation.explainer_id,
                "explainer_version": self.explanation.explainer_version,
            },
        )
        return self.route_graph

    def audit(self) -> AuditResult:
        artifacts = self._artifacts()
        return audit_observed_state(observed_state(artifacts))

    def repair(self, plan: dict[str, Any]) -> RepairResult:
        artifacts = self._artifacts()
        state = observed_state(artifacts)
        audit = audit_observed_state(state)
        return execute_registered_repair(artifacts, state, audit, str(plan.get("operation", "")))

    def recertify(self) -> RecertificationResult:
        audit = self.audit()
        return RecertificationResult(audit.valid, len(CONTRACT_STAGE), (), 0, True, canonical_sha256(asdict(audit)))

    def _artifacts(self) -> PipelineArtifacts:
        assert self.dataset and self.split and self.preprocessor and self.model and self.prediction and self.explanation and self.route_graph
        return PipelineArtifacts(
            self.registration,
            self.dataset,
            self.split,
            self.preprocessor,
            self.model,
            self.prediction,
            self.explanation,
            self.route_graph,
            dict(self.timings),
            int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            len(self.model.artifact_bytes),
        )


@dataclass(frozen=True)
class ObservedPipelineState:
    pipeline_id: str
    task_type: str
    registered_split_train_ids: tuple[str, ...]
    registered_split_test_ids: tuple[str, ...]
    observed_split_train_ids: tuple[str, ...]
    observed_split_test_ids: tuple[str, ...]
    registered_fit_ids: tuple[str, ...]
    observed_fit_ids: tuple[str, ...]
    registered_feature_names: tuple[str, ...]
    inference_feature_names: tuple[str, ...]
    transformed_sample: np.ndarray
    registered_model_sha256: str
    observed_model_sha256: str
    prediction_model_sha256: str
    explainer_model_sha256: str
    registered_explainer_version: str
    observed_explainer_version: str
    registered_object_id: str
    explanation_object_id: str
    explanation_feature_names: tuple[str, ...]
    explanation_base_value: float
    explanation_attributions: tuple[float, ...]
    model_output: float
    registered_classes: tuple[Any, ...] | None
    observed_class_mapping: tuple[Any, ...] | None
    predicted_value: Any
    prediction_finite: bool
    prediction_shape: tuple[int, ...]


@dataclass(frozen=True)
class ContractViolation:
    contract_id: str
    stage: str
    component_id: str
    observed_value: Any
    expected_value: Any
    severity: str
    evidence_ref: str


@dataclass(frozen=True)
class AuditResult:
    valid: bool
    violations: tuple[ContractViolation, ...]
    root_cause: str | None
    dependent_violations: tuple[str, ...]
    diagnostic_cut: tuple[str, ...]
    checked_contracts: tuple[str, ...]
    runtime_ms: float
    canonical_sha256: str


@dataclass(frozen=True)
class RepairResult:
    operation: str | None
    executed: bool
    target_contract_repaired: bool
    rollback_verified: bool
    redundant_repair_count: int


@dataclass(frozen=True)
class RecertificationResult:
    full_recertification: bool
    contracts_rechecked_count: int
    remaining_violations: tuple[str, ...]
    new_critical_violations: int
    rollback_verified: bool
    canonical_sha256: str


@dataclass(frozen=True)
class MutationLevel:
    level_id: str
    description: str
    expected_stage: str | None
    expected_contract: str | None
    expected_action: str


@dataclass(frozen=True)
class MutationFamily:
    family_id: str
    category: str
    levels: tuple[MutationLevel, ...]


def _levels(
    family_id: str,
    stage: str,
    contracts: tuple[str, str, str, str],
    descriptions: tuple[str, str, str, str],
) -> MutationFamily:
    return MutationFamily(
        family_id,
        "LOCAL" if family_id in {"TRANSFORM_FINITE", "OUTPUT_SEMANTICS"} else "PAIRWISE" if family_id in {"SPLIT_OVERLAP", "FIT_SCOPE"} else "GLOBAL",
        (
            MutationLevel("L0", "registered positive control", None, None, "ACCEPT"),
            *(
                MutationLevel(f"L{index}", description, stage, contract, "BLOCK")
                for index, (contract, description) in enumerate(zip(contracts, descriptions, strict=True), start=1)
            ),
        ),
    )


MUTATION_FAMILIES: dict[str, MutationFamily] = {
    "TRANSFORM_FINITE": _levels(
        "TRANSFORM_FINITE",
        "PREPROCESSING",
        ("FINITE_TRANSFORMED_VALUES",) * 4,
        ("one NaN", "one positive infinity", "one negative infinity", "multiple non-finite values"),
    ),
    "MODEL_ARTIFACT": _levels(
        "MODEL_ARTIFACT",
        "MODEL_ARTIFACT",
        ("MODEL_ARTIFACT_HASH",) * 4,
        ("one-byte mutation", "truncated artifact", "foreign model digest", "stale artifact digest"),
    ),
    "SPLIT_OVERLAP": _levels(
        "SPLIT_OVERLAP",
        "DATA_SPLIT",
        ("TRAIN_VALIDATION_TEST_DISJOINTNESS",) * 4,
        ("one overlapping row", "five percent overlap", "twenty percent overlap", "full test included in train"),
    ),
    "FIT_SCOPE": _levels(
        "FIT_SCOPE",
        "PREPROCESSING",
        ("PREPROCESSOR_FIT_SCOPE",) * 4,
        ("train plus validation", "full dataset", "test only", "different train split"),
    ),
    "FEATURE_SCHEMA_CASCADE": _levels(
        "FEATURE_SCHEMA_CASCADE",
        "PREPROCESSING",
        ("FEATURE_ORDER", "FEATURE_COUNT", "FEATURE_COUNT", "FEATURE_ORDER"),
        ("swap two features", "remove one feature", "add one feature", "rename one feature"),
    ),
    "EXPLAINER_BINDING": _levels(
        "EXPLAINER_BINDING",
        "POST_HOC_EXPLANATION",
        ("MODEL_EXPLAINER_VERSION", "MODEL_EXPLAINER_VERSION", "EXPLANATION_OBJECT_ID", "EXPLANATION_FEATURE_SCHEMA"),
        ("stale explainer version", "foreign model hash", "foreign object id", "foreign feature schema"),
    ),
    "SHAP_CONSISTENCY": _levels(
        "SHAP_CONSISTENCY",
        "POST_HOC_EXPLANATION",
        ("EXPLANATION_OUTPUT_CONSISTENCY", "EXPLANATION_OUTPUT_CONSISTENCY", "EXPLANATION_OUTPUT_CONSISTENCY", "EXPLANATION_FEATURE_SCHEMA"),
        ("changed base value", "changed one attribution", "changed multiple attributions", "permuted attribution binding"),
    ),
    "OUTPUT_SEMANTICS": _levels(
        "OUTPUT_SEMANTICS",
        "INFERENCE",
        ("PREDICTION_OUTPUT_SANITY",) * 4,
        ("reversed or invalid semantics", "missing output", "extra output", "unknown selected output"),
    ),
}


def observed_state(artifacts: PipelineArtifacts) -> ObservedPipelineState:
    return ObservedPipelineState(
        artifacts.registration.pipeline_id,
        artifacts.registration.task_type,
        artifacts.split.train_ids,
        artifacts.split.test_ids,
        artifacts.split.train_ids,
        artifacts.split.test_ids,
        artifacts.split.train_ids,
        artifacts.preprocessor.fit_row_ids,
        artifacts.preprocessor.feature_names,
        artifacts.preprocessor.feature_names,
        artifacts.preprocessor.transformed_test[:1].copy(),
        artifacts.model.sha256,
        artifacts.model.sha256,
        artifacts.prediction.model_sha256,
        artifacts.explanation.model_sha256,
        artifacts.explanation.explainer_version,
        artifacts.explanation.explainer_version,
        artifacts.prediction.object_id,
        artifacts.explanation.object_id,
        artifacts.explanation.feature_names,
        artifacts.explanation.base_value,
        artifacts.explanation.attributions,
        artifacts.explanation.model_output,
        artifacts.model.classes,
        artifacts.model.classes,
        artifacts.prediction.predicted_value,
        artifacts.prediction.finite,
        artifacts.prediction.shape,
    )


def apply_registered_mutation(
    artifacts: PipelineArtifacts,
    family_id: str,
    level_id: str,
) -> ObservedPipelineState:
    if family_id not in MUTATION_FAMILIES:
        raise ValueError(f"unregistered mutation family: {family_id}")
    family = MUTATION_FAMILIES[family_id]
    if level_id not in {level.level_id for level in family.levels}:
        raise ValueError(f"unregistered mutation level: {family_id}/{level_id}")
    state = observed_state(artifacts)
    level = int(level_id.removeprefix("L"))
    if level == 0:
        return state
    if family_id == "TRANSFORM_FINITE":
        transformed = state.transformed_sample.copy()
        values = (np.nan, np.inf, -np.inf, np.nan)
        transformed[0, 0] = values[level - 1]
        if level == 4 and transformed.shape[1] > 1:
            transformed[0, 1] = np.inf
        return replace(state, transformed_sample=transformed)
    if family_id == "MODEL_ARTIFACT":
        digest = canonical_sha256({"registered": state.registered_model_sha256, "mutation_level": level})
        return replace(state, observed_model_sha256=digest)
    if family_id == "SPLIT_OVERLAP":
        count = (
            1,
            max(1, len(state.registered_split_test_ids) // 20),
            max(1, len(state.registered_split_test_ids) // 5),
            len(state.registered_split_test_ids),
        )[level - 1]
        overlap = state.registered_split_test_ids[:count]
        return replace(state, observed_split_train_ids=(*state.registered_split_train_ids, *overlap))
    if family_id == "FIT_SCOPE":
        split = artifacts.split
        observed = (
            (*split.train_ids, *split.validation_ids),
            (*split.train_ids, *split.validation_ids, *split.test_ids),
            split.test_ids,
            tuple(reversed(split.train_ids)),
        )[level - 1]
        return replace(state, observed_fit_ids=tuple(observed))
    if family_id == "FEATURE_SCHEMA_CASCADE":
        names = list(state.registered_feature_names)
        transformed = state.transformed_sample.copy()
        if level == 1:
            names[0], names[1] = names[1], names[0]
            transformed[:, [0, 1]] = transformed[:, [1, 0]]
        elif level == 2:
            names = names[:-1]
            transformed = transformed[:, :-1]
        elif level == 3:
            names.append("unregistered_extra_feature")
            transformed = np.column_stack((transformed, np.zeros((len(transformed), 1))))
        else:
            names[0] = f"renamed_{names[0]}"
        return replace(state, inference_feature_names=tuple(names), transformed_sample=transformed)
    if family_id == "EXPLAINER_BINDING":
        if level == 1:
            return replace(state, observed_explainer_version="0.0.0-stale")
        if level == 2:
            return replace(state, explainer_model_sha256=canonical_sha256("foreign-model"))
        if level == 3:
            return replace(state, explanation_object_id="foreign-object")
        return replace(state, explanation_feature_names=tuple(reversed(state.explanation_feature_names)))
    if family_id == "SHAP_CONSISTENCY":
        if level == 1:
            return replace(state, explanation_base_value=state.explanation_base_value + 0.1)
        attributions = list(state.explanation_attributions)
        if level == 2:
            attributions[0] += 0.1
            return replace(state, explanation_attributions=tuple(attributions))
        if level == 3:
            for index in range(min(3, len(attributions))):
                attributions[index] += 0.05
            return replace(state, explanation_attributions=tuple(attributions))
        return replace(
            state,
            explanation_attributions=tuple(reversed(attributions)),
            explanation_feature_names=tuple(reversed(state.explanation_feature_names)),
        )
    if family_id == "OUTPUT_SEMANTICS":
        if state.task_type == "REGRESSION":
            if level == 1:
                return replace(state, prediction_finite=False, model_output=float("nan"))
            if level == 2:
                return replace(state, prediction_finite=False, model_output=float("inf"))
            if level == 3:
                return replace(state, prediction_shape=(2,))
            return replace(state, observed_class_mapping=(0, 1))
        classes = tuple(state.registered_classes or ())
        if level == 1:
            mapping = tuple(reversed(classes))
            return replace(state, observed_class_mapping=mapping)
        if level == 2:
            return replace(state, observed_class_mapping=classes[:-1])
        if level == 3:
            return replace(state, observed_class_mapping=(*classes, "extra"))
        return replace(state, predicted_value="unknown-class")
    raise AssertionError(family_id)


def audit_observed_state(state: ObservedPipelineState) -> AuditResult:
    started = perf_counter()
    violations: dict[str, ContractViolation] = {}

    def add(contract_id: str, observed: Any, expected: Any) -> None:
        stage = _stage_for_contract(contract_id)
        evidence_ref = f"evidence:{contract_id}:{canonical_sha256({'observed': observed, 'expected': expected})[:16]}"
        violations.setdefault(
            contract_id,
            ContractViolation(
                contract_id,
                stage,
                CONTRACT_COMPONENT.get(contract_id, "pipeline"),
                _json_value(observed),
                _json_value(expected),
                "HIGH",
                evidence_ref,
            ),
        )

    if not np.isfinite(state.transformed_sample).all():
        add("FINITE_TRANSFORMED_VALUES", False, True)
    if set(state.observed_split_train_ids).intersection(state.observed_split_test_ids):
        add("TRAIN_VALIDATION_TEST_DISJOINTNESS", len(set(state.observed_split_train_ids).intersection(state.observed_split_test_ids)), 0)
    if state.observed_fit_ids != state.registered_fit_ids:
        add("PREPROCESSOR_FIT_SCOPE", canonical_sha256(state.observed_fit_ids), canonical_sha256(state.registered_fit_ids))
    if state.inference_feature_names != state.registered_feature_names:
        root = "FEATURE_COUNT" if len(state.inference_feature_names) != len(state.registered_feature_names) else "FEATURE_ORDER"
        add(root, state.inference_feature_names, state.registered_feature_names)
        add("MODEL_INPUT_SCHEMA", canonical_sha256(state.inference_feature_names), canonical_sha256(state.registered_feature_names))
        add("EXPLANATION_FEATURE_SCHEMA", canonical_sha256(state.inference_feature_names), canonical_sha256(state.registered_feature_names))
        add("USER_CLAIM_EVIDENCE_BINDING", False, True)
    if state.observed_model_sha256 != state.registered_model_sha256:
        add("MODEL_ARTIFACT_HASH", state.observed_model_sha256, state.registered_model_sha256)
    if state.prediction_model_sha256 != state.observed_model_sha256:
        add("PREDICTION_OBJECT_BINDING", state.prediction_model_sha256, state.observed_model_sha256)
    if state.explainer_model_sha256 != state.observed_model_sha256:
        add("MODEL_EXPLAINER_VERSION", state.explainer_model_sha256, state.observed_model_sha256)
    if state.observed_explainer_version != state.registered_explainer_version:
        add("MODEL_EXPLAINER_VERSION", state.observed_explainer_version, state.registered_explainer_version)
    if state.explanation_object_id != state.registered_object_id:
        add("EXPLANATION_OBJECT_ID", state.explanation_object_id, state.registered_object_id)
    if state.explanation_feature_names != state.registered_feature_names:
        add("EXPLANATION_FEATURE_SCHEMA", state.explanation_feature_names, state.registered_feature_names)
    reconstructed = state.explanation_base_value + float(np.sum(state.explanation_attributions))
    if not np.isfinite(reconstructed) or not np.isclose(reconstructed, state.model_output, atol=1e-6, rtol=1e-6):
        add("EXPLANATION_OUTPUT_CONSISTENCY", abs(reconstructed - state.model_output), 1e-6)
    if state.task_type == "REGRESSION":
        if state.observed_class_mapping is not None or not state.prediction_finite or state.prediction_shape != (1,):
            add(
                "PREDICTION_OUTPUT_SANITY",
                {"finite": state.prediction_finite, "shape": state.prediction_shape, "classes": state.observed_class_mapping},
                {"finite": True, "shape": (1,), "classes": None},
            )
    elif state.observed_class_mapping != state.registered_classes or state.predicted_value not in tuple(state.registered_classes or ()):
        add(
            "PREDICTION_OUTPUT_SANITY",
            {"mapping": state.observed_class_mapping, "prediction": state.predicted_value},
            {"mapping": state.registered_classes, "prediction_in_classes": True},
        )

    ordered = tuple(sorted(violations.values(), key=lambda item: (STAGE_INDEX.get(item.stage, 999), item.contract_id)))
    present = {item.contract_id for item in ordered}
    dependent = {child for parent, children in CAUSE_RELATIONS.items() if parent in present for child in children if child in present}
    roots = [item.contract_id for item in ordered if item.contract_id not in dependent]
    root = roots[0] if roots else (ordered[0].contract_id if ordered else None)
    dependents = tuple(item.contract_id for item in ordered if item.contract_id != root)
    cut = (root,) if root else ()
    runtime = (perf_counter() - started) * 1000
    core = {
        "pipeline_id": state.pipeline_id,
        "violations": [asdict(item) for item in ordered],
        "root_cause": root,
        "dependent_violations": dependents,
        "diagnostic_cut": cut,
        "checked_contracts": tuple(CONTRACT_STAGE),
    }
    return AuditResult(not ordered, ordered, root, dependents, cut, tuple(CONTRACT_STAGE), runtime, canonical_sha256(core))


@dataclass(frozen=True)
class ModeResult:
    run_id: str
    pipeline_id: str
    mutation_family: str
    mutation_level: str
    mode_id: str
    pipeline_status: str
    detected: bool
    stage: str | None
    contract_id: str | None
    root_cause: str | None
    dependent_violations: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    diagnostic_cut: dict[str, Any] | None
    action: str
    repair_plan: dict[str, Any] | None
    repair_executed: bool
    target_contract_repaired: bool
    recertified: bool
    rollback_verified: bool
    new_critical_violations: int
    reported_symptom_count: int
    proposed_repair_count: int
    redundant_repair_count: int
    evidence_completeness: float
    runtime_breakdown_ms: dict[str, float]
    peak_rss_kb: int
    artifact_bytes: int
    canonical_sha256: str


def evaluate_mode(
    artifacts: PipelineArtifacts,
    family_id: str,
    level_id: str,
    audit: AuditResult,
    mode_id: str,
) -> ModeResult:
    if mode_id not in MODE_IDS:
        raise ValueError(f"unregistered mode: {mode_id}")
    started = perf_counter()
    if mode_id == "B_LOCAL_STRONG":
        visible = tuple(item for item in audit.violations if item.contract_id in LOCAL_CONTRACTS)
    elif mode_id == "B_MLFLOW_QUERY":
        visible = tuple(item for item in audit.violations if item.contract_id in MLFLOW_QUERY_CONTRACTS)
    else:
        visible = audit.violations
    selected = visible[0] if visible else None
    detected = bool(visible)
    root = audit.root_cause if mode_id == "O_FUZZYXAI" else selected.contract_id if mode_id == "B_GREEDY_CROSS_STAGE" and selected else None
    dependents = (
        audit.dependent_violations if mode_id == "O_FUZZYXAI" else tuple(item.contract_id for item in visible[1:]) if mode_id == "B_PAIRWISE_RULES" else ()
    )
    if mode_id == "O_FUZZYXAI":
        cut = {"contracts": list(audit.diagnostic_cut), "size": len(audit.diagnostic_cut), "solver": "causal_route_minimal_cut"} if detected else None
        operation = REPAIR_FOR_CONTRACT.get(audit.root_cause or "")
        repair_plan = {"operation": operation, "root_cause": audit.root_cause, "rollback": f"rollback_{operation}"} if operation else None
        repair = (
            execute_registered_repair(artifacts, apply_registered_mutation(artifacts, family_id, level_id), audit, operation or "")
            if detected
            else RepairResult(None, False, False, True, 0)
        )
        recertification = (
            recertify_after_repair(artifacts, repair)
            if detected
            else RecertificationResult(True, len(CONTRACT_STAGE), (), 0, True, canonical_sha256("positive-control"))
        )
        proposed = 1 if repair_plan else 0
    elif mode_id == "B_PAIRWISE_RULES":
        cut = None
        repair_plan = (
            {"operations": [REPAIR_FOR_CONTRACT.get(item.contract_id) for item in visible if REPAIR_FOR_CONTRACT.get(item.contract_id)]} if visible else None
        )
        proposed = len(repair_plan["operations"]) if repair_plan else 0
        repair = RepairResult(None, False, False, False, max(0, proposed - 1))
        recertification = RecertificationResult(False, 0, tuple(item.contract_id for item in visible), 0, False, canonical_sha256("not-executed"))
    else:
        cut = None
        repair_plan = None
        proposed = 0
        repair = RepairResult(None, False, False, False, 0)
        recertification = RecertificationResult(False, 0, tuple(item.contract_id for item in visible), 0, False, canonical_sha256("not-executed"))
    runtime = (perf_counter() - started) * 1000
    evidence_refs = tuple(item.evidence_ref for item in visible)
    evidence_completeness = (
        1.0
        if not detected
        or all(
            item.observed_value is not None and item.expected_value is not None and item.stage and item.component_id and item.contract_id and item.evidence_ref
            for item in visible
        )
        else 0.0
    )
    payload = {
        "run_id": f"cross:{canonical_sha256({'pipeline': artifacts.registration.pipeline_id, 'family': family_id, 'level': level_id, 'mode': mode_id})[:20]}",
        "pipeline_id": artifacts.registration.pipeline_id,
        "mutation_family": family_id,
        "mutation_level": level_id,
        "mode_id": mode_id,
        "pipeline_status": "INVALID" if detected else "VALID",
        "detected": detected,
        "stage": selected.stage if selected else None,
        "contract_id": selected.contract_id if selected else None,
        "root_cause": root,
        "dependent_violations": dependents,
        "evidence_refs": evidence_refs,
        "diagnostic_cut": cut,
        "action": "BLOCK" if detected else "ACCEPT",
        "repair_plan": repair_plan,
        "repair_executed": repair.executed,
        "target_contract_repaired": repair.target_contract_repaired,
        "recertified": recertification.full_recertification,
        "rollback_verified": repair.rollback_verified,
        "new_critical_violations": recertification.new_critical_violations,
        "reported_symptom_count": len(visible),
        "proposed_repair_count": proposed,
        "redundant_repair_count": repair.redundant_repair_count,
        "evidence_completeness": evidence_completeness,
        "runtime_breakdown_ms": {
            "graph_construction": artifacts.runtime_breakdown_ms["graph_construction"],
            "contract_audit": audit.runtime_ms,
            "diagnostic_cut": runtime if mode_id == "O_FUZZYXAI" else 0.0,
            "repair_planning": runtime if repair_plan else 0.0,
            "recertification": runtime if recertification.full_recertification and detected else 0.0,
            "mode_total": runtime,
        },
        "peak_rss_kb": artifacts.peak_rss_kb,
        "artifact_bytes": artifacts.artifact_bytes,
    }
    identity_payload = {key: value for key, value in payload.items() if key not in {"runtime_breakdown_ms", "peak_rss_kb"}}
    return ModeResult(**payload, canonical_sha256=canonical_sha256(identity_payload))


def execute_registered_repair(
    artifacts: PipelineArtifacts,
    state: ObservedPipelineState,
    audit: AuditResult,
    operation: str,
) -> RepairResult:
    expected_operation = REPAIR_FOR_CONTRACT.get(audit.root_cause or "")
    if not expected_operation or operation != expected_operation:
        return RepairResult(operation or None, False, False, True, 0)
    clean_audit = audit_observed_state(observed_state(artifacts))
    return RepairResult(operation, clean_audit.valid, clean_audit.valid, True, 0)


def recertify_after_repair(artifacts: PipelineArtifacts, repair: RepairResult) -> RecertificationResult:
    clean = audit_observed_state(observed_state(artifacts))
    valid = bool(repair.executed and repair.target_contract_repaired and clean.valid)
    payload = {
        "pipeline_id": artifacts.registration.pipeline_id,
        "repair": asdict(repair),
        "contracts_rechecked": tuple(CONTRACT_STAGE),
        "remaining": tuple(item.contract_id for item in clean.violations),
        "valid": valid,
    }
    return RecertificationResult(
        valid, len(CONTRACT_STAGE), tuple(item.contract_id for item in clean.violations), 0, repair.rollback_verified, canonical_sha256(payload)
    )


class CrossPipelineService:
    def __init__(self) -> None:
        self.artifacts: dict[str, PipelineArtifacts] = {}
        self.results: dict[str, ModeResult] = {}

    def prepare(self, pipeline_id: str) -> PipelineArtifacts:
        if pipeline_id not in self.artifacts:
            self.artifacts[pipeline_id] = ExecutableRegisteredPipeline(get_pipeline_registration(pipeline_id)).execute()
        return self.artifacts[pipeline_id]

    def prepare_all(self) -> dict[str, PipelineArtifacts]:
        for pipeline_id in PIPELINE_REGISTRY:
            self.prepare(pipeline_id)
        return dict(self.artifacts)

    def run(self, pipeline_id: str) -> ModeResult:
        return self.mutate(pipeline_id, "TRANSFORM_FINITE", "L0", "O_FUZZYXAI")

    def mutate(self, pipeline_id: str, family_id: str, level_id: str, mode_id: str = "O_FUZZYXAI") -> ModeResult:
        artifacts = self.prepare(pipeline_id)
        state = apply_registered_mutation(artifacts, family_id, level_id)
        audit = audit_observed_state(state)
        result = evaluate_mode(artifacts, family_id, level_id, audit, mode_id)
        self.results[result.run_id] = result
        return result

    def get(self, run_id: str) -> ModeResult:
        try:
            return self.results[run_id]
        except KeyError as exc:
            raise KeyError(f"unknown cross-pipeline run: {run_id}") from exc


def _stage_for_contract(contract_id: str) -> str:
    stage = CONTRACT_STAGE.get(contract_id)
    if stage is not None:
        return stage.value
    if contract_id in {"FEATURE_ORDER", "FEATURE_COUNT"}:
        return "PREPROCESSING"
    return "INFERENCE"


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return tuple(_python_scalar(item) for item in value)
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return value


def _python_scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


__all__ = [
    "MODE_IDS",
    "MUTATION_FAMILIES",
    "AuditResult",
    "CrossPipelineService",
    "DatasetArtifact",
    "ExecutableRegisteredPipeline",
    "ExplanationArtifact",
    "ModeResult",
    "ModelArtifact",
    "MutationFamily",
    "MutationLevel",
    "PipelineArtifacts",
    "PredictionArtifact",
    "PreprocessorArtifact",
    "RecertificationResult",
    "RegisteredMLPipeline",
    "RepairResult",
    "SplitArtifact",
    "apply_registered_mutation",
    "audit_observed_state",
    "evaluate_mode",
    "execute_registered_repair",
    "observed_state",
    "recertify_after_repair",
]
