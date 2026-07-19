from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from fuzzyxai import FuzzyXAI
from fuzzyxai.adapter_conformance import run_adapter_conformance
from fuzzyxai.adapters import (
    SklearnEnsembleAdapter,
    SklearnKNNAdapter,
    SklearnLinearAdapter,
    SklearnNaiveBayesAdapter,
    SklearnPipelineAdapter,
    SklearnSVMAdapter,
    SklearnTreeAdapter,
)
from fuzzyxai.adapters.contracts_v2 import EvidenceChannelDescriptor, ModelCapabilities
from fuzzyxai.planner import ExplanationPlanner


@pytest.fixture(scope="module")
def classification_data() -> tuple[pd.DataFrame, np.ndarray]:
    values, labels = make_classification(
        n_samples=140,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        random_state=42,
    )
    return pd.DataFrame(values, columns=[f"domain_feature_{index}" for index in range(values.shape[1])]), labels


@pytest.mark.parametrize(
    ("factory", "adapter_type"),
    [
        (lambda: LogisticRegression(max_iter=300, random_state=42), SklearnLinearAdapter),
        (lambda: RidgeClassifier(), SklearnLinearAdapter),
        (lambda: DecisionTreeClassifier(max_depth=4, random_state=42), SklearnTreeAdapter),
        (lambda: RandomForestClassifier(n_estimators=8, max_depth=4, random_state=42), SklearnEnsembleAdapter),
        (lambda: ExtraTreesClassifier(n_estimators=8, max_depth=4, random_state=42), SklearnEnsembleAdapter),
        (lambda: GradientBoostingClassifier(n_estimators=8, max_depth=2, random_state=42), SklearnEnsembleAdapter),
        (lambda: LinearSVC(random_state=42), SklearnSVMAdapter),
        (lambda: SVC(kernel="rbf", probability=True, random_state=42), SklearnSVMAdapter),
        (lambda: KNeighborsClassifier(n_neighbors=5), SklearnKNNAdapter),
        (lambda: GaussianNB(), SklearnNaiveBayesAdapter),
    ],
)
def test_sklearn_family_resolution_and_explanation(classification_data, factory, adapter_type) -> None:
    frame, labels = classification_data
    model = factory().fit(frame, labels)
    fx = FuzzyXAI.wrap(model, task="classification")
    assert isinstance(fx.model_adapter, adapter_type)
    result = fx.explain_one(
        frame.iloc[0].tolist(),
        object_id="case-0",
        feature_names=list(frame.columns),
        reference_data=frame.iloc[:30],
        reference_labels=labels[:30].tolist(),
    )
    assert result.prediction.predictions is not None
    assert result.explanation_graph.validate_reachability() == ()
    assert result.explain_for().decision.explanation
    assert result.capability_report()["resolution"]["selected_adapter"] == fx.model_adapter.adapter_id
    assert result.quality_report().status in {"pass", "partial", "insufficient_evidence"}


def test_decision_score_is_not_labeled_probability(classification_data) -> None:
    frame, labels = classification_data
    model = LinearSVC(random_state=42).fit(frame, labels)
    result = FuzzyXAI.wrap(model).explain_one(frame.iloc[0].tolist(), feature_names=list(frame.columns))
    assert result.prediction.probabilities is None
    assert result.prediction.metadata["score_semantics"] == "uncalibrated_decision_score"


def test_nonlinear_svm_surrogate_has_measured_fidelity(classification_data) -> None:
    frame, labels = classification_data
    fx = FuzzyXAI.wrap(SVC(kernel="rbf", probability=True, random_state=42).fit(frame, labels))
    internal = fx.model_adapter.extract_internal_evidence(frame.iloc[:1])
    descriptor = internal["evidence_descriptors"][0]
    assert descriptor["origin"] == "surrogate"
    assert descriptor["fidelity_status"] == "measured"
    assert internal["surrogate_fidelity"] is not None


def test_pipeline_restores_feature_provenance(classification_data) -> None:
    frame, labels = classification_data
    pipeline = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=300, random_state=42))]).fit(frame, labels)
    fx = FuzzyXAI.wrap(pipeline)
    assert isinstance(fx.model_adapter, SklearnPipelineAdapter)
    schema = fx.capability_report()["input_schema"]
    assert schema["feature_names"] == tuple(frame.columns)
    assert set(schema["transformed_feature_names"]) == set(frame.columns)
    result = fx.explain_one(frame.iloc[:1], feature_names=list(frame.columns))
    assert set(result.view_model.model["contributions"]) == set(frame.columns)


@pytest.mark.parametrize(
    "model",
    [
        LinearRegression(),
        DecisionTreeRegressor(max_depth=4, random_state=42),
        RandomForestRegressor(n_estimators=8, max_depth=4, random_state=42),
    ],
)
def test_regression_contract(model) -> None:
    values, targets = make_regression(n_samples=120, n_features=5, noise=0.1, random_state=42)
    model.fit(values, targets)
    fx = FuzzyXAI.wrap(model, task="regression")
    result = fx.explain_one(values[0], feature_names=[f"x{index}" for index in range(values.shape[1])])
    assert result.prediction.probabilities is None
    assert result.prediction.metadata["task_type"] == "regression"
    assert result.explanation_graph.validate_reachability() == ()


def test_batch_global_why_not_and_model_comparison(classification_data) -> None:
    frame, labels = classification_data
    linear = LogisticRegression(max_iter=300, random_state=42).fit(frame, labels)
    tree = DecisionTreeClassifier(max_depth=4, random_state=42).fit(frame, labels)
    fx = FuzzyXAI.wrap(linear)
    batch = fx.explain_batch(frame.iloc[:3], feature_names=list(frame.columns))
    assert len(batch.view_model.trace["object_ids"]) == 3
    global_result = fx.explain_global(frame.iloc[:20], labels[:20], feature_names=list(frame.columns))
    assert global_result.sample_count == 20
    why_not = batch.why_not(1 - int(batch.prediction.predictions[0]))
    assert why_not.status == "supported"
    comparison = FuzzyXAI.compare_models(
        {"linear": linear, "tree": tree},
        item=frame.iloc[0].tolist(),
        reference_data=frame.iloc[:20],
        reference_labels=labels[:20].tolist(),
        feature_names=list(frame.columns),
    )
    assert set(comparison.model_results) == {"linear", "tree"}


def test_conformance_and_planner_block_low_fidelity(classification_data) -> None:
    frame, labels = classification_data
    fx = FuzzyXAI.wrap(LogisticRegression(max_iter=300, random_state=42).fit(frame, labels))
    report = run_adapter_conformance(fx.model_adapter, sample_batch=frame.iloc[:2])
    assert report.status == "pass", report.errors
    capabilities = ModelCapabilities(
        channels=(
            EvidenceChannelDescriptor(
                "local_contributions",
                True,
                "surrogate",
                "local surrogate",
                fidelity_status="measured",
                fidelity=0.72,
            ),
        )
    )
    decision = ExplanationPlanner().plan(capabilities)
    assert "local_contributions" not in decision.selected_channels
    assert "below" in decision.skipped_channels["local_contributions"]


def test_optional_framework_import_does_not_load_heavy_libraries() -> None:
    assert "torch" not in sys.modules
    assert "tensorflow" not in sys.modules
    assert "onnxruntime" not in sys.modules
