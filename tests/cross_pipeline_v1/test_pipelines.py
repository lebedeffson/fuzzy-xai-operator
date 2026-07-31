from __future__ import annotations

import numpy as np
import pytest
from fuzzyxai.pipelines.practical import MUTATION_FAMILIES, ExecutableRegisteredPipeline
from fuzzyxai.pipelines.registry import PIPELINE_REGISTRY


def test_five_real_pipelines_are_registered() -> None:
    assert len(PIPELINE_REGISTRY) == 5


@pytest.mark.parametrize("pipeline_id", tuple(PIPELINE_REGISTRY))
def test_each_pipeline_really_trains_and_predicts(service, pipeline_id: str) -> None:
    artifacts = service.prepare(pipeline_id)
    assert artifacts.model.artifact_bytes
    assert artifacts.model.estimator.n_features_in_ == len(artifacts.preprocessor.feature_names)
    assert np.isfinite(artifacts.prediction.output)


@pytest.mark.parametrize("pipeline_id", tuple(PIPELINE_REGISTRY))
def test_each_pipeline_builds_real_consistent_explanation(service, pipeline_id: str) -> None:
    explanation = service.prepare(pipeline_id).explanation
    assert explanation.attributions
    assert explanation.absolute_error <= 1e-6
    assert explanation.explainer_version


def test_same_contract_runs_across_binary_and_multiclass(service) -> None:
    for pipeline_id in ("breast-cancer-logreg-linearshap", "wine-logreg-linearshap"):
        result = service.mutate(pipeline_id, "FIT_SCOPE", "L1")
        assert result.contract_id == "PREPROCESSOR_FIT_SCOPE"


def test_linear_shap_consistency_supports_classification_and_regression(service) -> None:
    for pipeline_id in ("wine-logreg-linearshap", "diabetes-ridge-linearshap"):
        assert service.prepare(pipeline_id).explanation.absolute_error <= 1e-6


def test_treeshap_adapter_has_unified_vector(service) -> None:
    artifacts = service.prepare("digits-random-forest-treeshap")
    assert len(artifacts.explanation.attributions) == len(artifacts.preprocessor.feature_names)
    assert artifacts.explanation.selected_output is not None


def test_mixed_pipeline_expands_feature_schema(service) -> None:
    artifacts = service.prepare("mixed-logreg-linearshap")
    assert artifacts.dataset.features.shape[1] == 6
    assert len(artifacts.preprocessor.feature_names) > 6


def test_all_mutations_are_preregistered_factor_levels() -> None:
    assert len(MUTATION_FAMILIES) == 8
    assert all(tuple(level.level_id for level in family.levels) == ("L0", "L1", "L2", "L3", "L4") for family in MUTATION_FAMILIES.values())


def test_positive_controls_are_not_blocked(service) -> None:
    for family_id in MUTATION_FAMILIES:
        result = service.mutate("breast-cancer-logreg-linearshap", family_id, "L0")
        assert result.pipeline_status == "VALID"
        assert result.action == "ACCEPT"


def test_unregistered_mutation_is_rejected(service) -> None:
    with pytest.raises(ValueError, match="unregistered mutation"):
        service.mutate("breast-cancer-logreg-linearshap", "UNLOCKED", "L1")


def test_pipeline_execution_is_deterministic() -> None:
    registration = PIPELINE_REGISTRY["diabetes-ridge-linearshap"]
    first = ExecutableRegisteredPipeline(registration).execute()
    second = ExecutableRegisteredPipeline(registration).execute()
    assert first.dataset.sha256 == second.dataset.sha256
    assert first.model.sha256 == second.model.sha256
    assert first.explanation.sha256 == second.explanation.sha256
