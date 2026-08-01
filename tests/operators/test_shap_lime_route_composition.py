import pytest
from fuzzyxai.operators import (
    ExplanationArtifact,
    OperatorExecutionError,
    route_explanation_artifacts,
)


def _artifact(method: str, **changes: str) -> ExplanationArtifact:
    values = {
        "artifact_id": f"{method}-1",
        "method": method,
        "model_version": "model-7",
        "preprocessing_version": "prep-3",
        "sample_id": "sample-42",
        "provenance_ref": f"registry://{method}-1",
    }
    values.update(changes)
    return ExplanationArtifact(**values)


def test_shap_lime_share_route_without_combining_values() -> None:
    state = route_explanation_artifacts(_artifact("SHAP"), _artifact("LIME"))

    assert state.values["sample_id"] == "sample-42"
    assert state.values["diagnostic_route_kind"] == "parallel_explanation_artifacts"
    assert "combined_attribution" not in state.values


@pytest.mark.parametrize(
    "changes",
    [
        {"model_version": "model-8"},
        {"preprocessing_version": "prep-4"},
        {"sample_id": "sample-99"},
        {"provenance_ref": None},
    ],
)
def test_shap_lime_reject_incompatible_artifacts(changes: dict[str, str]) -> None:
    with pytest.raises(OperatorExecutionError) as error:
        route_explanation_artifacts(_artifact("SHAP"), _artifact("LIME", **changes))

    assert error.value.code == "OPERATOR_CONTRACT_MISMATCH"
