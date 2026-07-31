from __future__ import annotations

from dataclasses import asdict

from fastapi.testclient import TestClient
from fuzzyxai.ml_vertical.api import app, get_pipeline_service
from fuzzyxai.ml_vertical.pipeline import ALL_SCENARIOS, MLPipelineService


def test_python_api_executes_all_eighteen_scenarios(pipeline_service: MLPipelineService) -> None:
    runs = [pipeline_service.execute_scenario(scenario_id) for scenario_id in ALL_SCENARIOS]

    assert len(runs) == 18
    assert all(run.observations for run in runs)
    assert all(run.route_graph["nodes"] and run.route_graph["edges"] for run in runs)
    assert all(run.views.keys() == {"user", "engineering", "audit"} for run in runs)
    assert all(asdict(run)["canonical_sha256"] for run in runs)


def test_rest_executes_all_scenarios_and_exposes_canonical_projections() -> None:
    get_pipeline_service.cache_clear()
    client = TestClient(app)

    for scenario_id in ALL_SCENARIOS:
        response = client.post(f"/api/v1/pipeline/scenario/{scenario_id}", json={"scenario_id": scenario_id})
        assert response.status_code == 200
        payload = response.json()
        run_id = payload["run_id"]
        assert client.get(f"/api/v1/pipeline/run/{run_id}").json()["canonical_sha256"] == payload["canonical_sha256"]
        assert client.get(f"/api/v1/pipeline/run/{run_id}/graph").json()["trace_sha256"] == payload["route_graph"]["trace_sha256"]
        assert client.get(f"/api/v1/pipeline/run/{run_id}/diagnosis").json() == payload["diagnosis"]
        if payload["repair_plan"]:
            repaired = client.post(f"/api/v1/pipeline/run/{run_id}/repair")
            assert repaired.status_code == 200
            assert repaired.json()["recertification"]["full_recertification"]


def test_route_graph_contains_required_pipeline_nodes_and_relations(pipeline_service: MLPipelineService) -> None:
    graph = pipeline_service.execute_scenario("S13_PREPROCESSOR_FULL_FIT").route_graph
    node_ids = {node["node_id"] for node in graph["nodes"]}
    relations = {edge["relation"] for edge in graph["edges"]}
    required_fields = {
        "stage",
        "component_id",
        "component_type",
        "version",
        "input_schema",
        "output_schema",
        "configuration",
        "artifact_sha256",
        "evidence_refs",
        "execution_status",
    }

    assert {
        "dataset",
        "split_manifest",
        "train_partition",
        "validation_partition",
        "test_partition",
        "preprocessor",
        "transformed_train",
        "training_configuration",
        "trained_model",
        "model_artifact",
        "inference_object",
        "prediction",
        "shap_explainer",
        "shap_values",
        "fuzzy_representation",
        "reduced_explanation",
        "user_view",
        "engineering_view",
        "audit_view",
        "repair_plan",
        "recertification_result",
    }.issubset(node_ids)
    assert {"derived_from", "fitted_on", "trained_on", "transformed_by", "produced_by", "explained_by", "represented_by", "reduced_to", "displayed_as", "recertified_as"}.issubset(relations)
    assert all(required_fields.issubset(node) for node in graph["nodes"])
