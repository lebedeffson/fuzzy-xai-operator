from __future__ import annotations

from dataclasses import asdict

from fastapi.testclient import TestClient
from fuzzyxai.pipelines.practical_api import app, get_service
from fuzzyxai.pipelines.practical_tracking import REQUIRED_ARTIFACTS, artifact_payloads

from apps.cross_pipeline_practical import ui_projection

PIPELINE = "wine-logreg-linearshap"


def test_rest_returns_registered_pipelines() -> None:
    response = TestClient(app).get("/api/v1/pipelines")
    assert response.status_code == 200
    assert len(response.json()["pipelines"]) == 5


def test_rest_returns_canonical_diagnosis() -> None:
    client = TestClient(app)
    response = client.post(f"/api/v1/pipelines/{PIPELINE}/mutate/FIT_SCOPE", json={"level_id": "L1"})
    assert response.status_code == 200
    payload = response.json()
    diagnosis = client.get(f"/api/v1/runs/{payload['run_id']}/diagnosis")
    assert diagnosis.json()["canonical_sha256"] == payload["canonical_sha256"]


def test_rest_repair_and_recertification_views() -> None:
    client = TestClient(app)
    result = client.post(f"/api/v1/pipelines/{PIPELINE}/mutate/MODEL_ARTIFACT", json={"level_id": "L1"}).json()
    repair = client.post(f"/api/v1/runs/{result['run_id']}/repair").json()
    recertification = client.post(f"/api/v1/runs/{result['run_id']}/recertify").json()
    assert repair["repair_executed"] and repair["rollback_verified"]
    assert recertification["recertified"] and recertification["new_critical_violations"] == 0


def test_ui_projects_api_fields_without_new_diagnosis(service) -> None:
    payload = asdict(service.mutate(PIPELINE, "SPLIT_OVERLAP", "L1"))
    projected = ui_projection(payload)
    assert projected["root_cause"] == payload["root_cause"]
    assert projected["contract_id"] == payload["contract_id"]


def test_mlflow_payload_has_complete_required_artifacts(service) -> None:
    result = service.mutate(PIPELINE, "EXPLAINER_BINDING", "L1")
    manifests = artifact_payloads(result, service.prepare(PIPELINE))
    assert tuple(manifests) == REQUIRED_ARTIFACTS


def test_route_graph_records_component_versions_and_hashes(service) -> None:
    graph = service.prepare(PIPELINE).route_graph.to_dict()
    assert graph["metadata"]["model_sha256"]
    assert graph["metadata"]["explainer_version"]
    assert all(node["evidence_refs"] for node in graph["nodes"])


def test_all_modes_receive_same_registered_mutation(service) -> None:
    contracts = {service.mutate(PIPELINE, "SPLIT_OVERLAP", "L3", mode).contract_id for mode in ("B_PAIRWISE_RULES", "B_GREEDY_CROSS_STAGE", "O_FUZZYXAI")}
    assert contracts == {"TRAIN_VALIDATION_TEST_DISJOINTNESS"}


def test_api_rejects_unregistered_mutation() -> None:
    response = TestClient(app).post(f"/api/v1/pipelines/{PIPELINE}/mutate/UNREGISTERED")
    assert response.status_code == 404


def test_api_and_python_service_share_canonical_result() -> None:
    get_service.cache_clear()
    client = TestClient(app)
    api_result = client.post(f"/api/v1/pipelines/{PIPELINE}/mutate/SHAP_CONSISTENCY", json={"level_id": "L2"}).json()
    python_result = get_service().mutate(PIPELINE, "SHAP_CONSISTENCY", "L2")
    assert api_result["canonical_sha256"] == python_result.canonical_sha256
