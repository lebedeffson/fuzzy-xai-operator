from __future__ import annotations

from fastapi.testclient import TestClient
from fuzzyxai.ml_vertical.api import app, get_service


def test_health_ready_and_scenarios() -> None:
    client = TestClient(app)
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/ready").json()["status"] == "ready"
    assert len(client.get("/scenarios").json()["scenarios"]) == 10


def test_full_public_api_route() -> None:
    client = TestClient(app)
    body = {"scenario_id": "S1_NORMAL", "object_id": "api:test"}
    explanation = client.post("/explain", json=body)
    assert explanation.status_code == 200
    run = explanation.json()
    assert run["observer"]["action"] == "ACCEPT"
    assert client.post("/predict", json=body).status_code == 200
    assert client.post("/diagnose", json=body).status_code == 200
    assert client.post("/repair/plan", json={"scenario_id": "S2_EXPLAINER_VERSION_MISMATCH"}).status_code == 200
    repaired = client.post("/repair/execute", json=body).json()
    assert repaired["repair"]["recertification"]["status"] == "full_success"
    assert client.post("/recertify", json=body).status_code == 200
    run_id = run["run_id"]
    for audience in ("user", "engineer", "auditor"):
        assert client.get(f"/runs/{run_id}/views/{audience}").status_code == 200


def test_api_rejects_gold_channel() -> None:
    client = TestClient(app)
    response = client.post("/explain", json={"scenario_id": "S1_NORMAL", "controls": {"gold_patch": "x"}})
    assert response.status_code == 422


def teardown_module() -> None:
    get_service.cache_clear()
