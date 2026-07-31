from __future__ import annotations

import os
from dataclasses import asdict
from functools import lru_cache
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .contracts import PredictionRequest
from .service import SCENARIOS, MLVerticalService
from .tracking import log_run


class RunRequest(BaseModel):
    scenario_id: str = "S1_NORMAL"
    object_id: str = "bcw:api"
    features: dict[str, float] | None = None
    controls: dict[str, Any] = Field(default_factory=dict)
    requested_view: str = "user"
    log_to_mlflow: bool = False


@lru_cache(maxsize=1)
def get_service() -> MLVerticalService:
    return MLVerticalService(persist_dir=os.getenv("FUZZYXAI_RUN_DIR"))


app = FastAPI(title="FuzzyXAI ML Vertical v1", version="1.0.0")


def _execute(body: RunRequest):
    service = get_service()
    if body.scenario_id not in SCENARIOS:
        raise HTTPException(404, f"unknown scenario: {body.scenario_id}")
    template = service.scenario_request(body.scenario_id)
    request = PredictionRequest(
        body.scenario_id,
        body.object_id,
        body.features if body.features is not None else template.features,
        {**template.controls, **body.controls},
        body.requested_view,
    )
    try:
        run = service.execute(request)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    result = asdict(run)
    if body.log_to_mlflow:
        result["mlflow"] = log_run(
            run,
            tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "file:./results/ml_vertical_v1/mlruns"),
        )
    return result


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict[str, Any]:
    service = get_service()
    return {"status": "ready", "model": service.model.manifest()}


@app.get("/scenarios")
def scenarios() -> dict[str, Any]:
    return {"scenarios": tuple(SCENARIOS)}


@app.post("/predict")
def predict(body: RunRequest) -> dict[str, Any]:
    run = _execute(body)
    return {"run_id": run["run_id"], "prediction": run["prediction"], "canonical_sha256": run["canonical_sha256"]}


@app.post("/explain")
def explain(body: RunRequest) -> dict[str, Any]:
    return _execute(body)


@app.post("/diagnose")
def diagnose(body: RunRequest) -> dict[str, Any]:
    run = _execute(body)
    return {"run_id": run["run_id"], "diagnosis": run["diagnosis"], "observer": run["observer"]}


@app.post("/repair/plan")
def repair_plan(body: RunRequest) -> dict[str, Any]:
    run = _execute(body)
    return {"run_id": run["run_id"], "repair_plan": run["diagnosis"].get("repair_plan")}


@app.post("/repair/execute")
def repair_execute(body: RunRequest) -> dict[str, Any]:
    payload = body.model_dump()
    payload["scenario_id"] = "S9_REGISTERED_REPAIR"
    run = _execute(RunRequest(**payload))
    return {"run_id": run["run_id"], "repair": run["repair"], "observer": run["observer"]}


@app.post("/recertify")
def recertify(body: RunRequest) -> dict[str, Any]:
    return repair_execute(body)


@app.get("/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    try:
        return asdict(get_service().get(run_id))
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.get("/runs/{run_id}/views/{audience}")
def get_view(run_id: str, audience: str) -> dict[str, Any]:
    if audience not in {"user", "engineer", "auditor"}:
        raise HTTPException(404, "audience must be user, engineer, or auditor")
    try:
        return get_service().get(run_id).views[audience]
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
