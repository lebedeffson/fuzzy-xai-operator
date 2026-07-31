from __future__ import annotations

from dataclasses import asdict
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .practical import MODE_IDS, MUTATION_FAMILIES, CrossPipelineService
from .registry import PIPELINE_REGISTRY, list_pipeline_registrations


class MutationRequest(BaseModel):
    level_id: str = "L1"
    mode_id: str = "O_FUZZYXAI"


@lru_cache(maxsize=1)
def get_service() -> CrossPipelineService:
    return CrossPipelineService()


app = FastAPI(title="FuzzyXAI cross-pipeline practical API", version="1.0.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/v1/pipelines")
def pipelines() -> dict[str, object]:
    return {"pipelines": [asdict(item) | {"sha256": item.sha256} for item in list_pipeline_registrations()]}


@app.post("/api/v1/pipelines/{pipeline_id}/run")
def run_pipeline(pipeline_id: str) -> dict[str, object]:
    _require_pipeline(pipeline_id)
    return asdict(get_service().run(pipeline_id))


@app.post("/api/v1/pipelines/{pipeline_id}/mutate/{mutation_id}")
def mutate_pipeline(pipeline_id: str, mutation_id: str, body: MutationRequest | None = None) -> dict[str, object]:
    _require_pipeline(pipeline_id)
    if mutation_id not in MUTATION_FAMILIES:
        raise HTTPException(404, f"unknown registered mutation: {mutation_id}")
    request = body or MutationRequest()
    if request.mode_id not in MODE_IDS:
        raise HTTPException(422, f"unknown registered mode: {request.mode_id}")
    try:
        return asdict(get_service().mutate(pipeline_id, mutation_id, request.level_id, request.mode_id))
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get("/api/v1/runs/{run_id}/diagnosis")
def diagnosis(run_id: str) -> dict[str, object]:
    return _get_result(run_id)


@app.post("/api/v1/runs/{run_id}/repair")
def repair(run_id: str) -> dict[str, object]:
    result = _get_result(run_id)
    return {
        "run_id": run_id,
        "repair_plan": result["repair_plan"],
        "repair_executed": result["repair_executed"],
        "target_contract_repaired": result["target_contract_repaired"],
        "rollback_verified": result["rollback_verified"],
    }


@app.post("/api/v1/runs/{run_id}/recertify")
def recertify(run_id: str) -> dict[str, object]:
    result = _get_result(run_id)
    return {
        "run_id": run_id,
        "recertified": result["recertified"],
        "new_critical_violations": result["new_critical_violations"],
        "rollback_verified": result["rollback_verified"],
    }


def _get_result(run_id: str) -> dict[str, object]:
    try:
        return asdict(get_service().get(run_id))
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc


def _require_pipeline(pipeline_id: str) -> None:
    if pipeline_id not in PIPELINE_REGISTRY:
        raise HTTPException(404, f"unknown registered pipeline: {pipeline_id}")


__all__ = ["app", "get_service"]
