from __future__ import annotations

from dataclasses import asdict
from typing import Any

from fuzzyxai.diagnostics.contracts import canonical_sha256

from .practical import ModeResult, PipelineArtifacts

REQUIRED_ARTIFACTS = (
    "pipeline_manifest.json",
    "dataset_manifest.json",
    "split_manifest.json",
    "preprocessor_manifest.json",
    "model_manifest.json",
    "explanation_manifest.json",
    "route_graph.json",
    "contract_report.json",
    "diagnosis.json",
    "repair_plan.json",
    "recertification.json",
    "canonical_result.json",
)


def log_practical_run(
    result: ModeResult,
    artifacts: PipelineArtifacts,
    *,
    tracking_uri: str,
    git_commit: str,
    scoring_metrics: dict[str, float | bool],
    experiment_name: str = "fuzzyxai-cross-pipeline-practical-v1",
) -> dict[str, Any]:
    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError("MLflow logging requires the 'mlflow' optional dependency") from exc

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    payload = asdict(result)
    manifests = artifact_payloads(result, artifacts)
    with mlflow.start_run(run_name=f"{result.pipeline_id}-{result.mutation_family}-{result.mutation_level}-{result.mode_id}") as active:
        mlflow.log_params(
            {
                "pipeline_id": result.pipeline_id,
                "task_type": artifacts.registration.task_type,
                "mutation_family": result.mutation_family,
                "mutation_level": result.mutation_level,
                "mode_id": result.mode_id,
                "dataset_sha256": artifacts.dataset.sha256,
                "split_sha256": artifacts.split.sha256,
                "preprocessor_sha256": artifacts.preprocessor.sha256,
                "model_sha256": artifacts.model.sha256,
                "explainer_id": artifacts.explanation.explainer_id,
                "explainer_version": artifacts.explanation.explainer_version,
                "git_commit": git_commit,
            }
        )
        mlflow.log_metrics(
            {
                "detected": float(result.detected),
                "stage_correct": float(scoring_metrics["stage_correct"]),
                "contract_correct": float(scoring_metrics["contract_correct"]),
                "root_cause_correct": float(scoring_metrics["root_cause_correct"]),
                "false_certification": float(scoring_metrics["false_certification"]),
                "evidence_completeness": result.evidence_completeness,
                "cut_size": float((result.diagnostic_cut or {}).get("size", 0)),
                "repair_success": float(result.target_contract_repaired),
                "recertification_success": float(result.recertified),
                "new_critical_violations": float(result.new_critical_violations),
                "runtime_ms": result.runtime_breakdown_ms["mode_total"],
            }
        )
        for name, artifact_payload in manifests.items():
            mlflow.log_dict(artifact_payload, name)
        return {
            "run_id": active.info.run_id,
            "artifact_count": len(manifests),
            "result_sha256": canonical_sha256(payload),
        }


def artifact_payloads(result: ModeResult, artifacts: PipelineArtifacts) -> dict[str, dict[str, Any]]:
    result_payload = asdict(result)
    diagnosis = {
        key: result_payload[key]
        for key in (
            "pipeline_status",
            "stage",
            "contract_id",
            "root_cause",
            "dependent_violations",
            "evidence_refs",
            "action",
        )
    }
    return {
        "pipeline_manifest.json": asdict(artifacts.registration) | {"registration_sha256": artifacts.registration.sha256},
        "dataset_manifest.json": artifacts.dataset.manifest(),
        "split_manifest.json": artifacts.split.manifest(),
        "preprocessor_manifest.json": artifacts.preprocessor.manifest(),
        "model_manifest.json": artifacts.model.manifest(),
        "explanation_manifest.json": artifacts.explanation.manifest(),
        "route_graph.json": artifacts.route_graph.to_dict(),
        "contract_report.json": {"contract_count": len(artifacts.registration.supported_contracts), "contracts": artifacts.registration.supported_contracts},
        "diagnosis.json": diagnosis,
        "repair_plan.json": result.repair_plan or {"available": False},
        "recertification.json": {
            "recertified": result.recertified,
            "new_critical_violations": result.new_critical_violations,
            "rollback_verified": result.rollback_verified,
        },
        "canonical_result.json": result_payload,
    }


__all__ = ["REQUIRED_ARTIFACTS", "artifact_payloads", "log_practical_run"]
