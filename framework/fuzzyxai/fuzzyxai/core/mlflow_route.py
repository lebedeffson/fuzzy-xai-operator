from __future__ import annotations

from dataclasses import replace

from fuzzyxai.core.external_tabular_route import (
    build_external_wine_classification_route,
)
from fuzzyxai.core.types import (
    AdaptedInput,
    OperatorEdge,
    OperatorNode,
    OperatorRoute,
)


def build_mlflow_tabular_route(
    adapted: AdaptedInput,
) -> OperatorRoute:
    base = build_external_wine_classification_route(
        replace(
            adapted,
            scenario_id="external_wine_classification",
        )
    )
    values = adapted.values
    run_id = str(values["run_id"])
    model_version = str(values["model_version"])
    artifact_uri = str(values["artifact_uri"])
    mlflow_run = OperatorNode(
        node_id="mlflow_run",
        title="MLflow run",
        title_ru="Запуск MLflow",
        operator_type="external_provenance",
        input_summary="локальное MLflow-хранилище",
        output_summary="зарегистрированные параметры и tags",
        value=run_id,
        status="passed",
        explanation="Метаданные запуска получены из локального MLflow.",
        trace_ref=f"mlflow-run:{run_id}",
        value_source="mlflow_tracking_store",
        raw={
            "run_id": run_id,
            "artifact_uri": artifact_uri,
            "mlflow_version": values["mlflow_version"],
            "params": values["mlflow_params"],
            "tags": values["mlflow_tags"],
        },
        output_refs=["mlflow_registered_model"],
        output_values={
            "run_id": run_id,
            "artifact_uri": artifact_uri,
        },
        status_reason_ru="Запуск и его метаданные зарегистрированы.",
        interpretation_ru=(
            "MLflow предоставляет внешние сведения происхождения; "
            "FuzzyXAI не пересчитывает их."
        ),
        next_node_ids=["mlflow_registered_model"],
        details={"provider": "MLflow"},
    )
    registered_model = OperatorNode(
        node_id="mlflow_registered_model",
        title="MLflow registered model",
        title_ru="Зарегистрированная модель MLflow",
        operator_type="model_registry",
        input_summary="MLflow run",
        output_summary="модель и версия",
        value=f"{values['model_name']}:{model_version}",
        status="passed",
        explanation="Версия модели связана с исходным MLflow run.",
        trace_ref=f"mlflow-model:{values['model_name']}:{model_version}",
        value_source="mlflow_model_registry",
        raw={
            "model_name": values["model_name"],
            "model_version": model_version,
            "run_id": run_id,
            "artifact_uri": artifact_uri,
        },
        input_refs=["mlflow_run"],
        output_refs=["input_artifact", "explanation_object"],
        input_values={"run_id": run_id},
        output_values={
            "model_version": model_version,
            "artifact_uri": artifact_uri,
        },
        status_reason_ru="Версия модели загружена из Model Registry.",
        interpretation_ru=(
            "Зарегистрированная версия передана в объяснительный маршрут."
        ),
        next_node_ids=["input_artifact", "explanation_object"],
        details={"provider": "MLflow"},
    )
    provenance_edges = [
        OperatorEdge(
            "edge_mlflow_run_model",
            "mlflow_run",
            "mlflow_registered_model",
            {
                "run_id": run_id,
                "model_version": model_version,
                "artifact_uri": artifact_uri,
            },
            "Запуск MLflow породил зарегистрированную версию модели.",
        ),
        OperatorEdge(
            "edge_mlflow_model_input",
            "mlflow_registered_model",
            "input_artifact",
            {
                "model_version": model_version,
                "artifact_uri": artifact_uri,
            },
            "Зарегистрированная модель передана внешнему адаптеру.",
        ),
        OperatorEdge(
            "edge_mlflow_model_explanation",
            "mlflow_registered_model",
            "explanation_object",
            {
                "run_id": run_id,
                "model_version": model_version,
            },
            "Объяснение связано с той же зарегистрированной моделью.",
        ),
    ]
    return replace(
        base,
        route_id=(
            f"mlflow:{run_id}:{values['model_name']}:{model_version}"
        ),
        scenario_id=adapted.scenario_id,
        scenario_title_ru="Локальная интеграция MLflow и FuzzyXAI",
        title="MLflow provenance FuzzyXAI OperatorRoute",
        nodes=[mlflow_run, registered_model, *base.nodes],
        edges=[*provenance_edges, *base.edges],
        verification_summary={
            "overall_status": "passed",
            "checks": [
                "mlflow_run_registered",
                "model_version_registered",
                "artifact_uri_registered",
                "run_model_explanation_provenance_linked",
            ],
        },
    )
