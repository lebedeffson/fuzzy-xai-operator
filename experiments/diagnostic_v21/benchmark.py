from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

from fuzzyxai.diagnostics import (
    ActionableRepairPlanner,
    DiagnosticReporter,
    DiagnosticValidator,
    MinimalDiagnosticCutFinder,
    RepairCostModel,
    RouteGraphBuilder,
)


PIPELINES = (
    ("tabular_tree", "tabular"),
    ("tabular_boosting", "tabular"),
    ("text_transformer", "text"),
    ("text_encoder", "text"),
    ("image_cnn", "image"),
    ("time_series_encoder", "time_series"),
)


def route_fixture(pipeline_id: str, modality: str, index: int) -> dict[str, object]:
    observed_version = "v2" if index % 3 else "v1"
    return {
        "route_id": f"{pipeline_id}:{index}",
        "nodes": [
            {
                "node_id": "preprocessor",
                "node_type": "preprocessing",
                "registered_attributes": {"version": "v1", "schema": f"{modality}:schema:v1"},
                "observed_attributes": {"version": observed_version, "schema": f"{modality}:schema:v1"},
                "evidence_refs": [f"manifest:{pipeline_id}:preprocessor"],
            },
            {
                "node_id": "model",
                "node_type": "model",
                "registered_attributes": {"version": "v4"},
                "observed_attributes": {"version": "v4"},
                "evidence_refs": [f"manifest:{pipeline_id}:model"],
            },
            {
                "node_id": "explainer",
                "node_type": "explainer",
                "registered_attributes": {"model_version": "v4"},
                "observed_attributes": {"model_version": "v4"},
                "evidence_refs": [f"manifest:{pipeline_id}:explainer"],
            },
        ],
        "edges": [
            {
                "edge_id": "preprocessor-to-model",
                "source": "preprocessor",
                "target": "model",
                "relation": "transforms",
                "registered_contract": {"compatible": True},
                "observed_contract": {"compatible": True},
            },
            {
                "edge_id": "model-to-explainer",
                "source": "model",
                "target": "explainer",
                "relation": "explains",
                "registered_contract": {"compatible": True},
                "observed_contract": {"compatible": True},
            },
        ],
        "metadata": {"pipeline_id": pipeline_id, "modality": modality},
    }


def run(*, repetitions: int = 200, output: str | Path = "reports/diagnostic_v21/performance.json") -> dict:
    builder = RouteGraphBuilder()
    validator = DiagnosticValidator()
    finder = MinimalDiagnosticCutFinder()
    planner = ActionableRepairPlanner()
    reporter = DiagnosticReporter()
    timings = {stage: [] for stage in ("graph", "validation", "cut", "plan", "report", "total")}
    processed = 0
    for pipeline_id, modality in PIPELINES:
        for index in range(repetitions):
            total_start = time.perf_counter()
            start = time.perf_counter()
            graph = builder.build(route_fixture(pipeline_id, modality, index))
            timings["graph"].append((time.perf_counter() - start) * 1000)
            start = time.perf_counter()
            validation = validator.validate(graph)
            timings["validation"].append((time.perf_counter() - start) * 1000)
            cut = None
            plan = None
            start = time.perf_counter()
            if validation.issues:
                cut = finder.find(graph, validation, RepairCostModel())
            timings["cut"].append((time.perf_counter() - start) * 1000)
            start = time.perf_counter()
            if cut:
                plan = planner.plan(graph, validation.issues, cut)
            timings["plan"].append((time.perf_counter() - start) * 1000)
            start = time.perf_counter()
            reporter.build(graph, validation, validation.issues, cut, plan, None)
            timings["report"].append((time.perf_counter() - start) * 1000)
            timings["total"].append((time.perf_counter() - total_start) * 1000)
            processed += 1

    def summary(values: list[float]) -> dict[str, float]:
        ordered = sorted(values)
        return {
            "median_ms": statistics.median(ordered),
            "mean_ms": statistics.mean(ordered),
            "p95_ms": ordered[int(0.95 * (len(ordered) - 1))],
            "p99_ms": ordered[int(0.99 * (len(ordered) - 1))],
            "max_ms": max(ordered),
        }

    payload = {
        "status": "PASS" if summary(timings["total"])["p95_ms"] < 1000 else "FAIL",
        "scope": "diagnostic operator only; excludes model, explainer, external repair, and human review",
        "pipelines": [pipeline for pipeline, _ in PIPELINES],
        "route_count": processed,
        "repetitions_per_pipeline": repetitions,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor() or "not_reported_by_platform",
        },
        "timings": {stage: summary(values) for stage, values in timings.items()},
    }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=200)
    parser.add_argument("--output", default="reports/diagnostic_v21/performance.json")
    args = parser.parse_args()
    result = run(repetitions=args.repetitions, output=args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
