"""Separated framework-overhead and local-explainer scalability measurements."""

from __future__ import annotations

import json
import tempfile
import tracemalloc
from pathlib import Path
from time import perf_counter, process_time
from typing import Callable, Sequence

import numpy as np


STAGES = (
    "load_data",
    "preprocessing",
    "prediction",
    "local_explanation",
    "evidence_normalization",
    "graph_construction",
    "uncertainty_selection",
    "diagnostics",
    "audience_reduction",
    "user_text",
    "json_export",
)


def run_end_to_end_scalability(
    output: Path,
    *,
    sizes: Sequence[int] = (1_000, 5_000, 10_000, 50_000, 100_000),
    seed: int = 4201,
) -> dict[str, object]:
    rows = [_measure_size(int(size), seed=seed) for size in sizes]
    api_modes = _measure_public_api(seed)
    payload = {
        "schema_version": "2.0",
        "measurements": rows,
        "public_api_modes": api_modes,
        "stage_order": list(STAGES),
        "cost_separation": {
            "framework_overhead": [
                "evidence_normalization",
                "graph_construction",
                "uncertainty_selection",
                "diagnostics",
                "audience_reduction",
                "user_text",
                "json_export",
            ],
            "local_explainer_cost": ["local_explanation"],
            "model_cost": ["prediction"],
        },
        "gpu": {"used": False, "time_seconds": None, "peak_memory_bytes": None},
        "claim_scope": "measured reference pipeline overhead; explainer and model costs are reported separately",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _measure_size(size: int, *, seed: int) -> dict[str, object]:
    rng = np.random.default_rng(seed + size)
    stage_times: dict[str, float] = {}
    cpu_started = process_time()
    wall_started = perf_counter()
    tracemalloc.start()

    values = _stage(stage_times, "load_data", lambda: rng.normal(size=(size, 20)).astype(np.float32))
    normalized = _stage(
        stage_times,
        "preprocessing",
        lambda: (values - values.mean(axis=0)) / np.maximum(values.std(axis=0), 1e-6),
    )
    weights = rng.normal(size=20)
    scores = _stage(stage_times, "prediction", lambda: 1.0 / (1.0 + np.exp(-(normalized @ weights))))
    contributions = _stage(stage_times, "local_explanation", lambda: normalized * weights)
    evidence = _stage(
        stage_times,
        "evidence_normalization",
        lambda: {
            "object_ids": np.arange(size, dtype=np.int32),
            "confidence": np.maximum(scores, 1.0 - scores),
            "top_feature": np.argmax(np.abs(contributions), axis=1).astype(np.int16),
        },
    )
    graph = _stage(
        stage_times,
        "graph_construction",
        lambda: {
            "nodes": 5 * size,
            "edges": 4 * size,
            "claim_ids": np.arange(size, dtype=np.int32),
        },
    )
    representation = _stage(
        stage_times,
        "uncertainty_selection",
        lambda: np.select(
            (evidence["confidence"] >= 0.85, evidence["confidence"] >= 0.70, evidence["confidence"] >= 0.55),
            (0, 1, 2),
            default=3,
        ).astype(np.int8),
    )
    diagnostics = _stage(
        stage_times,
        "diagnostics",
        lambda: (evidence["confidence"] < 0.60) | (np.max(np.abs(normalized), axis=1) > 4.0),
    )
    reasons = _stage(
        stage_times,
        "audience_reduction",
        lambda: np.argpartition(np.abs(contributions), -3, axis=1)[:, -3:].astype(np.int16),
    )
    text = _stage(
        stage_times,
        "user_text",
        lambda: [
            f"object={index}; decision={'review' if diagnostics[index] else 'accept'}; top={reasons[index].tolist()}"
            for index in range(size)
        ],
    )
    serialized = _stage(
        stage_times,
        "json_export",
        lambda: json.dumps(
            {
                "n_objects": size,
                "graph_nodes": graph["nodes"],
                "graph_edges": graph["edges"],
                "representation_counts": np.bincount(representation, minlength=4).tolist(),
                "diagnostics": int(diagnostics.sum()),
                "text": text,
            },
            separators=(",", ":"),
        ),
    )
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    wall = perf_counter() - wall_started
    cpu = process_time() - cpu_started
    framework_stages = STAGES[4:]
    return {
        "n_objects": size,
        "wall_time_seconds": wall,
        "cpu_time_seconds": cpu,
        "peak_ram_bytes": int(peak),
        "model_calls": 1,
        "graph_nodes": graph["nodes"],
        "graph_edges": graph["edges"],
        "serialized_bytes": len(serialized.encode("utf-8")),
        "stage_seconds": stage_times,
        "prediction_seconds": stage_times["prediction"],
        "local_explainer_seconds": stage_times["local_explanation"],
        "framework_overhead_seconds": float(sum(stage_times[name] for name in framework_stages)),
    }


def _measure_public_api(seed: int) -> list[dict[str, object]]:
    from sklearn.linear_model import LogisticRegression

    from fuzzyxai import FuzzyXAI

    rng = np.random.default_rng(seed)
    train = rng.normal(size=(500, 8))
    labels = (train[:, 0] + train[:, 1] > 0).astype(int)
    model = LogisticRegression(max_iter=200, random_state=seed).fit(train, labels)
    framework = FuzzyXAI.wrap(model)
    modes = []
    for name, count in (("single", 1), ("batch_100", 100), ("batch_1000", 1000)):
        values = rng.normal(size=(count, 8))
        started = perf_counter()
        result = framework.explain(values, reference_data=train, include_similar_cases=False)
        with tempfile.TemporaryDirectory() as directory:
            json_path = result.export_json(Path(directory) / "explanation.json")
            serialized_bytes = json_path.stat().st_size
        modes.append(
            {
                "mode": name,
                "n_objects": count,
                "wall_time_seconds": perf_counter() - started,
                "serialized_bytes": serialized_bytes,
                "explanation_level": result.explanation_level,
                "action": result.action,
            }
        )
    global_started = perf_counter()
    global_result = framework.explain_global(rng.normal(size=(1000, 8)))
    modes.append(
        {
            "mode": "global_explanation",
            "n_objects": 1000,
            "wall_time_seconds": perf_counter() - global_started,
            "serialized_bytes": len(json.dumps(global_result.to_dict(), default=str)),
        }
    )
    compare_started = perf_counter()
    comparison = FuzzyXAI.compare_models(
        {"model_a": model, "model_b": model},
        item=rng.normal(size=8),
        reference_data=train,
    )
    modes.append(
        {
            "mode": "model_comparison",
            "n_objects": 100,
            "wall_time_seconds": perf_counter() - compare_started,
            "serialized_bytes": len(json.dumps(comparison.to_dict(), default=str)),
        }
    )
    return modes


def _stage(times: dict[str, float], name: str, operation: Callable[[], object]) -> object:
    started = perf_counter()
    result = operation()
    times[name] = perf_counter() - started
    return result
