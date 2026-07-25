from __future__ import annotations

import csv
import hashlib
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

from fuzzyxai.evidence_path import (
    StaticArtifactCache,
    StaticManifest,
    audit_batch,
)
from fuzzyxai.experiments.h9_e2e_latency import (
    BATCH_SIZES,
    REPETITIONS,
    WARMUPS,
    _pipelines,
    _slice,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _manifest(pipeline_id: str) -> StaticManifest:
    return StaticManifest(
        pipeline_id,
        "v2",
        _digest(f"{pipeline_id}:model"),
        _digest(f"{pipeline_id}:explainer"),
        _digest(f"{pipeline_id}:preprocessing"),
        _digest(f"{pipeline_id}:schema"),
        _digest(f"{pipeline_id}:route"),
        _digest("contract-registry:v2"),
    )


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_once(pipeline: object, batch_size: int, mode: str, cache: StaticArtifactCache) -> dict[str, object]:
    values = _slice(pipeline.inputs, batch_size)
    started = time.perf_counter_ns()
    predictions = pipeline.predict(values)
    model_done = time.perf_counter_ns()
    explanations = pipeline.explain(values)
    explainer_done = time.perf_counter_ns()
    prediction_array = np.asarray(predictions)
    explanation_array = np.asarray(explanations)
    sample_ids = tuple(
        f"{pipeline.pipeline_id}:{batch_size}:{index}"
        for index in range(batch_size)
    )
    report = audit_batch(
        prediction_array,
        explanation_array,
        _manifest(pipeline.pipeline_id),
        sample_ids,
        cache=cache,
        mode=mode,
    )
    completed = time.perf_counter_ns()
    model_ms = (model_done - started) / 1_000_000
    explainer_ms = (explainer_done - model_done) / 1_000_000
    audit_ms = (completed - explainer_done) / 1_000_000
    base_ms = model_ms + explainer_ms
    return {
        "pipeline": pipeline.pipeline_id,
        "batch_size": batch_size,
        "mode": mode,
        "model_ms": model_ms,
        "explainer_ms": explainer_ms,
        "base_ms": base_ms,
        "fuzzyxai_ms": audit_ms,
        "fuzzyxai_ms_per_object": audit_ms / batch_size,
        "relative_overhead": audit_ms / max(base_ms, 1e-12),
        "root_digest": report.merkle_root,
        **report.timings_ms,
    }


def run(root: Path) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for pipeline in _pipelines():
        cache = StaticArtifactCache()
        for batch_size in BATCH_SIZES:
            for mode in ("online", "full"):
                for _ in range(WARMUPS):
                    _run_once(pipeline, batch_size, mode, cache)
                for repetition in range(REPETITIONS):
                    row = _run_once(pipeline, batch_size, mode, cache)
                    rows.append({**row, "repetition": repetition})
    summaries: list[dict[str, object]] = []
    for pipeline_id, batch_size, mode in sorted(
        {
            (str(row["pipeline"]), int(row["batch_size"]), str(row["mode"]))
            for row in rows
        }
    ):
        selected = [
            row
            for row in rows
            if (row["pipeline"], row["batch_size"], row["mode"])
            == (pipeline_id, batch_size, mode)
        ]
        absolute = sorted(float(row["fuzzyxai_ms_per_object"]) for row in selected)
        relative = sorted(float(row["relative_overhead"]) for row in selected)
        bases = sorted(float(row["base_ms"]) for row in selected)
        summaries.append(
            {
                "pipeline": pipeline_id,
                "batch_size": batch_size,
                "mode": mode,
                "median_base_ms": statistics.median(bases),
                "median_fuzzyxai_ms_per_object": statistics.median(absolute),
                "p95_fuzzyxai_ms_per_object": absolute[min(len(absolute) - 1, int(0.95 * len(absolute)))],
                "median_relative_overhead": statistics.median(relative),
                "p95_relative_overhead": relative[min(len(relative) - 1, int(0.95 * len(relative)))],
                "relative_gate_applicable": statistics.median(bases) >= 1.0,
            }
        )
    online = [row for row in summaries if row["mode"] == "online"]
    absolute_pass = all(
        float(row["median_fuzzyxai_ms_per_object"]) <= 0.10
        and float(row["p95_fuzzyxai_ms_per_object"]) <= 0.50
        for row in online
    )
    applicable = [row for row in summaries if row["relative_gate_applicable"]]
    relative_pass = all(
        float(row["median_relative_overhead"]) <= 0.10
        and float(row["p95_relative_overhead"]) <= 0.15
        for row in applicable
    )
    status = (
        "H9_E2E_V2_TARGET_MET"
        if absolute_pass and relative_pass
        else "H9_E2E_V2_TARGET_NOT_MET"
    )
    result = {
        "protocol_id": "h9-e2e-v2-optimized-evidence-path",
        "status": status,
        "measurements": len(rows),
        "online_absolute_gate": absolute_pass,
        "relative_gate": relative_pass,
        "relative_groups": len(applicable),
        "worst_online_median_ms_per_object": max(
            float(row["median_fuzzyxai_ms_per_object"]) for row in online
        ),
        "worst_online_p95_ms_per_object": max(
            float(row["p95_fuzzyxai_ms_per_object"]) for row in online
        ),
        "worst_applicable_median_relative_overhead": max(
            (float(row["median_relative_overhead"]) for row in applicable),
            default=0.0,
        ),
        "worst_applicable_p95_relative_overhead": max(
            (float(row["p95_relative_overhead"]) for row in applicable),
            default=0.0,
        ),
        "human_time_claim": False,
        "benchmark_scope": "registered_local_microbenchmark_pipelines",
        "parent_result": "H9_E2E_TARGET_NOT_MET",
        "parent_result_modified": False,
    }
    output = root / "results/h9_e2e_v2"
    per_run_path = output / "PER_RUN_TIMES.csv"
    summary_path = output / "PIPELINE_SUMMARY.csv"
    status_path = output / "H9_E2E_V2_FINAL_STATUS.json"
    _write_csv(per_run_path, rows)
    _write_csv(summary_path, summaries)
    status_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    environment_path = output / "ENVIRONMENT.json"
    environment_path.write_text(
        json.dumps(
            {
                "python": sys.version,
                "platform": platform.platform(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "numpy": np.__version__,
                "warmups_per_group": WARMUPS,
                "repetitions_per_group": REPETITIONS,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    report = root / "reports/h9_e2e_v2/OPTIMIZED_EVIDENCE_PATH.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "# H9 E2E v2 Optimized Evidence Path\n\n"
        + "\n".join(f"- {key}: `{value}`" for key, value in result.items())
        + "\n\nOnline and archival modes are reported separately. This is a "
        "registered local microbenchmark; human time and industrial latency "
        "are out of scope.\n",
        encoding="utf-8",
    )
    checksum_paths = (per_run_path, summary_path, status_path, environment_path, report)
    (output / "SHA256SUMS").write_text(
        "".join(
            f"{_sha256(path)}  {path.relative_to(root).as_posix()}\n"
            for path in checksum_paths
        ),
        encoding="utf-8",
    )
    return result
