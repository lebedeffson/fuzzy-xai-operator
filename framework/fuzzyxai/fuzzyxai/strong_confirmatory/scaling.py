"""Streaming operator-layer scalability without retaining per-object artifacts."""

from __future__ import annotations

import hashlib
import resource
from collections.abc import Sequence
from time import perf_counter

import numpy as np


def run_streaming_scalability(
    *,
    sizes: Sequence[int] = (1_000, 10_000, 100_000, 500_000, 1_000_000),
    batch_size: int = 10_000,
    seed: int = 4201,
) -> dict[str, object]:
    if not sizes or any(size <= 0 for size in sizes) or batch_size <= 0:
        raise ValueError("scalability sizes and batch size must be positive")
    rows = [_measure(int(size), batch_size=batch_size, seed=seed) for size in sizes]
    exponent = float(np.polyfit(np.log([row["n_objects"] for row in rows]), np.log([row["wall_time_seconds"] for row in rows]), 1)[0])
    repeat = _measure(int(sizes[0]), batch_size=batch_size, seed=seed)
    return {
        "phase": "formative_measurement",
        "mode": "streaming_cached_operator_layer",
        "local_explainer_included": False,
        "measurements": rows,
        "empirical_scaling_exponent": exponent,
        "deterministic_repeat": repeat["result_sha256"] == rows[0]["result_sha256"],
        "formative_target_met": exponent <= 1.10 and repeat["result_sha256"] == rows[0]["result_sha256"],
        "confirmatory_claim_allowed": False,
    }


def _measure(size: int, *, batch_size: int, seed: int) -> dict[str, object]:
    started = perf_counter()
    latencies = []
    result = hashlib.sha256()
    processed = 0
    while processed < size:
        count = min(batch_size, size - processed)
        batch_started = perf_counter()
        rng = np.random.default_rng(seed + processed)
        evidence = rng.normal(size=(count, 12)).astype(np.float32)
        confidence = 1.0 / (1.0 + np.exp(-evidence[:, 0]))
        discrepancy = np.max(np.abs(evidence[:, 1:5]), axis=1) / 5.0
        risk = np.clip(0.45 * (1.0 - confidence) + 0.35 * discrepancy + 0.20 * (evidence[:, 5] > 1.5), 0.0, 1.0)
        actions = np.select((risk < 0.30, risk < 0.55, risk < 0.80), (0, 1, 2), default=3).astype(np.uint8)
        result.update(actions.tobytes())
        latencies.append(perf_counter() - batch_started)
        processed += count
    wall = perf_counter() - started
    latency_ms = np.asarray(latencies) * 1000.0
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "n_objects": size,
        "batch_size": batch_size,
        "wall_time_seconds": wall,
        "objects_per_second": size / max(wall, 1e-12),
        "batch_latency_ms_p50": float(np.quantile(latency_ms, 0.50)),
        "batch_latency_ms_p95": float(np.quantile(latency_ms, 0.95)),
        "batch_latency_ms_p99": float(np.quantile(latency_ms, 0.99)),
        "peak_rss_bytes": int(peak_kib * 1024),
        "result_sha256": result.hexdigest(),
    }
