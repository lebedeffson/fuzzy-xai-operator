"""Resource measurements and conservative scaling diagnostics."""

from __future__ import annotations

import tracemalloc
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Callable, Sequence

import numpy as np


@dataclass(frozen=True)
class ScalingMeasurement:
    n_objects: int
    elapsed_seconds: float
    peak_memory_bytes: int
    graph_nodes: int
    graph_edges: int
    serialized_bytes: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def measure_scaling(
    sizes: Sequence[int],
    operation: Callable[[int], tuple[int, int, int]],
) -> dict[str, object]:
    rows: list[ScalingMeasurement] = []
    for size in sizes:
        tracemalloc.start()
        start = perf_counter()
        nodes, edges, serialized_bytes = operation(int(size))
        elapsed = perf_counter() - start
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        rows.append(ScalingMeasurement(int(size), elapsed, int(peak), nodes, edges, serialized_bytes))
    slope, intercept, r_squared = log_log_fit(
        [row.n_objects for row in rows],
        [max(row.elapsed_seconds, 1e-12) for row in rows],
    )
    return {
        "measurements": [row.to_dict() for row in rows],
        "log_log_fit": {"slope": slope, "intercept": intercept, "r_squared": r_squared},
        "linear_scalability_claim_allowed": bool(0.8 <= slope <= 1.2 and r_squared >= 0.9),
    }


def log_log_fit(sizes: Sequence[float], measurements: Sequence[float]) -> tuple[float, float, float]:
    if len(sizes) != len(measurements) or len(sizes) < 2:
        raise ValueError("scaling fit requires at least two aligned measurements")
    x_values = np.log(np.asarray(sizes, dtype=float))
    y_values = np.log(np.asarray(measurements, dtype=float))
    design = np.column_stack((x_values, np.ones_like(x_values)))
    slope, intercept = np.linalg.lstsq(design, y_values, rcond=None)[0]
    fitted = slope * x_values + intercept
    residual = float(np.sum((y_values - fitted) ** 2))
    total = float(np.sum((y_values - y_values.mean()) ** 2))
    r_squared = 1.0 - residual / total if total else 1.0
    return float(slope), float(intercept), r_squared
