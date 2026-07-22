"""Quality audit for conditional rule samplers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ConditionalSamplerAudit:
    mean_standardized_shift: float
    covariance_relative_error: float
    nearest_reference_distance_p95: float
    support_violation_rate: float
    passed: bool


def audit_conditional_sampler(
    reference: np.ndarray,
    generated: np.ndarray,
    *,
    support_minimum: np.ndarray | None = None,
    support_maximum: np.ndarray | None = None,
    mean_shift_maximum: float = 0.20,
    covariance_error_maximum: float = 0.30,
    support_violation_maximum: float = 0.02,
) -> ConditionalSamplerAudit:
    source = np.asarray(reference, dtype=float)
    sample = np.asarray(generated, dtype=float)
    if source.ndim != 2 or sample.ndim != 2 or source.shape[1] != sample.shape[1]:
        raise ValueError("reference and generated samples must be aligned matrices")
    scales = source.std(axis=0)
    scales[scales < 1e-9] = 1.0
    shift = float(np.mean(np.abs((sample.mean(axis=0) - source.mean(axis=0)) / scales)))
    source_cov = np.cov(source, rowvar=False)
    sample_cov = np.cov(sample, rowvar=False)
    covariance_error = float(np.linalg.norm(sample_cov - source_cov) / max(np.linalg.norm(source_cov), 1e-9))
    distances = []
    for batch_start in range(0, len(sample), 256):
        batch = sample[batch_start : batch_start + 256]
        squared = np.sum((batch[:, None, :] - source[None, :, :]) ** 2, axis=2)
        distances.extend(np.sqrt(np.min(squared, axis=1)).tolist())
    minimum = source.min(axis=0) if support_minimum is None else np.asarray(support_minimum)
    maximum = source.max(axis=0) if support_maximum is None else np.asarray(support_maximum)
    violations = np.any((sample < minimum) | (sample > maximum), axis=1)
    violation_rate = float(np.mean(violations))
    passed = (
        shift <= mean_shift_maximum
        and covariance_error <= covariance_error_maximum
        and violation_rate <= support_violation_maximum
    )
    return ConditionalSamplerAudit(shift, covariance_error, float(np.quantile(distances, 0.95)), violation_rate, passed)
