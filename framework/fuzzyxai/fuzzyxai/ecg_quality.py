"""Transparent, technical ECG signal-quality gates for chapter 5.

This is not a diagnostic classifier. The score and thresholds are engineering
controls, not trained or clinically calibrated estimates of disease risk.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import median
from typing import Any, Iterable


@dataclass(frozen=True)
class ECGQualityThresholds:
    min_duration_sec: float = 1.0
    severe_missing_fraction: float = 0.10
    severe_marked_noise_fraction: float = 0.20
    severe_flatline_fraction: float = 0.30
    severe_score_below: float = 0.65
    caution_score_below: float = 0.90


DEFAULT_THRESHOLDS = ECGQualityThresholds()


def _fraction_or_error(value: Any, name: str) -> float:
    try:
        out = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(out) or not 0.0 <= out <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return out


def _positive_rate(value: Any) -> float:
    try:
        rate = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("sampling_rate must be numeric") from exc
    if not math.isfinite(rate) or not 25.0 <= rate <= 10000.0:
        raise ValueError("sampling_rate must be in [25, 10000] Hz")
    return rate


def _count(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a nonnegative integer")
    try:
        f = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{name} must be a nonnegative integer") from exc
    if not math.isfinite(f) or f < 0 or int(f) != f:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(f)


def _mask(values: Iterable[Any] | None, n: int, name: str) -> list[bool]:
    if values is None:
        return [False] * n
    items = list(values)
    if len(items) != n:
        raise ValueError(f"{name} length must equal signal length")
    result: list[bool] = []
    for value in items:
        if value in (1, '1', True):
            result.append(True)
        elif value in (0, '0', False, '', None):
            result.append(False)
        else:
            raise ValueError(f"{name} values must be binary")
    return result


def _run_count(mask: list[bool]) -> int:
    return sum(value and (i == 0 or not mask[i - 1]) for i, value in enumerate(mask))


def extract_ecg_quality(
    samples: Iterable[Any],
    sampling_rate: Any,
    *,
    noise_mask: Iterable[Any] | None = None,
    missing_mask: Iterable[Any] | None = None,
) -> dict[str, Any]:
    """Derive measurements from raw samples, never from a supplied quality score.

    Blanks/None are missing. Non-finite nonblank values are rejected, not coerced
    to silence. Noise markings are external annotations, NOT inferred pathology.
    """
    rate = _positive_rate(sampling_rate)
    input_samples = list(samples)
    n = len(input_samples)
    if n == 0:
        raise ValueError("ECG signal cannot be empty")
    noise = _mask(noise_mask, n, "noise_mask")
    missing = _mask(missing_mask, n, "missing_mask")
    clean_values: list[float | None] = []
    for i, value in enumerate(input_samples):
        if missing[i] or value is None or value == '':
            missing[i] = True
            clean_values.append(None)
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"invalid ECG value at sample {i}") from exc
        if not math.isfinite(numeric):
            raise ValueError(f"non-finite ECG value at sample {i}")
        clean_values.append(numeric)
    valid = [v for v in clean_values if v is not None]
    missing_samples = sum(missing)
    noise_samples = sum(noise)
    if valid:
        value_range = max(valid) - min(valid)
        jumps = [abs(b - a) for a, b in zip(clean_values, clean_values[1:]) if a is not None and b is not None]
        instability = float(median(jumps) / max(value_range, 1e-12)) if jumps else 0.0
        flat_adjacent = sum(abs(a - b) <= 1e-10 for a, b in zip(clean_values, clean_values[1:]) if a is not None and b is not None)
        flat_fraction = flat_adjacent / max(n - 1, 1)
    else:
        value_range, instability, flat_fraction = 0.0, 0.0, 1.0
    missing_fraction = missing_samples / n
    marked_noise_fraction = noise_samples / n
    score = max(0.0, min(1.0, 1.0 - 1.4 * missing_fraction - 0.8 * marked_noise_fraction - 0.65 * flat_fraction))
    return {
        'sampling_rate': rate,
        'n_samples': n,
        'duration_sec': round(n / rate, 6),
        'missing_samples': missing_samples,
        'missing_fragments': missing_samples,  # legacy API name counts samples
        'missing_runs': _run_count(missing),
        'missing_fraction': round(missing_fraction, 6),
        'noise_samples': noise_samples,
        'noise_segments': _run_count(noise),
        'marked_noise_fraction': round(marked_noise_fraction, 6),
        'noise_mask_available': noise_mask is not None,
        'flatline_fraction': round(flat_fraction, 6),
        'baseline_instability': round(instability, 6),
        'peak_to_peak': round(value_range, 6),
        'quality_score': round(score, 6),
        'quality_score_kind': 'uncalibrated_engineering_heuristic',
        'quality_features_computed': True,
    }


def assess_ecg_quality(payload: dict[str, Any], thresholds: ECGQualityThresholds = DEFAULT_THRESHOLDS) -> dict[str, Any]:
    """Select *technical audit action*, never a clinical diagnosis or accept."""
    score = _fraction_or_error(payload.get('quality_score'), 'quality_score')
    rate = _positive_rate(payload.get('sampling_rate'))
    missing_samples = _count(payload.get('missing_fragments'), 'missing_fragments')
    noise_segments = _count(payload.get('noise_segments'), 'noise_segments')
    instability = _fraction_or_error(payload.get('baseline_instability'), 'baseline_instability')
    missing_fraction = _fraction_or_error(payload.get('missing_fraction', 0.0), 'missing_fraction')
    noise_fraction = _fraction_or_error(payload.get('marked_noise_fraction', 0.0), 'marked_noise_fraction')
    flat_fraction = _fraction_or_error(payload.get('flatline_fraction', 0.0), 'flatline_fraction')
    n_samples = payload.get('n_samples')
    if n_samples is not None:
        n_samples = _count(n_samples, 'n_samples')
        if n_samples == 0 or missing_samples > n_samples:
            raise ValueError('inconsistent ECG sample counts')
        observed_fraction = missing_samples / n_samples
        if 'missing_fraction' in payload and abs(observed_fraction - missing_fraction) > 0.00001:
            raise ValueError('inconsistent missing_fraction and missing_fragments')
        missing_fraction = observed_fraction
    reasons: list[str] = []
    severe = False
    if score < thresholds.severe_score_below:
        severe = True
        reasons.append('low_quality_score')
    if missing_fraction >= thresholds.severe_missing_fraction:
        severe = True
        reasons.append('missing_signal')
    if noise_fraction >= thresholds.severe_marked_noise_fraction:
        severe = True
        reasons.append('marked_noise')
    if flat_fraction >= thresholds.severe_flatline_fraction:
        severe = True
        reasons.append('flatline')
    if n_samples is not None and n_samples / rate < thresholds.min_duration_sec:
        severe = True
        reasons.append('insufficient_duration')
    if severe:
        action, diagnostic = 'defer_to_human', 'D_signal_quality'
    elif missing_samples or noise_segments or score < thresholds.caution_score_below or instability > 0.20:
        action, diagnostic = 'request_more_data', 'D_signal_quality_warning'
        reasons.append('quality_requires_review')
    else:
        action, diagnostic = 'audit_report', ''
    return {
        'action': action,
        'diagnostic_id': diagnostic,
        'quality_reasons': sorted(set(reasons)),
        'clinical_diagnosis': False,
        'automatic_clinical_accept_allowed': False,
        'policy_version': 'technical_ecg_quality_v1',
    }
