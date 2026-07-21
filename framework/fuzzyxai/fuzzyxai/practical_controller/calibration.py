"""Development-only calibration selection for practical predictive risk."""

from __future__ import annotations

import numpy as np


METHODS = ("platt", "isotonic", "temperature", "conformal_selective")


def compare_calibrators(scores: np.ndarray, labels: np.ndarray, *, seed: int = 4201) -> dict[str, object]:
    scores, labels = _validate(scores, labels)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(scores))
    split = max(2, int(0.7 * len(scores)))
    train, validation = order[:split], order[split:]
    if len(validation) < 2:
        raise ValueError("calibration comparison requires a validation subset")
    rows = []
    for method in METHODS:
        parameters = fit_calibrator(method, scores[train], labels[train])
        calibrated = apply_calibrator(method, parameters, scores[validation])
        rows.append(
            {
                "method": method,
                "brier_score": float(np.mean((calibrated - labels[validation]) ** 2)),
                "calibration_error": _ece(calibrated, labels[validation]),
                "parameters": list(parameters),
            }
        )
    selected = min(rows, key=lambda row: (row["brier_score"], row["calibration_error"], row["method"]))
    final_parameters = fit_calibrator(str(selected["method"]), scores, labels)
    return {
        "selection_partition": "development_only",
        "candidates": rows,
        "selected_method": selected["method"],
        "selected_parameters": list(final_parameters),
        "confirmatory_test_used": False,
    }


def fit_calibrator(method: str, scores: np.ndarray, labels: np.ndarray) -> tuple[float, ...]:
    scores, labels = _validate(scores, labels)
    clipped = np.clip(scores, 1e-6, 1.0 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped))
    if method == "platt":
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(max_iter=1000, random_state=4201).fit(logits.reshape(-1, 1), labels)
        return float(model.coef_[0, 0]), float(model.intercept_[0])
    if method == "isotonic":
        from sklearn.isotonic import IsotonicRegression

        model = IsotonicRegression(out_of_bounds="clip").fit(clipped, labels)
        return tuple(float(value) for pair in zip(model.X_thresholds_, model.y_thresholds_, strict=True) for value in pair)
    if method == "temperature":
        candidates = np.linspace(0.5, 3.0, 101)
        losses = [np.mean((_sigmoid(logits / temperature) - labels) ** 2) for temperature in candidates]
        return (float(candidates[int(np.argmin(losses))]),)
    if method == "conformal_selective":
        return tuple(float(value) for value in np.sort(clipped))
    raise ValueError(f"unknown calibration method: {method}")


def apply_calibrator(method: str, parameters: tuple[float, ...], scores: np.ndarray | float) -> np.ndarray:
    values = np.asarray(scores, dtype=float)
    clipped = np.clip(values, 1e-6, 1.0 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped))
    if method == "platt":
        return _sigmoid(parameters[0] * logits + parameters[1])
    if method == "isotonic":
        x = np.asarray(parameters[0::2], dtype=float)
        y = np.asarray(parameters[1::2], dtype=float)
        return np.interp(clipped, x, y)
    if method == "temperature":
        return _sigmoid(logits / parameters[0])
    if method == "conformal_selective":
        reference = np.asarray(parameters, dtype=float)
        return np.searchsorted(reference, clipped, side="right") / max(1, len(reference))
    raise ValueError(f"unknown calibration method: {method}")


def _validate(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if scores.ndim != 1 or labels.ndim != 1 or len(scores) != len(labels) or len(scores) < 10:
        raise ValueError("calibration inputs must be aligned vectors with at least ten observations")
    if np.any((scores < 0.0) | (scores > 1.0)) or set(np.unique(labels)) != {0, 1}:
        raise ValueError("calibration requires probabilities and binary labels")
    return scores, labels


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))


def _ece(probabilities: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        selected = (probabilities >= lower) & (probabilities <= upper if upper == 1.0 else probabilities < upper)
        if selected.any():
            result += float(selected.mean()) * abs(float(probabilities[selected].mean()) - float(labels[selected].mean()))
    return result
