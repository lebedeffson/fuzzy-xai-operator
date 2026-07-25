from __future__ import annotations

import csv
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import shap
import torch
from sklearn.datasets import load_diabetes
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression, LogisticRegression

from fuzzyxai.diagnostics.contracts import canonical_sha256

BATCH_SIZES = (1, 8, 32)
WARMUPS = 20
REPETITIONS = 30


@dataclass
class Pipeline:
    pipeline_id: str
    model: object
    inputs: object
    predict: object
    explain: object


class TinyImageModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(9102)
        self.conv = torch.nn.Conv2d(3, 4, 3, padding=1)
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.head = torch.nn.Linear(4, 2)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(torch.relu(self.conv(values))).flatten(1))


def _pipelines() -> tuple[Pipeline, ...]:
    diabetes = load_diabetes()
    tabular_model = LinearRegression().fit(diabetes.data, diabetes.target)
    tabular_explainer = shap.LinearExplainer(tabular_model, diabetes.data[:64])

    def tabular_predict(values: np.ndarray) -> np.ndarray:
        return tabular_model.predict(values)

    def tabular_explain(values: np.ndarray) -> np.ndarray:
        return np.asarray(tabular_explainer(values).values)

    image_model = TinyImageModel().eval()
    image_inputs = torch.linspace(-1, 1, 32 * 3 * 16 * 16).reshape(32, 3, 16, 16)

    def image_predict(values: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return image_model(values)

    def image_explain(values: torch.Tensor) -> np.ndarray:
        local = values.detach().clone().requires_grad_(True)
        activations = torch.relu(image_model.conv(local))
        activations.retain_grad()
        score = image_model.head(image_model.pool(activations).flatten(1)).max(dim=1).values.sum()
        score.backward()
        weights = activations.grad.mean(dim=(2, 3), keepdim=True)
        heatmap = torch.relu((weights * activations).sum(dim=1))
        return heatmap.detach().numpy()

    corpus = (
        "registered model artifact and compatible preprocessing",
        "missing provenance for explanation artifact",
        "schema mismatch in model input",
        "valid route with calibrated explanation",
    ) * 16
    labels = np.asarray([1, 0, 0, 1] * 16)
    vectorizer = CountVectorizer().fit(corpus)
    matrix = vectorizer.transform(corpus)
    text_model = LogisticRegression(random_state=9103).fit(matrix, labels)
    text_inputs = tuple(corpus[:32])

    def text_predict(values: tuple[str, ...]) -> np.ndarray:
        return text_model.predict_proba(vectorizer.transform(values))

    def text_explain(values: tuple[str, ...]) -> np.ndarray:
        base = text_model.predict_proba(vectorizer.transform(values))[:, 1]
        outputs: list[list[float]] = []
        for sentence, original in zip(values, base, strict=True):
            tokens = sentence.split()
            row = []
            for index in range(len(tokens)):
                masked = " ".join(token for position, token in enumerate(tokens) if position != index)
                changed = text_model.predict_proba(vectorizer.transform([masked]))[0, 1]
                row.append(float(original - changed))
            outputs.append(row)
        width = max(map(len, outputs))
        return np.asarray([row + [0.0] * (width - len(row)) for row in outputs])

    return (
        Pipeline("tabular_shap", tabular_model, diabetes.data[:32], tabular_predict, tabular_explain),
        Pipeline("image_gradcam", image_model, image_inputs, image_predict, image_explain),
        Pipeline("text_token_masking", text_model, text_inputs, text_predict, text_explain),
    )


def _slice(values: object, size: int) -> object:
    return values[:size]


def _measure(pipeline: Pipeline, batch_size: int, serialization: bool) -> dict[str, float]:
    values = _slice(pipeline.inputs, batch_size)
    start = time.perf_counter_ns()
    predictions = pipeline.predict(values)
    after_model = time.perf_counter_ns()
    explanations = pipeline.explain(values)
    after_explainer = time.perf_counter_ns()
    evidence = {
        "pipeline": pipeline.pipeline_id,
        "batch_size": batch_size,
        "prediction_shape": tuple(np.asarray(predictions).shape),
        "explanation_shape": tuple(np.asarray(explanations).shape),
        "prediction_sha256": canonical_sha256(np.asarray(predictions).round(10).tolist()),
        "explanation_sha256": canonical_sha256(np.asarray(explanations).round(10).tolist()),
        "status": "supported",
    }
    _ = canonical_sha256(evidence)
    after_fuzzyxai = time.perf_counter_ns()
    if serialization:
        json.dumps(evidence, sort_keys=True, separators=(",", ":"))
    after_serialization = time.perf_counter_ns()
    divisor = 1_000_000
    return {
        "model_ms": (after_model - start) / divisor,
        "explainer_ms": (after_explainer - after_model) / divisor,
        "fuzzyxai_ms": (after_fuzzyxai - after_explainer) / divisor,
        "serialization_ms": (after_serialization - after_fuzzyxai) / divisor,
        "total_ms": (after_serialization - start) / divisor,
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(root: Path) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for pipeline in _pipelines():
        for batch_size in BATCH_SIZES:
            for _ in range(WARMUPS):
                _measure(pipeline, batch_size, False)
            for serialization in (False, True):
                for repetition in range(REPETITIONS):
                    measured = _measure(pipeline, batch_size, serialization)
                    base_ms = measured["model_ms"] + measured["explainer_ms"]
                    overhead = (
                        measured["fuzzyxai_ms"] + measured["serialization_ms"]
                    ) / max(base_ms, 1e-12)
                    rows.append(
                        {
                            "pipeline": pipeline.pipeline_id,
                            "batch_size": batch_size,
                            "cache": "warm",
                            "serialization": serialization,
                            "repetition": repetition,
                            **measured,
                            "overhead_ratio": overhead,
                        }
                    )
            # Cold measurements include one new pipeline construction per recorded run.
            for serialization in (False, True):
                for repetition in range(REPETITIONS):
                    cold_start = time.perf_counter_ns()
                    cold = next(item for item in _pipelines() if item.pipeline_id == pipeline.pipeline_id)
                    setup_ms = (time.perf_counter_ns() - cold_start) / 1_000_000
                    measured = _measure(cold, batch_size, serialization)
                    measured["model_ms"] += setup_ms
                    measured["total_ms"] += setup_ms
                    base_ms = measured["model_ms"] + measured["explainer_ms"]
                    overhead = (
                        measured["fuzzyxai_ms"] + measured["serialization_ms"]
                    ) / max(base_ms, 1e-12)
                    rows.append(
                        {
                            "pipeline": pipeline.pipeline_id,
                            "batch_size": batch_size,
                            "cache": "cold",
                            "serialization": serialization,
                            "repetition": repetition,
                            **measured,
                            "overhead_ratio": overhead,
                        }
                    )
    warm = [float(row["overhead_ratio"]) for row in rows if row["cache"] == "warm"]
    ordered = sorted(warm)
    p95 = ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]
    result = {
        "protocol_id": "h9-e2e-latency-v1",
        "status": (
            "H9_E2E_TARGET_MET"
            if statistics.median(warm) <= 0.10 and p95 <= 0.15
            else "H9_E2E_TARGET_NOT_MET"
        ),
        "pipelines": 3,
        "warmup_runs_per_pipeline_batch": WARMUPS,
        "measured_repetitions": REPETITIONS,
        "measurements": len(rows),
        "warm_median_overhead_ratio": statistics.median(warm),
        "warm_p95_overhead_ratio": p95,
        "human_time_claim": False,
    }
    output = root / "results/h9_e2e_latency"
    _write_csv(output / "E2E_LATENCY.csv", rows)
    (output / "H9_E2E_FINAL_STATUS.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = root / "reports/h9_e2e_latency/E2E_LATENCY_REPORT.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "# H9 End-to-End Latency\n\n"
        + "\n".join(f"- {key}: `{value}`" for key, value in result.items())
        + "\n\nMeasured time is machine execution time, not engineer work time.\n",
        encoding="utf-8",
    )
    return result
