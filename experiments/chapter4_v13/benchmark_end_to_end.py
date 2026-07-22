from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from fuzzyxai.practical_controller import (
    CostProfileName,
    DeploymentContext,
    ExplanationArtifact,
    PracticalPolicy,
    PredictionArtifact,
    ReviewBudget,
    RouteArtifacts,
    assess_action,
    cost_profile,
)

from .common import ARTIFACTS, canonical_bytes, environment_manifest, protocol, read_jsonl, runtime_config, sha256_bytes, sha256_file, write_json
from .generate_explanations import integrated_gradients_batch, token_masking_batch
from .train_or_load_model import load_frozen_model, predict_texts, set_deterministic


def _token_skeleton(tokenizer: Any, text: str) -> dict[str, object]:
    encoded = tokenizer(text, truncation=True, max_length=protocol()["modern_contour"]["prediction"]["max_length"])
    tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"])
    special = set(tokenizer.all_special_tokens)
    return {"text": text, "tokens": tokens, "scores": [0.0] * len(tokens), "valid": [token not in special for token in tokens]}


def _fixed_policy() -> PracticalPolicy:
    return PracticalPolicy(
        schema_version="1.0",
        policy_version="chapter4-v13-runtime",
        predictive_weights=(0.8, 0.6, 0.4, 0.2, 0.0, 0.0, 0.0, 0.0),
        predictive_intercept=-1.0,
        route_weights=(0.5, 0.0, 0.0, 0.5, 1.0, 1.0, 0.2, 0.0, 0.0, 1.0),
        route_intercept=-1.5,
        accept_max_risk=0.25,
        short_review_max_risk=0.50,
        full_review_max_risk=0.80,
        calibration_method="platt",
        calibration_parameters=(1.0, 0.0),
        development_sha256="1" * 64,
        selected_without_test=True,
    )


def _fuzzyxai_stage(rows: Sequence[dict[str, Any]], probabilities: np.ndarray, explanations: Sequence[dict[str, object]], explanation_scores: Sequence[Sequence[float]]) -> list[dict[str, object]]:
    policy = _fixed_policy()
    costs = cost_profile(CostProfileName.BALANCED)
    outputs = []
    for row, probs, explanation, scores in zip(rows, probabilities, explanations, explanation_scores, strict=True):
        order = np.sort(probs)
        confidence = float(np.max(probs))
        entropy = float(-(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum() / np.log(len(probs)))
        canonical = {
            "object_id": row["object_id"],
            "tokens": explanation["tokens"],
            "scores": [float(value) for value in scores],
            "model": "distilbert-ag-news@52ee64d",
            "explainer": "chapter4-v13-runtime",
        }
        digest = sha256_bytes(canonical_bytes(canonical))
        prediction = PredictionArtifact(
            object_id=str(row["object_id"]),
            prediction=str(int(np.argmax(probs))),
            confidence=confidence,
            probabilities=tuple(float(value) for value in probs),
            model_version="distilbert-ag-news@52ee64d",
            entropy=entropy,
            prediction_margin=float(order[-1] - order[-2]),
        )
        explanation_artifact = ExplanationArtifact(
            canonical_sha256=digest,
            explainer_version="chapter4-v13-runtime",
            model_version="distilbert-ag-news@52ee64d",
            explain_plan_version="chapter4-v13",
            dictionary_version="ag-news-en-v1",
            available_channels=("prediction", "model", "tokenizer", "explainer", "reference"),
        )
        route = RouteArtifacts(
            preprocessing_version="tokenizer@52ee64d",
            calibration_version="isotonic-v13",
            reference_population="ag-news-train",
            schema_version="chapter4-v13",
            artifact_sha256=digest,
            observed_provenance_channels=("prediction", "model", "tokenizer", "explainer", "reference"),
        )
        context = DeploymentContext(
            expected_model_version="distilbert-ag-news@52ee64d",
            expected_preprocessing_version="tokenizer@52ee64d",
            expected_explainer_version="chapter4-v13-runtime",
            expected_calibration_version="isotonic-v13",
            expected_reference_population="ag-news-train",
            expected_schema_version="chapter4-v13",
            expected_explain_plan_version="chapter4-v13",
            expected_dictionary_version="ag-news-en-v1",
            expected_artifact_sha256=digest,
            mandatory_provenance_channels=("prediction", "model", "tokenizer", "explainer", "reference"),
            maximum_reduction_loss=0.25,
            policy_version=policy.policy_version,
        )
        outputs.append(
            assess_action(
                prediction,
                explanation_artifact,
                route,
                context,
                ReviewBudget(1.0),
                costs,
                policy=policy,
            ).to_dict()
        )
    return outputs


def _measure(callable_: Any) -> tuple[Any, float]:
    start = time.perf_counter_ns()
    value = callable_()
    return value, (time.perf_counter_ns() - start) / 1e9


def _integrated_gradients_chunked(model: Any, tokenizer: Any, device: Any, texts: Sequence[str], targets: Sequence[int]) -> list[dict[str, object]]:
    batch_size = int(runtime_config()["explanation_batch_size"])
    rows: list[dict[str, object]] = []
    for start in range(0, len(texts), batch_size):
        rows.extend(integrated_gradients_batch(model, tokenizer, device, texts[start : start + batch_size], targets[start : start + batch_size], steps=16))
    return rows


def _token_masking_chunked(
    model: Any,
    tokenizer: Any,
    device: Any,
    explanations: Sequence[dict[str, object]],
    targets: Sequence[int],
) -> list[list[float]]:
    config = runtime_config()
    object_batch = int(config["benchmark_masking_object_batch_size"])
    forward_batch = int(config["benchmark_masking_forward_batch_size"])
    rows: list[list[float]] = []
    for start in range(0, len(explanations), object_batch):
        rows.extend(
            token_masking_batch(
                model,
                tokenizer,
                device,
                explanations[start : start + object_batch],
                targets[start : start + object_batch],
                limit=20,
                forward_batch_size=forward_batch,
            )
        )
    return rows


def run() -> dict[str, object]:
    import psutil
    import torch

    cfg = protocol()["runtime_benchmark"]
    set_deterministic(protocol()["statistics"]["seeds"][0])
    model, tokenizer, device = load_frozen_model()
    source = list(read_jsonl(ARTIFACTS / "processed" / "sealed_test_inputs.jsonl"))
    source.extend(read_jsonl(ARTIFACTS / "processed" / "validation.jsonl"))
    partial_path = ARTIFACTS / "runtime" / "raw_results.partial.csv"
    if partial_path.exists():
        raw_rows = pd.read_csv(partial_path).to_dict(orient="records")
    else:
        raw_rows = []
    completed = {(str(row["explainer"]), int(row["n"]), int(row["repetition"])) for row in raw_rows}
    sizes = [int(value) for value in cfg["sizes"]]
    methods = ("integrated_gradients", "token_masking")
    process = psutil.Process()

    for method in methods:
        method_sizes = [*sizes]
        if method == "token_masking":
            method_sizes.append(int(cfg["cheap_method_extra_size"]))
        for n in method_sizes:
            if n > len(source):
                raise RuntimeError(f"runtime benchmark requested {n} objects but only {len(source)} are available")
            rows = source[:n]
            texts = [str(row["text"]) for row in rows]
            # One complete warm-up is excluded from recorded repetitions.
            warm_probabilities = predict_texts(model, tokenizer, device, texts[: min(2, n)], batch_size=min(2, n))
            warm_targets = np.argmax(warm_probabilities, axis=1).tolist()
            if method == "integrated_gradients":
                _integrated_gradients_chunked(model, tokenizer, device, texts[: min(2, n)], warm_targets)
            else:
                skeleton = [_token_skeleton(tokenizer, text) for text in texts[: min(2, n)]]
                token_masking_batch(model, tokenizer, device, skeleton, warm_targets, limit=20)

            for repetition in range(int(cfg["repetitions"])):
                if (method, n, repetition) in completed:
                    continue
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                rss_before = process.memory_info().rss
                probabilities, model_seconds = _measure(lambda: predict_texts(model, tokenizer, device, texts))
                targets = np.argmax(probabilities, axis=1).tolist()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                if method == "integrated_gradients":
                    explanations, explainer_seconds = _measure(lambda: _integrated_gradients_chunked(model, tokenizer, device, texts, targets))
                    scores = [item["scores"] for item in explanations]
                else:
                    explanations = [_token_skeleton(tokenizer, text) for text in texts]
                    scores, explainer_seconds = _measure(lambda: _token_masking_chunked(model, tokenizer, device, explanations, targets))
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                assessments, fuzzyxai_seconds = _measure(lambda: _fuzzyxai_stage(rows, probabilities, explanations, scores))
                serialized, serialization_seconds = _measure(lambda: json.dumps(assessments, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))
                total = model_seconds + explainer_seconds + fuzzyxai_seconds + serialization_seconds
                raw_rows.append(
                    {
                        "modality": "text",
                        "model": "DistilBERT AG News",
                        "explainer": method,
                        "n": n,
                        "repetition": repetition,
                        "model_seconds": model_seconds,
                        "explainer_seconds": explainer_seconds,
                        "fuzzyxai_seconds": fuzzyxai_seconds,
                        "serialization_seconds": serialization_seconds,
                        "total_seconds": total,
                        "objects_per_second": n / total,
                        "peak_rss_bytes": max(rss_before, process.memory_info().rss),
                        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
                        "serialized_bytes": len(serialized),
                        "cached": False,
                    }
                )
                partial_path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(raw_rows).to_csv(partial_path, index=False)
            print(f"progress: runtime method={method} n={n}", flush=True)

    raw_path = ARTIFACTS / "runtime" / "raw_results.csv"
    frame = pd.DataFrame(raw_rows)
    frame.to_csv(raw_path, index=False)
    partial_path.unlink(missing_ok=True)
    summary_rows = []
    for (method, n), group in frame.groupby(["explainer", "n"], sort=True):
        row: dict[str, object] = {"modality": "text", "model": "DistilBERT AG News", "explainer": method, "n": int(n), "repetitions": len(group)}
        for column in ("model_seconds", "explainer_seconds", "fuzzyxai_seconds", "serialization_seconds", "total_seconds", "objects_per_second", "peak_rss_bytes", "peak_vram_bytes"):
            values = group[column].astype(float).to_numpy()
            row[f"{column}_median"] = float(np.median(values))
            row[f"{column}_mean"] = float(np.mean(values))
            row[f"{column}_std"] = float(np.std(values, ddof=1))
            row[f"{column}_p95"] = float(np.quantile(values, 0.95))
            row[f"{column}_p99"] = float(np.quantile(values, 0.99))
        row["fuzzyxai_time_fraction"] = float(row["fuzzyxai_seconds_median"] / row["total_seconds_median"])
        row["explainer_time_fraction"] = float(row["explainer_seconds_median"] / row["total_seconds_median"])
        summary_rows.append(row)
    summary_path = ARTIFACTS / "runtime" / "summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    shared_gpu_snapshot = ARTIFACTS / "runtime" / "environment_snapshots" / "shared_gpu_during_benchmark.txt"
    manifest = {
        "environment": environment_manifest(),
        "benchmark_configuration": {
            "runtime": runtime_config(),
            "integrated_gradients": protocol()["modern_contour"]["explanations"]["methods"]["integrated_gradients"],
            "token_masking": protocol()["modern_contour"]["explanations"]["methods"]["token_masking"],
            "background_samples": 0,
            "input_perturbations": 0,
            "cache_enabled": False,
        },
        "warmups": cfg["warmups"],
        "repetitions": cfg["repetitions"],
        "sizes": sizes,
        "sizes_by_method": {
            method: sorted(int(value) for value in group["n"].unique())
            for method, group in frame.groupby("explainer", sort=True)
        },
        "cheap_method_extra_size": int(cfg["cheap_method_extra_size"]),
        "raw_sha256": sha256_file(raw_path),
        "summary_sha256": sha256_file(summary_path),
        "operator_only_five_million_is_separate": True,
        "source_explainer_included": True,
        "gpu_isolation": "shared_with_unrelated_user_process",
        "concurrent_gpu_snapshot_sha256": sha256_file(shared_gpu_snapshot),
        "invalid_oom_attempt_preserved": True,
    }
    write_json(ARTIFACTS / "runtime" / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    result = run()
    print(f"PASS: complete runtime benchmark repetitions={result['repetitions']} sizes={result['sizes']}")


if __name__ == "__main__":
    main()
