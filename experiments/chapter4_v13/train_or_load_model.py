from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from .common import ARTIFACTS, environment_manifest, measured_stage, protocol, read_jsonl, runtime_config, sha256_file, write_json, write_jsonl


def set_deterministic(seed: int) -> None:
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def load_frozen_model() -> tuple[Any, Any, Any]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    cfg = protocol()["modern_contour"]
    runtime = runtime_config()
    model_id = cfg["model"]["id"]
    revision = cfg["model"]["revision"]
    cache_dir = ARTIFACTS / "model_cache"
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, cache_dir=cache_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, revision=revision, cache_dir=cache_dir)
    device = torch.device(runtime["device"] if runtime["device"] == "cuda" and torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, tokenizer, device


def predict_texts(model: Any, tokenizer: Any, device: Any, texts: Sequence[str], *, batch_size: int | None = None) -> np.ndarray:
    import torch

    cfg = protocol()["modern_contour"]
    runtime = runtime_config()
    batch_size = int(batch_size or runtime["prediction_batch_size"])
    rows: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            encoded = tokenizer(
                list(texts[start : start + batch_size]),
                padding=True,
                truncation=True,
                max_length=cfg["prediction"]["max_length"],
                return_tensors="pt",
            )
            encoded = {name: value.to(device) for name, value in encoded.items()}
            logits = model(**encoded).logits
            rows.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(rows, axis=0)


def _entropy(probabilities: np.ndarray) -> np.ndarray:
    classes = probabilities.shape[1]
    return -(probabilities * np.log(np.clip(probabilities, 1e-12, 1.0))).sum(axis=1) / math.log(classes)


def _prediction_rows(source: Sequence[dict[str, Any]], probabilities: np.ndarray, *, include_label: bool) -> list[dict[str, object]]:
    order = np.sort(probabilities, axis=1)
    entropy = _entropy(probabilities)
    rows = []
    for index, (item, probs) in enumerate(zip(source, probabilities, strict=True)):
        row: dict[str, object] = {
            "object_id": item["object_id"],
            "normalized_text_sha256": item["normalized_text_sha256"],
            "probabilities": [float(value) for value in probs],
            "prediction": int(np.argmax(probs)),
            "confidence": float(np.max(probs)),
            "entropy": float(entropy[index]),
            "margin": float(order[index, -1] - order[index, -2]),
        }
        if include_label:
            row["label"] = int(item["label"])
            row["is_correct"] = bool(row["prediction"] == row["label"])
        rows.append(row)
    return rows


def run_predictions(*, splits: Sequence[str] = ("train", "validation", "sealed_test")) -> dict[str, object]:
    import torch

    cfg = protocol()["modern_contour"]
    seed = protocol()["statistics"]["seeds"][0]
    set_deterministic(seed)
    model, tokenizer, device = load_frozen_model()
    output = ARTIFACTS / "predictions"
    output.mkdir(parents=True, exist_ok=True)
    timings = []
    summaries: dict[str, object] = {}
    for split in splits:
        path = ARTIFACTS / "processed" / ("sealed_test_inputs.jsonl" if split == "sealed_test" else f"{split}.jsonl")
        source = list(read_jsonl(path))
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        with measured_stage(f"predict_{split}") as timing:
            probabilities = predict_texts(model, tokenizer, device, [str(row["text"]) for row in source])
        timings.append(timing)
        rows = _prediction_rows(source, probabilities, include_label=split != "sealed_test")
        output_path = output / f"{split}.jsonl"
        write_jsonl(output_path, rows)
        summary: dict[str, object] = {
            "rows": len(rows),
            "sha256": sha256_file(output_path),
            "mean_confidence": float(probabilities.max(axis=1).mean()),
            "labels_present": split != "sealed_test",
        }
        if split != "sealed_test":
            summary["accuracy"] = float(np.mean([bool(row["is_correct"]) for row in rows]))
        summaries[split] = summary
    write_json(output / "timings.json", timings)
    write_json(
        ARTIFACTS / "manifests" / "model_manifest.json",
        {
            "model": cfg["model"],
            "prediction": cfg["prediction"],
            "environment": environment_manifest(),
            "splits": summaries,
            "test_labels_loaded": False,
        },
    )
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "sealed_test"])
    args = parser.parse_args()
    result = run_predictions(splits=args.splits)
    print("PASS: frozen model predictions " + ", ".join(f"{name}={value['rows']}" for name, value in result.items()))


if __name__ == "__main__":
    main()
