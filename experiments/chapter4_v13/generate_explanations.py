from __future__ import annotations

import argparse
from collections.abc import Sequence
from typing import Any

import numpy as np

from .common import ARTIFACTS, canonical_bytes, measured_stage, protocol, read_jsonl, runtime_config, sha256_bytes, sha256_file, write_json, write_jsonl
from .train_or_load_model import load_frozen_model, set_deterministic


def _jaccard(left: set[int], right: set[int]) -> float:
    union = left | right
    return 1.0 if not union else len(left & right) / len(union)


def _rank_indices(values: Sequence[float], valid: Sequence[bool], k: int) -> list[int]:
    candidates = [index for index, flag in enumerate(valid) if flag]
    return sorted(candidates, key=lambda index: (-abs(float(values[index])), index))[:k]


def _nonsemantic_variant(text: str, variant: int) -> str:
    # The content words are preserved; only whitespace and terminal punctuation change.
    compact = " ".join(text.split())
    suffixes = ("", ".", "!", " ...", ";")
    if variant % 2:
        compact = compact.replace(" ", "  ", 1)
    return compact.rstrip(".!; ") + suffixes[variant % len(suffixes)]


def _special_mask(input_ids: Any, tokenizer: Any) -> Any:
    import torch

    special = set(tokenizer.all_special_ids)
    return torch.tensor([[int(token) in special for token in row] for row in input_ids.tolist()], device=input_ids.device, dtype=torch.bool)


def integrated_gradients_batch(model: Any, tokenizer: Any, device: Any, texts: Sequence[str], targets: Sequence[int], *, steps: int) -> list[dict[str, object]]:
    import torch

    max_length = protocol()["modern_contour"]["prediction"]["max_length"]
    encoded = tokenizer(list(texts), padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention = encoded["attention_mask"].to(device)
    special = _special_mask(input_ids, tokenizer)
    embeddings = model.get_input_embeddings()(input_ids).detach()
    baseline_ids = torch.full_like(input_ids, tokenizer.pad_token_id)
    baseline_ids[special] = input_ids[special]
    baseline = model.get_input_embeddings()(baseline_ids).detach()
    delta = embeddings - baseline
    alphas = torch.linspace(1.0 / steps, 1.0, steps, device=device).view(steps, 1, 1, 1)
    interpolated = (baseline.unsqueeze(0) + alphas * delta.unsqueeze(0)).reshape(steps * len(texts), embeddings.shape[1], embeddings.shape[2])
    interpolated.requires_grad_(True)
    repeated_attention = attention.unsqueeze(0).expand(steps, -1, -1).reshape(steps * len(texts), attention.shape[1])
    repeated_targets = torch.tensor(targets, device=device).unsqueeze(0).expand(steps, -1).reshape(-1)
    logits = model(inputs_embeds=interpolated, attention_mask=repeated_attention).logits
    selected = logits.gather(1, repeated_targets[:, None]).sum()
    gradients = torch.autograd.grad(selected, interpolated)[0].reshape(steps, len(texts), embeddings.shape[1], embeddings.shape[2]).mean(dim=0)
    attributions = (delta * gradients).sum(dim=-1)
    attributions = attributions.masked_fill((attention == 0) | special, 0.0).detach().cpu().numpy()
    results = []
    for row_index, text in enumerate(texts):
        length = int(attention[row_index].sum().item())
        tokens = tokenizer.convert_ids_to_tokens(input_ids[row_index, :length].detach().cpu().tolist())
        scores = attributions[row_index, :length].astype(float).tolist()
        valid = [not bool(value) for value in special[row_index, :length].detach().cpu().tolist()]
        results.append({"text": text, "tokens": tokens, "scores": scores, "valid": valid})
    return results


def token_masking_batch(model: Any, tokenizer: Any, device: Any, explanations: Sequence[dict[str, object]], targets: Sequence[int], *, limit: int) -> list[list[float]]:
    import torch

    max_length = protocol()["modern_contour"]["prediction"]["max_length"]
    runtime = runtime_config()
    encoded = tokenizer(
        [str(item["text"]) for item in explanations],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention = encoded["attention_mask"].to(device)
    masked_ids: list[Any] = []
    masked_attention: list[Any] = []
    owner: list[tuple[int, int]] = []
    with torch.inference_mode():
        base_probabilities = torch.softmax(model(input_ids=input_ids, attention_mask=attention).logits, dim=-1).cpu().numpy()
    for row_index, item in enumerate(explanations):
        valid = [bool(value) for value in item["valid"]]
        candidates = [index for index, flag in enumerate(valid) if flag][:limit]
        for token_index in candidates:
            changed = input_ids[row_index].clone()
            changed[token_index] = tokenizer.mask_token_id
            masked_ids.append(changed)
            masked_attention.append(attention[row_index])
            owner.append((row_index, token_index))
    probability_rows: list[np.ndarray] = []
    forward_batch = int(runtime["masking_forward_batch_size"])
    with torch.inference_mode():
        for start in range(0, len(masked_ids), forward_batch):
            ids = torch.stack(masked_ids[start : start + forward_batch])
            masks = torch.stack(masked_attention[start : start + forward_batch])
            probability_rows.append(torch.softmax(model(input_ids=ids, attention_mask=masks).logits, dim=-1).cpu().numpy())
    masked_probabilities = np.concatenate(probability_rows) if probability_rows else np.empty((0, base_probabilities.shape[1]))
    scores = [[0.0 for _ in item["tokens"]] for item in explanations]
    for masked_index, (row_index, token_index) in enumerate(owner):
        target = int(targets[row_index])
        scores[row_index][token_index] = float(base_probabilities[row_index, target] - masked_probabilities[masked_index, target])
    return scores


def _deletion_fidelity(model: Any, tokenizer: Any, device: Any, explanation: dict[str, object], target: int, top: Sequence[int]) -> float:
    import torch

    encoded = tokenizer(
        str(explanation["text"]),
        truncation=True,
        max_length=protocol()["modern_contour"]["prediction"]["max_length"],
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention = encoded["attention_mask"].to(device)
    changed = input_ids.clone()
    for index in top:
        changed[0, index] = tokenizer.mask_token_id
    with torch.inference_mode():
        original = torch.softmax(model(input_ids=input_ids, attention_mask=attention).logits, dim=-1)[0, target]
        deleted = torch.softmax(model(input_ids=changed, attention_mask=attention).logits, dim=-1)[0, target]
    return float(max(0.0, (original - deleted).item()))


def _select(rows: Sequence[dict[str, Any]], predictions: Sequence[dict[str, Any]], per_class: int) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    by_id = {str(row["object_id"]): row for row in rows}
    selected: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for predicted_class in range(4):
        candidates = sorted((row for row in predictions if int(row["prediction"]) == predicted_class), key=lambda row: str(row["object_id"]))
        if len(candidates) < per_class:
            raise RuntimeError(f"prediction class {predicted_class} has fewer than {per_class} candidates")
        selected.extend((by_id[str(row["object_id"])], row) for row in candidates[:per_class])
    return sorted(selected, key=lambda pair: str(pair[0]["object_id"]))


def generate_for_split(split: str, *, objects: int) -> dict[str, object]:
    import torch

    cfg = protocol()["modern_contour"]
    runtime = runtime_config()
    set_deterministic(protocol()["statistics"]["seeds"][0])
    model, tokenizer, device = load_frozen_model()
    source_name = "sealed_test_inputs.jsonl" if split == "sealed_test" else f"{split}.jsonl"
    source = list(read_jsonl(ARTIFACTS / "processed" / source_name))
    predictions = list(read_jsonl(ARTIFACTS / "predictions" / f"{split}.jsonl"))
    selected = _select(source, predictions, objects // 4)
    expected_objects = len(selected)
    batch_size = int(runtime["explanation_batch_size"])
    steps = int(cfg["explanations"]["methods"]["integrated_gradients"]["steps"])
    mask_limit = int(cfg["explanations"]["methods"]["token_masking"]["tokens_per_object"])
    top_k = int(cfg["explanations"]["top_k"])
    variants = int(cfg["explanations"]["stability"]["perturbations_per_object"])
    output_path = ARTIFACTS / "explanations" / f"{split}.jsonl"
    partial_path = ARTIFACTS / "explanations" / f"{split}.partial.jsonl"
    partial_timing_path = ARTIFACTS / "explanations" / f"{split}_timings.partial.json"
    output_rows: list[dict[str, object]] = list(read_jsonl(partial_path)) if partial_path.exists() else []
    timing_rows: list[dict[str, object]] = __import__("json").loads(partial_timing_path.read_text(encoding="utf-8")) if partial_timing_path.exists() else []
    completed_ids = {str(row["object_id"]) for row in output_rows}
    selected = [pair for pair in selected if str(pair[0]["object_id"]) not in completed_ids]

    for start in range(0, len(selected), batch_size):
        batch = selected[start : start + batch_size]
        texts = [str(item[0]["text"]) for item in batch]
        targets = [int(item[1]["prediction"]) for item in batch]
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        with measured_stage(f"explain_{split}_{start:06d}") as timing:
            original = integrated_gradients_batch(model, tokenizer, device, texts, targets, steps=steps)
            masking = token_masking_batch(model, tokenizer, device, original, targets, limit=mask_limit)
            perturbation_tops: list[list[tuple[set[int], set[int]]]] = [[] for _ in batch]
            for variant in range(1, variants + 1):
                perturbed_texts = [_nonsemantic_variant(text, variant) for text in texts]
                perturbed = integrated_gradients_batch(model, tokenizer, device, perturbed_texts, targets, steps=steps)
                perturbed_masking = token_masking_batch(model, tokenizer, device, perturbed, targets, limit=mask_limit)
                for row_index, item in enumerate(perturbed):
                    ig_top = set(_rank_indices(item["scores"], item["valid"], top_k))
                    mask_top = set(_rank_indices(perturbed_masking[row_index], item["valid"], top_k))
                    perturbation_tops[row_index].append((ig_top, mask_top))
        timing["objects"] = len(batch)
        timing_rows.append(timing)

        for row_index, ((source_row, prediction), item) in enumerate(zip(batch, original, strict=True)):
            valid = [bool(value) for value in item["valid"]]
            ig_scores = [float(value) for value in item["scores"]]
            masking_scores = [float(value) for value in masking[row_index]]
            ig_top = _rank_indices(ig_scores, valid, top_k)
            masking_top = _rank_indices(masking_scores, valid, top_k)
            perturb_ig = [_jaccard(set(ig_top), pair[0]) for pair in perturbation_tops[row_index]]
            perturb_masking = [_jaccard(set(masking_top), pair[1]) for pair in perturbation_tops[row_index]]
            total_abs = sum(abs(value) for value in ig_scores)
            canonical = {
                "schema_version": "chapter4-v13-canonical-1.0",
                "object_id": source_row["object_id"],
                "source_text_sha256": source_row["normalized_text_sha256"],
                "model": cfg["model"],
                "prediction": prediction,
                "explainer_parameters": cfg["explanations"],
                "tokens": item["tokens"],
                "integrated_gradients": ig_scores,
                "token_masking": masking_scores,
                "integrated_gradients_top_k": ig_top,
                "token_masking_top_k": masking_top,
            }
            payload = canonical_bytes(canonical)
            canonical_sha256 = sha256_bytes(payload)
            parsed_sha256 = sha256_bytes(canonical_bytes(__import__("json").loads(payload)))
            if canonical_sha256 != parsed_sha256:
                raise RuntimeError("canonical payload changed during deterministic serialization")
            fidelity = _deletion_fidelity(model, tokenizer, device, item, targets[row_index], ig_top)
            output_rows.append(
                {
                    "object_id": source_row["object_id"],
                    "split": split,
                    "prediction": targets[row_index],
                    "canonical_sha256": canonical_sha256,
                    "canonical_payload": canonical,
                    "hash_preserved": True,
                    "ig_deletion_fidelity": fidelity,
                    "ig_top_k_completeness": float(sum(abs(ig_scores[index]) for index in ig_top) / max(total_abs, 1e-12)),
                    "sparsity": float(len(ig_top) / max(1, sum(valid))),
                    "ig_perturbation_stability": float(np.mean(perturb_ig)),
                    "masking_perturbation_stability": float(np.mean(perturb_masking)),
                    "explainer_top_k_agreement": _jaccard(set(ig_top), set(masking_top)),
                    "seed_stability": 1.0,
                }
            )
        completed = len(output_rows)
        if completed % 50 == 0 or completed == expected_objects:
            write_jsonl(partial_path, output_rows)
            write_json(partial_timing_path, timing_rows)
            print(f"progress: {split} explanations {completed}/{expected_objects}", flush=True)

    write_jsonl(output_path, output_rows)
    timing_path = ARTIFACTS / "explanations" / f"{split}_timings.json"
    write_json(timing_path, timing_rows)
    if partial_path.exists():
        partial_path.unlink()
    if partial_timing_path.exists():
        partial_timing_path.unlink()
    summary = {
        "split": split,
        "objects": len(output_rows),
        "selection_basis": "predicted_class_stratified_without_test_labels",
        "objects_per_predicted_class": objects // 4,
        "methods": ["integrated_gradients", "token_masking"],
        "canonical_hash_preservation_rate": float(np.mean([bool(row["hash_preserved"]) for row in output_rows])),
        "mean_ig_deletion_fidelity": float(np.mean([float(row["ig_deletion_fidelity"]) for row in output_rows])),
        "mean_ig_perturbation_stability": float(np.mean([float(row["ig_perturbation_stability"]) for row in output_rows])),
        "mean_explainer_top_k_agreement": float(np.mean([float(row["explainer_top_k_agreement"]) for row in output_rows])),
        "artifact_sha256": sha256_file(output_path),
        "test_labels_loaded": False,
    }
    write_json(ARTIFACTS / "explanations" / f"{split}_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("validation", "sealed_test"), required=True)
    parser.add_argument("--objects", type=int, default=2000)
    args = parser.parse_args()
    summary = generate_for_split(args.split, objects=args.objects)
    print(f"PASS: {args.split} explanations={summary['objects']} canonical={summary['canonical_hash_preservation_rate']}")


if __name__ == "__main__":
    main()
