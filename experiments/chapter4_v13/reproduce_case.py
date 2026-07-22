from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np

from .benchmark_end_to_end import _fuzzyxai_stage
from .common import ARTIFACTS, canonical_bytes, load_yaml, protocol, read_jsonl, sha256_bytes, sha256_file, write_json
from .generate_explanations import integrated_gradients_batch, token_masking_batch
from .train_or_load_model import load_frozen_model, predict_texts, set_deterministic


def _timed(callable_: Any) -> tuple[Any, float]:
    started = time.perf_counter_ns()
    result = callable_()
    return result, (time.perf_counter_ns() - started) / 1e9


def reproduce(config_path: Path, output: Path) -> dict[str, object]:
    config = load_yaml(config_path)["case"]
    if config["selection_rule"] != "first_explained_object_by_sorted_object_id":
        raise ValueError("unsupported frozen case selection rule")
    explanations = sorted(read_jsonl(ARTIFACTS / "explanations" / "sealed_test.jsonl"), key=lambda row: str(row["object_id"]))
    selected = explanations[0]
    object_id = str(selected["object_id"])
    inputs = {str(row["object_id"]): row for row in read_jsonl(ARTIFACTS / "processed" / "sealed_test_inputs.jsonl")}
    source = inputs[object_id]
    text = str(source["text"])
    set_deterministic(protocol()["statistics"]["seeds"][0])
    model, tokenizer, device = load_frozen_model()
    probabilities, model_seconds = _timed(lambda: predict_texts(model, tokenizer, device, [text]))
    target = [int(np.argmax(probabilities[0]))]
    ig, ig_seconds = _timed(lambda: integrated_gradients_batch(model, tokenizer, device, [text], target, steps=16))
    masking, masking_seconds = _timed(lambda: token_masking_batch(model, tokenizer, device, ig, target, limit=20))
    canonical = dict(selected["canonical_payload"])
    saved_hash = str(selected["canonical_sha256"])
    canonical_hash = sha256_bytes(canonical_bytes(canonical))
    if canonical_hash != saved_hash:
        raise RuntimeError("saved canonical payload does not match its digest")
    assessments, fuzzyxai_seconds = _timed(lambda: _fuzzyxai_stage([source], probabilities, ig, masking))
    assessment = assessments[0]
    serialized, serialization_seconds = _timed(lambda: canonical_bytes(assessment))

    output.mkdir(parents=True, exist_ok=True)
    write_json(
        output / "input_reference.json",
        {
            "object_id": object_id,
            "dataset": protocol()["modern_contour"]["dataset"]["id"],
            "source_split": source["source_split"],
            "source_index": source["source_index"],
            "normalized_text_sha256": source["normalized_text_sha256"],
            "raw_text_included": False,
            "limitation": "upstream AG News dataset card reports an unknown license; fetch by source reference",
        },
    )
    write_json(output / "prediction.json", {"class": target[0], "probabilities": probabilities[0].tolist(), "model": protocol()["modern_contour"]["model"]})
    write_json(
        output / "local_explanation.json",
        {
            "tokens": ig[0]["tokens"],
            "integrated_gradients": ig[0]["scores"],
            "token_masking": masking[0],
            "methods": ["integrated_gradients", "token_masking"],
        },
    )
    write_json(output / "canonical_artifact.json", {"sha256": canonical_hash, "payload": canonical, "hash_preserved": True})
    write_json(
        output / "provenance_graph.json",
        {
            "nodes": [
                {"id": f"dataset:{source['normalized_text_sha256']}", "type": "evidence"},
                {"id": "model:distilbert-ag-news@52ee64d", "type": "model"},
                {"id": f"canonical:{canonical_hash}", "type": "canonical_explanation"},
                {"id": f"action:{assessment['trace_id']}", "type": "action"},
            ],
            "edges": [
                [f"dataset:{source['normalized_text_sha256']}", "model:distilbert-ag-news@52ee64d"],
                ["model:distilbert-ag-news@52ee64d", f"canonical:{canonical_hash}"],
                [f"canonical:{canonical_hash}", f"action:{assessment['trace_id']}"],
            ],
        },
    )
    write_json(output / "diagnostic_state.json", {"hard_guard_status": assessment["hard_guard_status"], "reason_codes": assessment["reason_codes"], "missing_evidence": assessment["missing_evidence"]})
    write_json(output / "action.json", assessment)
    timings = {
        "model_seconds": model_seconds,
        "integrated_gradients_seconds": ig_seconds,
        "token_masking_seconds": masking_seconds,
        "fuzzyxai_seconds": fuzzyxai_seconds,
        "serialization_seconds": serialization_seconds,
        "total_seconds": model_seconds + ig_seconds + masking_seconds + fuzzyxai_seconds + serialization_seconds,
        "serialized_action_bytes": len(serialized),
    }
    write_json(output / "stage_timings.json", timings)
    top = np.argsort(-np.abs(np.asarray(ig[0]["scores"], dtype=float)))[:3]
    reasons = [str(ig[0]["tokens"][index]) for index in top]
    (output / "user_card.md").write_text(
        "# Результат\n\n"
        f"Модель отнесла текст к классу `{target[0]}`.\n\n"
        "# Основные основания\n\n"
        f"Наиболее заметные токены: {', '.join(reasons)}. Это локальные ассоциации модели, а не доказанные причины события.\n\n"
        "# Ограничения\n\n"
        "Integrated Gradients и маскирование токенов описывают чувствительность frozen DistilBERT. Они не подтверждают истинность новости и не заменяют предметную проверку.\n\n"
        "# Действие\n\n"
        f"Контроллер выбрал `{assessment['action']}`; структурный статус `{assessment['hard_guard_status']}`.\n",
        encoding="utf-8",
    )
    (output / "audit.md").write_text(
        "# Аудиторский вывод\n\n"
        f"- object_id: `{object_id}`\n"
        f"- canonical SHA256: `{canonical_hash}`\n"
        f"- trace_id: `{assessment['trace_id']}`\n"
        f"- deterministic replay SHA256: `{assessment['deterministic_replay_sha256']}`\n"
        f"- model revision: `{protocol()['modern_contour']['model']['revision']}`\n"
        "- test label was not loaded by the case reproducer.\n",
        encoding="utf-8",
    )
    write_json(
        output / "release_manifest.json",
        {
            "public_files": [
                "input_reference.json",
                "prediction.json",
                "provenance_graph.json",
                "diagnostic_state.json",
                "action.json",
                "stage_timings.json",
                "audit.md",
                "release_manifest.json",
                "SHA256SUMS",
            ],
            "local_only_files": ["local_explanation.json", "canonical_artifact.json", "user_card.md"],
            "reason": "AG News upstream dataset card reports license unknown; token-bearing derivatives are not redistributed",
            "reproduction_command": "python -m experiments.chapter4_v13.reproduce_case --config config/chapter4_v13_case.yaml --output artifacts/chapter4_v13/end_to_end_case",
        },
    )
    files = sorted(path for path in output.iterdir() if path.is_file() and path.name != "SHA256SUMS")
    (output / "SHA256SUMS").write_text("".join(f"{sha256_file(path)}  {path.name}\n" for path in files), encoding="utf-8")
    summary = {"object_id": object_id, "canonical_sha256": canonical_hash, "action": assessment["action"], "timings": timings}
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/chapter4_v13_case.yaml"))
    parser.add_argument("--output", type=Path, default=ARTIFACTS / "end_to_end_case")
    args = parser.parse_args()
    result = reproduce(args.config, args.output)
    print(f"PASS: reproduced case object={result['object_id']} action={result['action']}")


if __name__ == "__main__":
    main()
