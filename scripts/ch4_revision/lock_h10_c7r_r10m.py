#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

from fuzzyxai.experiments.h10_c7r_r10m import (
    BGE_RERANKER_ID,
    BGE_RERANKER_REVISION,
    GRAPHCODEBERT_ID,
    GRAPHCODEBERT_REVISION,
    R10MConfig,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inventory(path: Path) -> tuple[list[dict[str, object]], str]:
    files = [
        {
            "path": str(file_path.relative_to(path)),
            "size": file_path.stat().st_size,
            "sha256": _sha256(file_path),
        }
        for file_path in sorted(path.rglob("*"))
        if file_path.is_file()
        and ".cache" not in file_path.relative_to(path).parts
    ]
    payload = "".join(
        f"{value['sha256']}  {value['path']}\n" for value in files
    ).encode()
    return files, hashlib.sha256(payload).hexdigest()


def _version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "NOT_INSTALLED_NOT_USED"


def _model(
    *,
    role: str,
    model_id: str,
    revision: str,
    local_path: Path,
    weight_name: str,
    tokenizer_names: tuple[str, ...],
    precision: str,
    maximum_input_length: int,
    batch_size: int,
) -> dict[str, object]:
    files, snapshot = _inventory(local_path)
    by_name = {str(value["path"]): value for value in files}
    tokenizer_payload = "".join(
        f"{by_name[name]['sha256']}  {name}\n"
        for name in tokenizer_names
        if name in by_name
    ).encode()
    return {
        "role": role,
        "model_id": model_id,
        "revision": revision,
        "local_path": str(local_path.resolve()),
        "portable_mount_path": f"/models/h10-c7r-r10m/{local_path.name}",
        "snapshot_sha256": snapshot,
        "weight_files": [weight_name],
        "weights_sha256": by_name[weight_name]["sha256"],
        "tokenizer_files": list(tokenizer_names),
        "tokenizer_sha256": hashlib.sha256(tokenizer_payload).hexdigest(),
        "config_sha256": by_name["config.json"]["sha256"],
        "precision": precision,
        "maximum_input_length": maximum_input_length,
        "batch_size": batch_size,
        "files": files,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphcodebert", type=Path, required=True)
    parser.add_argument("--bge", type=Path, required=True)
    parser.add_argument("--input-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch

    config = R10MConfig()
    model_lock = {
        "protocol_id": "H10-C7R-R10M-v1",
        "status": "LOCKED_BEFORE_DEVELOPMENT_SCORING",
        "network_allowed_during_scoring": False,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": _version("transformers"),
            "sentence_transformers": _version("sentence-transformers"),
            "huggingface_hub": _version("huggingface-hub"),
            "device": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else "CUDA_UNAVAILABLE"
            ),
        },
        "models": [
            _model(
                role="dense_retrieval",
                model_id=GRAPHCODEBERT_ID,
                revision=GRAPHCODEBERT_REVISION,
                local_path=args.graphcodebert.resolve(),
                weight_name="pytorch_model.bin",
                tokenizer_names=(
                    "merges.txt",
                    "special_tokens_map.json",
                    "tokenizer_config.json",
                    "vocab.json",
                ),
                precision="float16",
                maximum_input_length=config.graphcodebert_max_length,
                batch_size=config.graphcodebert_batch_size,
            ),
            _model(
                role="pair_reranking",
                model_id=BGE_RERANKER_ID,
                revision=BGE_RERANKER_REVISION,
                local_path=args.bge.resolve(),
                weight_name="model.safetensors",
                tokenizer_names=(
                    "sentencepiece.bpe.model",
                    "special_tokens_map.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                ),
                precision="float16",
                maximum_input_length=config.bge_max_length,
                batch_size=config.bge_batch_size,
            ),
        ],
    }
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    model_path = output / "R10M_MODEL_LOCK.json"
    model_path.write_text(
        json.dumps(model_lock, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    root = Path.cwd().resolve()
    source_paths = (
        "framework/fuzzyxai/fuzzyxai/experiments/h10_c7r_r10m.py",
        "scripts/ch4_revision/prepare_h10_c7r_r10m_development.py",
        "scripts/ch4_revision/run_h10_c7r_r10m_development.py",
        "scripts/ch4_revision/lock_h10_c7r_r10m.py",
    )
    audit = json.loads(args.input_audit.read_text(encoding="utf-8"))
    method_lock = {
        "protocol_id": "H10-C7R-R10M-v1",
        "status": "LOCKED_BEFORE_DEVELOPMENT_SCORING",
        "retrieval": {
            "file_channels": [
                "causal_chronology",
                "traceback_execution",
                "bm25",
                "graphcodebert",
            ],
            "file_limit": config.file_limit,
            "symbol_pool_limit": config.symbol_pool_limit,
            "symbol_pool_channel_limits": {
                "runtime": config.runtime_pool_limit,
                "traceback": config.traceback_pool_limit,
                "value_flow": config.value_flow_pool_limit,
                "graphcodebert": config.dense_pool_limit,
                "bm25": config.bm25_pool_limit,
                "graph": config.graph_pool_limit,
            },
            "final_channels": [
                "causal",
                "bm25",
                "graphcodebert",
                "bge_reranker",
            ],
            "final_limit": config.final_limit,
            "rrf_constant": config.rrf_constant,
            "contract_inference_changes_localization": False,
            "bounded_probes_in_primary_result": False,
            "learned_on_development": False,
        },
        "baselines": ["B_TRACE", "B_BM25", "R9A", "R10A"],
        "query_forbidden_fields": [
            "gold_patch",
            "fix_commit",
            "changed_files",
            "changed_symbols",
            "diff",
        ],
        "input_hashes": {
            "observable_manifest": audit["observable_manifest_sha256"],
            "runtime_readiness": audit["runtime_readiness_sha256"],
            "development_gold": audit["development_gold_sha256"],
            "v1_evidence": audit["v1_evidence_sha256"],
        },
        "model_lock_sha256": _sha256(model_path),
        "endpoint_lock_sha256": _sha256(output / "R10M_ENDPOINTS.json"),
        "source_hashes": {
            value: _sha256(root / value) for value in source_paths
        },
        "scientific_result": "NOT_EVALUATED",
        "held_out_created": False,
        "held_out_scored": False,
    }
    (output / "R10M_METHOD_LOCK.json").write_text(
        json.dumps(method_lock, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "model_lock_sha256": _sha256(model_path),
                "method_lock_sha256": _sha256(
                    output / "R10M_METHOD_LOCK.json"
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
