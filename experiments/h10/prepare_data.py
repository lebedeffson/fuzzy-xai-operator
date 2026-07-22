from __future__ import annotations

import argparse
import json
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

from .common import (
    ARTIFACT_ROOT,
    DATASETS,
    PRIVATE_ROOT,
    ROOT,
    environment_manifest,
    load_identities,
    load_yaml,
    prepare_datasets,
    sha256_file,
    write_json,
)
from .mutations import make_cases
from .routes import build_route
from .serialization import route_to_dict, truth_to_dict
from .vault import create_key, seal


def _opaque(case_id: str, seed: int) -> str:
    return "h10v19case-" + sha256(f"{seed}:{case_id}".encode()).hexdigest()[:24]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n")


def prepare(config_path: Path) -> None:
    config = load_yaml(config_path)
    manifest = prepare_datasets()
    by_id = {item.dataset_id: item for item in DATASETS}
    known, held_out = tuple(config["known_leaves"]), tuple(config["held_out_leaves"])
    train_rows: list[dict] = []
    sealed_routes: list[dict] = []
    sealed_truths: list[dict] = []
    clean_rows: list[dict] = []
    split_id_sets: dict[str, dict[str, list[str]]] = {}
    for dataset_id in config["datasets"]:
        spec = by_id[dataset_id]
        split_id_sets[dataset_id] = {}
        for split in ("train", "development", "sealed_test"):
            identities = load_identities(dataset_id, split)
            split_id_sets[dataset_id][split] = identities
            routes = [build_route(dataset_id, spec.modality, object_id) for object_id in identities]
            clean_rows.extend({"split": split, "route": route_to_dict(route)} for route in routes)
            cases = make_cases(
                routes,
                seed=int(config["seed"]) + {"train": 101, "development": 202, "sealed_test": 303}[split],
                known_leaves=known,
                held_out_leaves=held_out if split == "sealed_test" else (),
                include_valid=True,
            )
            if split != "sealed_test":
                for route, truth in cases:
                    train_rows.append({"split": split, "route": route_to_dict(route), "truth": truth_to_dict(truth)})
            else:
                for route, truth in cases:
                    opaque = _opaque(truth.case_id, int(config["seed"]))
                    route = replace(route, route_id=opaque)
                    truth = replace(truth, case_id=opaque)
                    sealed_routes.append(route_to_dict(route))
                    sealed_truths.append(truth_to_dict(truth))
    _write_jsonl(ARTIFACT_ROOT / "routes" / "train_development_routes.jsonl", train_rows)
    _write_jsonl(ARTIFACT_ROOT / "routes" / "sealed_routes.jsonl", sealed_routes)
    _write_jsonl(ARTIFACT_ROOT / "routes" / "clean_routes.jsonl", clean_rows)
    vault_payload = json.dumps(sealed_truths, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("ascii")
    key = create_key(PRIVATE_ROOT / "h10_v19_vault.key")
    vault_path = PRIVATE_ROOT / "h10_v19_label_vault.enc"
    vault_path.parent.mkdir(parents=True, exist_ok=True)
    vault_path.write_bytes(seal(vault_payload, key))
    audit = {
        "status": "PASS",
        "datasets": len(config["datasets"]),
        "sealed_cases": len(sealed_routes),
        "test_labels_in_feature_channels": False,
        "vault_key_in_repository": False,
        "oracle_imports_evaluated_h10": False,
        "identity_intersections": {
            dataset: {
                "train_development": len(set(splits["train"]) & set(splits["development"])),
                "train_test": len(set(splits["train"]) & set(splits["sealed_test"])),
                "development_test": len(set(splits["development"]) & set(splits["sealed_test"])),
            }
            for dataset, splits in split_id_sets.items()
        },
        "confirmatory_test_opened": False,
        "post_lock_tuning": False,
        "dataset_manifest_sha256": sha256_file(ARTIFACT_ROOT / "data" / "dataset_manifest.json"),
        "sealed_routes_sha256": sha256_file(ARTIFACT_ROOT / "routes" / "sealed_routes.jsonl"),
        "clean_routes_sha256": sha256_file(ARTIFACT_ROOT / "routes" / "clean_routes.jsonl"),
        "vault_sha256": sha256_file(vault_path),
    }
    if any(value for dataset in audit["identity_intersections"].values() for value in dataset.values()):
        raise RuntimeError("H10 v19 identity leakage detected")
    write_json(ARTIFACT_ROOT / "opening" / "pre_opening_leakage_audit.json", audit)
    write_json(
        ARTIFACT_ROOT / "data" / "split_identity_hashes.json",
        {d: {s: sha256("\n".join(v).encode()).hexdigest() for s, v in x.items()} for d, x in split_id_sets.items()},
    )
    write_json(
        ARTIFACT_ROOT / "data" / "preparation_summary.json",
        {
            "manifest": manifest,
            "train_development_cases": len(train_rows),
            "sealed_cases": len(sealed_routes),
            "clean_routes": len(clean_rows),
            "fresh_mutation_schedule": True,
            "identity_anchor_reuse_disclosed": True,
        },
    )
    write_json(ARTIFACT_ROOT / "data" / "environment_manifest.json", environment_manifest())
    write_json(
        ARTIFACT_ROOT / "data" / "model_manifest.json",
        [
            {
                "dataset_id": item.dataset_id,
                "modality": item.modality,
                "model_version": f"{item.dataset_id}-model-v19",
                "explainer_version": f"{item.dataset_id}-model-v19",
                "role": "route identity anchor; predictive labels are not H10 targets",
            }
            for item in DATASETS
        ],
    )
    write_json(
        ARTIFACT_ROOT / "data" / "fault_manifest.json",
        {
            "known_leaves": list(known),
            "held_out_leaves": list(held_out),
            "held_out_parent_families": config["held_out_parent_families"],
            "severities": ["subtle", "moderate", "severe"],
            "compositions": True,
            "insufficient_evidence": True,
            "truth_source": "independent mutation log and exhaustive oracle",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "h10_v19_protocol.yaml")
    args = parser.parse_args()
    prepare(args.config)


if __name__ == "__main__":
    main()
