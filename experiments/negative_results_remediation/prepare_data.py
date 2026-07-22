from __future__ import annotations

import hashlib

import numpy as np
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split

from .common import ARTIFACTS, CONFIG, git_commit, read_json, sha256_json, verify_protocol, write_json


def _dataset_record(dataset_id: str, values: np.ndarray, labels: np.ndarray, *, seed: int) -> tuple[dict[str, object], dict[str, object]]:
    ids = np.asarray([f"{dataset_id}:{index}" for index in range(len(labels))])
    train_ids, test_ids, train_labels, test_labels = train_test_split(ids, labels, test_size=0.2, stratify=labels, random_state=seed)
    train_ids, dev_ids, _, dev_labels = train_test_split(train_ids, train_labels, test_size=0.25, stratify=train_labels, random_state=seed + 1)
    assert not (set(train_ids) & set(dev_ids) or set(train_ids) & set(test_ids) or set(dev_ids) & set(test_ids))
    content_hash = hashlib.sha256(np.ascontiguousarray(values).tobytes() + np.ascontiguousarray(labels).tobytes()).hexdigest()
    manifest = {
        "dataset_id": dataset_id,
        "source": "scikit-learn bundled UCI dataset",
        "license": "see scikit-learn dataset documentation and original UCI terms",
        "n_objects": len(labels),
        "n_features": values.shape[1],
        "n_classes": len(np.unique(labels)),
        "content_sha256": content_hash,
        "role": "formative_and_rule_effect_evaluation",
        "confirmatory_independence": False,
    }
    split = {
        "dataset_id": dataset_id,
        "seed": seed,
        "train_ids": sorted(train_ids.tolist()),
        "development_ids": sorted(dev_ids.tolist()),
        "sealed_test_ids": sorted(test_ids.tolist()),
        "identity_overlap_count": 0,
        "test_labels_as_features": False,
        "threshold_selection_scope": "development_only",
        "split_sha256": sha256_json({"train": sorted(train_ids.tolist()), "development": sorted(dev_ids.tolist()), "test": sorted(test_ids.tolist())}),
        "class_counts": {
            "development": np.bincount(dev_labels).tolist(),
            "test": np.bincount(test_labels).tolist(),
        },
    }
    return manifest, split


def main() -> None:
    protocol_hash = verify_protocol()
    cancer = load_breast_cancer()
    wine = load_wine()
    datasets = [
        _dataset_record("breast_cancer_wisconsin", np.asarray(cancer.data), np.asarray(cancer.target), seed=4201),
        _dataset_record("wine_recognition", np.asarray(wine.data), np.asarray(wine.target), seed=4202),
    ]
    manifest = {
        "schema_version": "1.0",
        "protocol_id": "FXAI-NEGATIVE-RESULTS-REMEDIATION",
        "protocol_sha256": protocol_hash,
        "commit": git_commit(),
        "status": "registered_formative_real_tabular_data",
        "datasets": [item[0] for item in datasets],
        "confirmatory_claims_enabled": False,
        "limitation": "Bundled datasets support implementation and formative H6 checks; they are not a new sealed independent confirmation.",
    }
    splits = {
        "schema_version": "1.0",
        "protocol_sha256": protocol_hash,
        "splits": [item[1] for item in datasets],
        "all_identity_overlap_count": 0,
        "test_labels_as_features": False,
    }
    write_json(ARTIFACTS / "data" / "dataset_manifest.json", manifest)
    write_json(ARTIFACTS / "data" / "split_manifest.json", splits)
    write_json(
        ARTIFACTS / "data" / "leakage_audit.json",
        {
            "status": "pass",
            "identity_overlap_count": 0,
            "test_labels_in_feature_manifest": False,
            "normalization_fit_scope": "train_only_when_models_are_fit",
            "policy_selection_scope": "OOF development only",
            "confirmatory_test_opened": False,
        },
    )
    # Keep the registered config immutable; measured manifests live under artifacts.
    registered = read_json(CONFIG / "negative_remediation_dataset_manifest.json")
    if registered["confirmatory_claims_enabled"]:
        raise RuntimeError("pre-run dataset registry unexpectedly enabled confirmatory claims")
    print(f"PASS remediation-data datasets={len(datasets)} confirmatory=false")


if __name__ == "__main__":
    main()
