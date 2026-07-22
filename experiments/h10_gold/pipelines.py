from __future__ import annotations

import copy
import hashlib
from typing import Any


PIPELINES = (
    ("tabular_tree_a", "tabular", "tree_ensemble", "tree_shap"),
    ("tabular_boost_b", "tabular", "gradient_boosting", "kernel_shap"),
    ("text_transformer_a", "text", "transformer", "integrated_gradients"),
    ("text_encoder_b", "text", "encoder_classifier", "token_masking"),
    ("image_cnn_a", "image", "convolutional_network", "grad_cam"),
    ("timeseries_encoder_a", "time_series", "temporal_encoder", "window_masking"),
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def build_clean_graph(pipeline_id: str, modality: str, model_family: str, explainer_family: str) -> dict[str, Any]:
    preprocess_kind = {
        "text": "tokenizer",
        "image": "image_normalizer",
        "time_series": "window_builder",
    }.get(modality, "feature_preprocessor")
    node_specs = (
        ("source", "data_source", 1.0),
        ("preprocessor", preprocess_kind, 2.0),
        ("model", model_family, 8.0),
        ("explainer", explainer_family, 4.0),
        ("reference", "reference_population", 2.0),
        ("calibration", "calibration_artifact", 1.0),
        ("canonical", "canonical_explanation", 2.0),
        ("reducer", "representation_reducer", 2.0),
        ("output", "audit_output", 1.0),
    )
    nodes = []
    for node_id, kind, cost in node_specs:
        nodes.append(
            {
                "id": node_id,
                "kind": kind,
                "version": "registered-v1",
                "checksum": _digest(f"{pipeline_id}:{node_id}:registered-v1"),
                "model_binding": f"{pipeline_id}:model-v1",
                "run_id": f"{pipeline_id}:run-clean",
                "schema": f"{modality}:schema-v1",
                "reference_population": f"{pipeline_id}:reference-v1",
                "repair_cost": cost,
                "derived_status": "consistent",
            }
        )
    edges = [
        {"source": "source", "target": "preprocessor"},
        {"source": "preprocessor", "target": "model"},
        {"source": "model", "target": "explainer"},
        {"source": "reference", "target": "explainer"},
        {"source": "model", "target": "calibration"},
        {"source": "explainer", "target": "canonical"},
        {"source": "canonical", "target": "reducer"},
        {"source": "calibration", "target": "output"},
        {"source": "reducer", "target": "output"},
    ]
    return {
        "pipeline_id": pipeline_id,
        "modality": modality,
        "nodes": nodes,
        "edges": edges,
        "audit_paths": [
            ["source", "preprocessor", "model", "explainer", "canonical", "reducer", "output"],
            ["reference", "explainer", "canonical", "reducer", "output"],
            ["source", "preprocessor", "model", "calibration", "output"],
        ],
    }


def pipeline_graphs() -> dict[str, dict[str, Any]]:
    return {item[0]: build_clean_graph(*item) for item in PIPELINES}


def propagate_contract_symptoms(clean: dict[str, Any], mutated: dict[str, Any]) -> dict[str, Any]:
    """Materialize downstream symptoms without adding them to mutation truth."""
    from gold_oracle.graph_diff import diff_graphs

    output = copy.deepcopy(mutated)
    changed = set(diff_graphs(clean, mutated).changed_nodes)
    changed_ids = {item.split(":", 1)[1] for item in changed}
    successors: dict[str, set[str]] = {}
    for edge in clean["edges"]:
        successors.setdefault(edge["source"], set()).add(edge["target"])
    frontier = list(changed_ids)
    affected = set(changed_ids)
    while frontier:
        current = frontier.pop()
        for target in successors.get(current, ()):
            if target not in affected:
                affected.add(target)
                frontier.append(target)
    for node in output["nodes"]:
        if node["id"] in affected and node["id"] not in changed_ids:
            node["derived_status"] = "upstream_contract_unresolved"
    return output
