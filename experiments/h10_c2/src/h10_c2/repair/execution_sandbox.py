from __future__ import annotations

import copy
from time import perf_counter
from typing import Any

from ..hashing import object_sha256


def _valid(route: dict[str, Any]) -> bool:
    return all(node["registered_attributes"] == node["observed_attributes"] for node in route["nodes"])


def execute_plan(
    observed_route: dict[str, Any],
    clean_provider_snapshot: dict[str, Any],
    actions: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    started = perf_counter()
    working = copy.deepcopy(observed_route)
    before_valid = _valid(working)
    before_hash = object_sha256(working)
    provider_nodes = {node["node_id"]: node for node in clean_provider_snapshot["nodes"]}
    audit = []
    human_actions = 0
    for index, action in enumerate(actions):
        target = str(action.get("target", ""))
        precondition = target in provider_nodes
        state_before = object_sha256(working)
        changed = False
        status = "not_executable"
        if precondition and action.get("operation") in {
            "restore_from_registered_provider",
            "restore_registered_preprocessing",
            "load_registered_model_version",
            "restore_artifact_by_verified_hash",
            "obtain_required_field_from_registered_source",
            "load_registered_dictionary",
            "rerun_explainer_with_registered_components",
            "refresh_calibration_artifact",
            "rebuild_reduction_with_registered_ceiling",
            "restore_registered_provenance",
        }:
            node = next((item for item in working["nodes"] if item["node_id"] == target), None)
            if node is not None:
                node["observed_attributes"] = copy.deepcopy(provider_nodes[target]["observed_attributes"])
                changed = True
                status = "completed"
        else:
            human_actions += 1
        audit.append(
            {
                "step": index,
                "target": target,
                "precondition_passed": precondition,
                "status": status,
                "changed": changed,
                "before_sha256": state_before,
                "after_sha256": object_sha256(working),
            }
        )
    after_valid = _valid(working)
    return {
        "route_valid_before": before_valid,
        "route_valid_after": after_valid,
        "full_recertification_success": after_valid and all(item["status"] == "completed" for item in audit),
        "partial_recovery": not after_valid and object_sha256(working) != before_hash,
        "new_critical_issues": int(before_valid and not after_valid),
        "human_actions": human_actions,
        "runtime_ms": (perf_counter() - started) * 1000,
        "audit": audit,
        "after_sha256": object_sha256(working),
    }

