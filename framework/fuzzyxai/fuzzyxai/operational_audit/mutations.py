from __future__ import annotations

from dataclasses import replace

from .contracts import RouteArtifact


def mutate_route_artifact(value: RouteArtifact, family: str, severity: str) -> RouteArtifact:
    if severity not in {"subtle", "moderate", "severe"}:
        raise ValueError("unknown mutation severity")
    suffix = {"subtle": "patch", "moderate": "minor", "severe": "foreign"}[severity]
    changes: dict[str, object] = {"artifact_id": f"{value.artifact_id}:{family}:{severity}"}
    if family == "model_explainer_mismatch":
        changes["explainer_model_id"] = f"{value.model_id}-{suffix}"
    elif family == "stale_calibration":
        changes["calibration_model_id"] = f"{value.model_id}-{suffix}"
    elif family == "preprocessing_order_change":
        steps = list(value.preprocessing_steps)
        steps[-2:] = reversed(steps[-2:])
        changes["preprocessing_steps"] = tuple(steps)
    elif family == "feature_schema_incompatibility":
        schema = list(value.feature_schema)
        schema[-1] = f"{schema[-1]}-{suffix}"
        changes["explainer_feature_schema"] = tuple(schema)
    elif family == "cross_model_artifact_mix":
        changes.update(explainer_model_id=f"explainer-{suffix}", calibration_model_id=f"calibration-{suffix}")
    elif family == "checksum_corruption":
        changes["observed_sha256"] = (value.observed_sha256[:-1] + suffix[0])
    elif family == "reduction_link_loss":
        changes["reduction_target_source_id"] = f"reduction-{suffix}"
    elif family == "reference_population_substitution":
        changes["reference_population_id"] = f"reference-{suffix}"
    elif family == "partial_provenance_deletion":
        drop = {"subtle": 1, "moderate": 2, "severe": len(value.provenance_nodes)}[severity]
        changes["provenance_nodes"] = value.provenance_nodes[:-drop]
    elif family == "dictionary_or_tokenizer_version_change":
        changes["dictionary_version"] = f"dictionary-{suffix}"
    else:
        raise ValueError(f"unknown mutation family: {family}")
    return replace(value, **changes)
