from __future__ import annotations

from hashlib import sha256

from fuzzyxai.audit_h10.models import RouteObservation


MANDATORY_FIELDS = (
    "artifact_sha256",
    "artifact_uri",
    "model_version",
    "explainer_version",
    "model_family",
    "explainer_model_family",
    "preprocessing_signature",
    "dictionary_version",
    "reference_population",
    "calibration_version",
    "deployment_context",
    "source_uri",
    "dependency_digest",
    "artifact_model_id",
    "model_id",
    "reduction_loss",
    "projection_type",
    "canonical_source_id",
)


def build_route(dataset_id: str, modality: str, object_id: str) -> RouteObservation:
    version = f"{dataset_id}-model-v19"
    digest = sha256(f"{dataset_id}:{object_id}".encode()).hexdigest()
    expected = {
        "artifact_sha256": digest,
        "artifact_uri": f"artifact://{dataset_id}/{object_id}",
        "model_version": version,
        "explainer_version": version,
        "model_family": f"{modality}-classifier",
        "explainer_model_family": f"{modality}-classifier",
        "preprocessing_signature": f"{modality}:load>normalize>predict",
        "dictionary_version": "dictionary-v19" if modality == "text" else "not-applicable",
        "reference_population": f"{dataset_id}:train",
        "calibration_version": f"{dataset_id}:calibration-v19",
        "calibration_age_days": 30.0,
        "deployment_context": f"{dataset_id}:default",
        "source_uri": f"dataset://{dataset_id}/{object_id}",
        "dependency_digest": digest[:32],
        "artifact_model_id": version,
        "model_id": version,
        "reduction_loss": 0.10,
        "projection_type": f"{modality}:registered-projection",
        "canonical_source_id": f"canonical:{object_id}",
    }
    paths = (
        ("artifact_sha256", "artifact_uri", "canonical_source_id"),
        ("model_version", "explainer_version", "model_family", "explainer_model_family"),
        ("preprocessing_signature", "dictionary_version", "dependency_digest"),
        ("reference_population", "calibration_version", "calibration_age_days", "deployment_context"),
        ("source_uri", "artifact_model_id", "model_id", "canonical_source_id"),
        ("reduction_loss", "projection_type", "canonical_source_id"),
    )
    costs = {field: 1.0 + (int(sha256(field.encode()).hexdigest()[:2], 16) % 5) * 0.25 for field in expected}
    return RouteObservation(
        f"route:{dataset_id}:{object_id}",
        dataset_id,
        modality,
        object_id,
        expected,
        dict(expected),
        MANDATORY_FIELDS,
        paths,
        costs,
    )
