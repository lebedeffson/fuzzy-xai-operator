from __future__ import annotations

from fuzzyxai import FuzzyXAI
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def _wrapped_result_with_raw_text(raw_text: str):
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
    model = LogisticRegression(max_iter=2000).fit(X_train, y_train)
    fx = FuzzyXAI.wrap(model, adapter="auto", task="classification")
    return fx.explain_one(X_test[0], object_id="p0", raw_object=raw_text)


def test_object_representation_property_matches_visual_spec() -> None:
    result = _wrapped_result_with_raw_text("feature_0 and feature_1 are elevated")
    assert result.object_representation is not None
    assert result.object_representation == result.view_model.visual_spec.get("object_representation")


def test_export_json_redacts_raw_text_by_default() -> None:
    secret = "SECRET_PATIENT_NOTE_1234"
    result = _wrapped_result_with_raw_text(f"feature_0 is elevated per {secret}")
    default_json = result.view_model.to_json()
    assert secret not in default_json
    assert "omitted by default" in default_json


def test_export_json_include_raw_true_keeps_raw_text() -> None:
    secret = "SECRET_PATIENT_NOTE_1234"
    result = _wrapped_result_with_raw_text(f"feature_0 is elevated per {secret}")
    full_json = result.view_model.to_json(include_raw=True)
    assert secret in full_json


def test_export_json_to_file_respects_include_raw(tmp_path) -> None:
    secret = "SECRET_PATIENT_NOTE_1234"
    result = _wrapped_result_with_raw_text(f"feature_0 is elevated per {secret}")
    default_path = result.export_json(tmp_path / "default.json")
    assert secret not in default_path.read_text(encoding="utf-8")
    full_path = result.export_json(tmp_path / "full.json", include_raw=True)
    assert secret in full_path.read_text(encoding="utf-8")


def test_redaction_only_touches_text_modality_raw_fields() -> None:
    # Structured spans (offsets/feature names/weights) are not raw free text
    # and must survive redaction — only the full raw string is stripped.
    result = _wrapped_result_with_raw_text("feature_0 SECRET_MARKER feature_1")
    default_dict = result.to_dict()
    representation = default_dict["visual_spec"]["object_representation"]
    assert representation["modality"] == "text"
    assert representation["spans"], "spans (structured offsets) must remain present after redaction"
    assert "SECRET_MARKER" not in str(representation["raw_excerpt"])
    assert "SECRET_MARKER" not in str(representation["highlighted_html"])
