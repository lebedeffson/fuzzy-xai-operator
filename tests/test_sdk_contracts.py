from __future__ import annotations

from fuzzyxai.sdk import BaseAdapter, ExplanationArtifact
from fuzzyxai.sdk.examples.medical_image_adapter import MedicalImageAdapter
from fuzzyxai.sdk.examples.simple_tabular_adapter import SimpleTabularAdapter


def test_base_adapter_examples_return_artifacts() -> None:
    adapter = SimpleTabularAdapter()
    assert isinstance(adapter, BaseAdapter)
    raw = adapter.explain({'payload': {'features': {'x1': 1.0, 'x2': 2.0}}})
    artifact = adapter.to_explanation_artifact(raw)
    assert isinstance(artifact, ExplanationArtifact)
    assert artifact.has_explanation_object is True
    assert artifact.registry_id == 'simple_tabular_example'


def test_medical_image_adapter_diagnostic_state() -> None:
    adapter = MedicalImageAdapter()
    raw = adapter.explain({'payload': {'image_id': 'img_1', 'source_conflict': True}})
    artifact = adapter.to_explanation_artifact(raw)
    assert artifact.has_diagnostic_state is True
    assert artifact.payload['image_id'] == 'img_1'
