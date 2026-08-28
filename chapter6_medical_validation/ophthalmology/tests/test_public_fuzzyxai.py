from __future__ import annotations

import numpy as np
import torch
from fuzzyxai.adapters.optional_v2 import TorchAdapter

from chapter6_medical_validation.ophthalmology.src.evidence_adapter import MedicalRunFacts, export_public_result, run_public_explanation


class TinyEyeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(12, 5)

    def forward(self, value):
        return self.linear(value.flatten(1))


def test_public_explain_one_builds_system_and_typed_image_evidence(tmp_path):
    torch.manual_seed(7)
    model = TinyEyeModel()
    numeric = np.arange(12, dtype=np.float32).reshape(1, -1) / 12
    adapter = TorchAdapter(model, task="classification", ig_steps=8, input_transform=lambda value: torch.as_tensor(value, dtype=torch.float32).reshape(-1, 3, 2, 2))
    probabilities = tuple(float(value) for value in adapter.predict(numeric).probabilities[0])
    facts = MedicalRunFacts("fixture_eye", "test", "eye-1", "a" * 64, {"config_sha256": "b" * 64, "output_sha256": "c" * 64}, "tiny", "d" * 64, "run-1", probabilities, int(np.argmax(probabilities)), "not_calibrated", {"quality_score": 1.0, "blur_laplacian_variance": 10.0, "underexposure_fraction": 0.0, "overexposure_fraction": 0.0, "field_of_view_coverage": 1.0, "parameters": {}})
    result = run_public_explanation(model, adapter, numeric, np.zeros((2, 2, 3), dtype=np.uint8), facts, plan_path="chapter6_medical_validation/ophthalmology/configs/explain_plan_eye.yaml", grad_cam_map=np.ones((2, 2)), grad_cam_metadata={"target_class": facts.predicted_grade, "target_layer": "linear"})
    assert result.system is not None
    assert result.system.alignment["transform"]["transform_id"] == "ch6_eye_grade_probability_to_technical_risk_v1"
    assert result.system.risk.rho is not None
    payload = result.to_dict(detail="audit")
    assert payload["system_evidence"] is not None
    assert "optional_idrid_lesion_masks" in payload["layers"]["missing"]
    assert not result.view_model.quality_status.get("required_missing", [])
    export_public_result(result, tmp_path / "case")
    assert (tmp_path / "case" / "result.json").is_file()
    assert (tmp_path / "case" / "provenance_action.png").stat().st_size > 0


def test_registered_critical_fault_fails_closed():
    torch.manual_seed(8)
    model = TinyEyeModel(); numeric = np.ones((1, 12), dtype=np.float32)
    adapter = TorchAdapter(model, task="classification", ig_steps=4, input_transform=lambda value: torch.as_tensor(value, dtype=torch.float32).reshape(-1, 3, 2, 2))
    probabilities = tuple(float(value) for value in adapter.predict(numeric).probabilities[0])
    facts = MedicalRunFacts("fixture_eye", "test", "eye-2", "a" * 64, {"config_sha256": "b" * 64, "output_sha256": "c" * 64}, "tiny", "d" * 64, "run-2", probabilities, int(np.argmax(probabilities)), "not_calibrated", {"quality_score": 1.0, "blur_laplacian_variance": 10.0, "underexposure_fraction": 0.0, "overexposure_fraction": 0.0, "field_of_view_coverage": 1.0, "parameters": {}})
    result = run_public_explanation(model, adapter, numeric, np.zeros((2, 2, 3), dtype=np.uint8), facts, plan_path="chapter6_medical_validation/ophthalmology/configs/explain_plan_eye.yaml", grad_cam_map=np.ones((2, 2)), grad_cam_metadata={"target_class": facts.predicted_grade, "target_layer": "linear"}, critical_fault={"critical": True, "missing_required_trace": True, "code": "controlled_missing_trace"})
    assert result.system.risk.critical_override is True
    assert result.system.risk.action == "block"
