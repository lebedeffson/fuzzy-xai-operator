from __future__ import annotations

import numpy as np
import pytest

from fuzzyxai import FuzzyXAI


@pytest.mark.optional_integration
def test_onnx_prediction_only_contract(tmp_path) -> None:
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    helper = onnx.helper
    tensor = onnx.TensorProto
    graph = helper.make_graph(
        [helper.make_node("Sigmoid", ["input"], ["probability"])],
        "fuzzyxai-smoke",
        [helper.make_tensor_value_info("input", tensor.FLOAT, [None, 1])],
        [helper.make_tensor_value_info("probability", tensor.FLOAT, [None, 1])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = min(model.ir_version, 9)
    path = tmp_path / "model.onnx"
    onnx.save(model, path)
    result = FuzzyXAI.wrap(path).explain_one(np.asarray([[0.4]], dtype=np.float32))
    assert result.adapter_id == "onnxruntime_v2"
    assert result.view_model.trace["adapter_capabilities"]["gradients"] is False
    assert result.model_evidence["contribution_method"] is None
