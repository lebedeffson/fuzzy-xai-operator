from __future__ import annotations

import numpy as np
import pytest

from fuzzyxai import FuzzyXAI
from fuzzyxai.adapter_conformance import run_adapter_conformance


@pytest.mark.optional_integration
def test_keras_native_gradient_channel() -> None:
    tf = pytest.importorskip("tensorflow")
    model = tf.keras.Sequential([tf.keras.layers.Input((3,)), tf.keras.layers.Dense(2)])
    values = np.asarray([[0.2, -0.1, 0.7]], dtype=np.float32)
    fx = FuzzyXAI.wrap(model)
    result = fx.explain_one(values, feature_names=("a", "b", "c"))
    assert result.adapter_id == "keras_v2"
    assert result.model_evidence["contribution_method"] == "derived_native_input_gradient"
    assert run_adapter_conformance(fx.model_adapter, sample_batch=values).status == "pass"
