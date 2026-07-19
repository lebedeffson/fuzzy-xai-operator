from __future__ import annotations

import numpy as np
import pytest

from fuzzyxai import FuzzyXAI


@pytest.mark.optional_integration
def test_keras_native_gradient_channel() -> None:
    tf = pytest.importorskip("tensorflow")
    model = tf.keras.Sequential([tf.keras.layers.Input((3,)), tf.keras.layers.Dense(2)])
    values = np.asarray([[0.2, -0.1, 0.7]], dtype=np.float32)
    result = FuzzyXAI.wrap(model).explain_one(values, feature_names=("a", "b", "c"))
    assert result.adapter.adapter_id == "keras_v2"
    assert result.model_evidence["contribution_method"] == "derived_native_input_gradient"
