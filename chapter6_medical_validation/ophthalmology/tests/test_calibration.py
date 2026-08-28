import numpy as np

from chapter6_medical_validation.ophthalmology.src.calibration import fit_temperature, probabilities_from_logits


def test_temperature_is_fitted_only_from_registered_inputs_and_improves_nll():
    logits = np.asarray([[8.0, 0, 0, 0, 0], [0, 8.0, 0, 0, 0], [8.0, 0, 0, 0, 0], [0, 8.0, 0, 0, 0]])
    labels = np.asarray([0, 1, 1, 0])
    result = fit_temperature(logits, labels)
    assert result["status"] == "fitted_on_validation_only"
    assert result["calibrated_nll"] <= result["uncalibrated_nll"]
    probabilities = probabilities_from_logits(logits, float(result["temperature"]))
    assert np.allclose(probabilities.sum(axis=1), 1.0)
