import pytest
from fuzzyxai.experiments.h10_c4 import build_scenarios, calibrate_runtime


@pytest.fixture(scope="session")
def development_scenarios():
    return build_scenarios(split="development", scenarios_per_family=4)


@pytest.fixture(scope="session")
def calibration(development_scenarios):
    return calibrate_runtime(development_scenarios)
