from fuzzyxai.repair import CostWeights, action_cost


def test_four_cost_models_are_explicit(development_scenarios, calibration) -> None:
    action = development_scenarios[0].actions[0]
    values = {
        model: action_cost(action, model, calibration, CostWeights())
        for model in ("uniform", "runtime", "dependency_weighted", "hybrid")
    }

    assert values["uniform"] == 1.0
    assert all(value > 0 for value in values.values())
    assert values["hybrid"] != values["uniform"]
