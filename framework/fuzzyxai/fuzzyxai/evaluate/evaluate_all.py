from __future__ import annotations

from fuzzyxai.evaluate.common import evaluate_control_model
from fuzzyxai.practice.fixtures import SCENARIOS


def main() -> None:
    for scenario_id in SCENARIOS:
        evaluate_control_model(scenario_id)
        print(f"evaluated: {scenario_id}")


if __name__ == "__main__":
    main()

