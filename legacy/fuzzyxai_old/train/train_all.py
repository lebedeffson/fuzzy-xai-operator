from __future__ import annotations

from fuzzyxai.practice.fixtures import SCENARIOS
from fuzzyxai.train.common import train_control_model


def main() -> None:
    for scenario_id in SCENARIOS:
        train_control_model(scenario_id)
        print(f"trained: {scenario_id}")


if __name__ == "__main__":
    main()

