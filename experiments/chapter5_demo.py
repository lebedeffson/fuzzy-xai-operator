from __future__ import annotations

import argparse
import json
from pathlib import Path

from fuzzyxai.risk import policy_from_calibration

from .chapter5_experiments import SCENARIOS, decision_for


def _scenario(name: str):
    for sc in SCENARIOS:
        if sc.scenario == name:
            return sc
    raise KeyError(name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-json', default='reports/chapter5/chapter5_experiments.json')
    parser.add_argument('--out', default='reports/chapter5/chapter5_demo.json')
    args = parser.parse_args()

    policy = policy_from_calibration(args.weights_json)
    weights = dict(policy.risk_weights)
    cases = [_scenario('S4'), _scenario('S0')]
    rows = []
    print('Chapter 5 calibrated observer demo')
    print('-' * 42)
    for sc in cases:
        action, rho = decision_for(sc, weights)
        row = {
            'scenario': sc.scenario,
            'uncertainty_type': sc.uncertainty_type,
            'representation': sc.representation,
            'rupture': sc.rupture,
            'rho': rho,
            'expected_action': sc.expected_action,
            'actual_action': action,
            'match': action == sc.expected_action,
        }
        rows.append(row)
        print(f"{sc.scenario}: rupture={sc.rupture} rho={rho:.4f} action={action} expected={sc.expected_action}")
    out = {'weights': weights, 'cases': rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'JSON: {args.out}')


if __name__ == '__main__':
    main()
