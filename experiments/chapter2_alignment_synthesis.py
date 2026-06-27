from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fuzzyxai.core.alignment_synthesis import write_alignment_report
from fuzzyxai.core.explanation_object import ExplanationObject, Rule, Trace
from fuzzyxai.hierarchy.f0 import F0

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / 'reports' / 'chapter2' / 'alignment_synthesis_report.json'


def _E(label: str, activation: float) -> ExplanationObject:
    return ExplanationObject(
        terms={'low', 'high'},
        representation=F0(lambda _x, val=activation: val, label=label),
        rules=[Rule('r_high', {'risk': 'high'}, 'audit')],
        activations={'r_high': activation},
        uncertainty=0.2,
        trace=Trace(label, 'v1', datetime.now(timezone.utc).isoformat()),
        reduction_loss=0.05,
    )


def run() -> dict[str, Any]:
    plan = {
        'alignment_synthesis': {
            'gamma_max': 0.4,
            'lambda_delta': 0.2,
            'candidates': [
                {'name': 'identity', 'term_map': {'low': 'low', 'high': 'high'}, 'rule_map': {'r_high': 'r_high'}}
            ],
        }
    }
    return write_alignment_report(REPORT_PATH, _E('E_i', 0.7), _E('E_j', 0.72), plan)


if __name__ == '__main__':
    print(json.dumps(run(), ensure_ascii=False, indent=2))
