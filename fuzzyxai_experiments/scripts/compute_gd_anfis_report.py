from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    rules = list(csv.DictReader((ROOT / 'data/fixtures/gd_anfis_rules.csv').open(encoding='utf-8')))
    shap_rows = list(csv.DictReader((ROOT / 'data/fixtures/gd_anfis_shap_values.csv').open(encoding='utf-8')))
    delta = round(max(float(r['u_k']) for r in rules) - min(float(r['u_k']) for r in rules), 6)
    i_pre = round(1 - sum(float(r['u_k']) for r in rules) / len(rules), 6)
    report = {
        'registry_id': 'gd_anfis_shap',
        'status': 'source-pending',
        'n_rules': len(rules),
        'alpha_k': {r['rule_id']: float(r['alpha_k']) for r in rules},
        'eta_k': {r['feature']: float(r['eta_k']) for r in shap_rows},
        'Delta': delta,
        'u_k': {r['rule_id']: float(r['u_k']) for r in rules},
        'I_pre': i_pre,
        'action': 'audit_report',
        'claim_scope': 'контрольный маршрут; качество исходной модели не заявляется',
    }
    out = ROOT / 'reports/chapter5/gd_anfis_shap_report.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(out)

if __name__ == '__main__':
    main()
