from __future__ import annotations

import argparse
import csv
import json
import platform
from pathlib import Path
from typing import Any

import sklearn

from experiments.chapter2_real_operator_case import generate_case
from fuzzyxai.core.explain_plan import hash_explain_plan, load_explain_plan, validate_explain_plan


DEFAULT_PLAN = Path('configs/explain_plan_chapter2.yaml')


def write_plan_hash(*, plan_path: str | Path = DEFAULT_PLAN, out_dir: str | Path = 'reports/chapter2') -> dict[str, Any]:
    plan = load_explain_plan(plan_path)
    validate_explain_plan(plan)
    payload = {
        'plan_path': str(plan_path),
        'sha256': hash_explain_plan(plan),
        'validated': True,
        'required_trace_fields': list(plan.get('trace_required', [])),
        'python': platform.python_version(),
        'sklearn': sklearn.__version__,
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'explain_plan_hash.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return payload


def write_sample113(*, out_dir: str | Path = 'reports/chapter2', plan_path: str | Path = DEFAULT_PLAN) -> dict[str, Any]:
    plan_hash = write_plan_hash(plan_path=plan_path, out_dir=out_dir)
    result = generate_case(out_dir='reports/chapter2_real_operator_case', target_probability=0.72)
    result = dict(result)
    tau = dict(result.get('tau_fields', {}))
    tau['timestamp'] = 'seeded-reproduction'
    tau['params'] = {'seed': 42, 'target_probability': 0.72}
    tau['hash'] = f"{tau.get('id', 'sample_113')}:{tau.get('version', 'rf_breast_cancer_v1')}:chapter2"
    result['tau_fields'] = tau
    result['explain_plan_hash'] = plan_hash['sha256']
    result['explain_plan_path'] = str(plan_path)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'sample_113_report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    (out / 'sample_113_trace.json').write_text(json.dumps(result.get('tau_fields', {}), ensure_ascii=False, indent=2), encoding='utf-8')

    table_path = out / 'sample_113_table.csv'
    fields = [
        'sample_id',
        'P(malignant)',
        'mu_low',
        'mu_medium',
        'mu_high',
        'U_model',
        'U_rules',
        'U_trace',
        'u_M',
        'I_pre',
        'gamma_model_to_risk',
        'chi_R',
        'chi_R_crit',
        'action',
        'explain_plan_hash',
    ]
    with table_path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow({k: result.get(k) for k in fields})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', default=str(DEFAULT_PLAN))
    parser.add_argument('--out-dir', default='reports/chapter2')
    parser.add_argument('--only-plan', action='store_true')
    args = parser.parse_args()
    if args.only_plan:
        result = write_plan_hash(plan_path=args.plan, out_dir=args.out_dir)
    else:
        result = write_sample113(out_dir=args.out_dir, plan_path=args.plan)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
