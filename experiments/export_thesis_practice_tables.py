from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from apps.chapter5_web_demo import build_backend


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding='utf-8'))


def _tex_escape(text: str) -> str:
    return (
        text.replace('\\', '\\textbackslash{}')
        .replace('_', '\\_')
        .replace('%', '\\%')
        .replace('&', '\\&')
        .replace('#', '\\#')
    )


def _fmt(v: Any) -> str:
    if v is None:
        return 'N/A'
    if isinstance(v, float):
        return f'{v:.6f}'.rstrip('0').rstrip('.')
    return str(v)


def _latex_table(headers: list[str], rows: list[list[Any]], caption: str, label: str) -> str:
    cols = '|'.join(['l'] * len(headers))
    out = [
        '\\begin{table}[h]',
        '\\centering',
        f'\\caption{{{_tex_escape(caption)}}}',
        f'\\label{{{_tex_escape(label)}}}',
        f'\\begin{{tabular}}{{|{cols}|}}',
        '\\hline',
        ' & '.join(_tex_escape(h) for h in headers) + ' \\\\',
        '\\hline',
    ]
    for row in rows:
        out.append(' & '.join(_tex_escape(_fmt(v)) for v in row) + ' \\\\')
        out.append('\\hline')
    out += ['\\end{tabular}', '\\end{table}', '']
    return '\n'.join(out)


def export(*, out_dir: str | Path = 'reports/thesis_tables') -> dict[str, str]:
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)

    summary = _load_json(Path('reports/datasets/breast_cancer/summary.json'))
    calibration = _load_json(Path('reports/datasets/breast_cancer/calibration.json'))
    ablation = _load_json(Path('reports/datasets/breast_cancer/ablation.json'))
    baseline = _load_json(Path('reports/datasets/breast_cancer/baseline_comparison.json'))
    baseline_diabetes = _load_json(Path('reports/datasets/diabetes_binary/baseline_comparison.json'))
    baseline_wine = _load_json(Path('reports/datasets/wine_risk/baseline_comparison.json'))
    baseline_synth = _load_json(Path('reports/datasets/synthetic_ruptures/baseline_comparison.json'))
    baseline_synth_native = _load_json(Path('reports/datasets/synthetic_ruptures/baseline_comparison_native.json'))
    baseline_synth_equal = _load_json(Path('reports/datasets/synthetic_ruptures/baseline_comparison_equal_guardrail.json'))
    structure = _load_json(Path('reports/structure_aware_benchmark/breast_cancer.json'))
    structure_wine = _load_json(Path('reports/structure_aware_benchmark/wine_risk.json'))
    structure_diabetes = _load_json(Path('reports/structure_aware_benchmark/diabetes_binary.json'))
    defense = _load_json(Path('reports/defense_cases/summary.json'))
    backend = build_backend()
    plan = backend.plan

    quant_rows = [
        ['I_pre', summary.get('i_pre_mean'), summary.get('i_pre_std'), summary.get('i_pre_median'), summary.get('i_pre_p25'), summary.get('i_pre_p75')],
        ['rho', summary.get('rho_mean'), summary.get('rho_std'), summary.get('rho_median'), summary.get('rho_p25'), summary.get('rho_p75')],
    ]
    (root / 'table_breast_cancer_quantiles.tex').write_text(
        _latex_table(
            ['metric', 'mean', 'std', 'median', 'p25', 'p75'],
            quant_rows,
            'Breast Cancer quantiles for I_pre and rho',
            'tab:breast_quantiles',
        ),
        encoding='utf-8',
    )

    before = calibration.get('before_calibration', {})
    after = calibration.get('after_calibration', {})
    calib_rows = [
        ['before', before.get('agreement_proxy'), before.get('agreement_reference'), before.get('missed_critical_ruptures'), before.get('false_auto_accept_rate')],
        ['after', after.get('agreement_proxy'), after.get('agreement_reference'), after.get('missed_critical_ruptures'), after.get('false_auto_accept_rate')],
    ]
    (root / 'table_observer_calibration.tex').write_text(
        _latex_table(
            ['mode', 'agreement_proxy', 'agreement_reference', 'missed_critical_ruptures', 'false_auto_accept_rate'],
            calib_rows,
            'Observer calibration comparison on breast_cancer',
            'tab:observer_calibration',
        ),
        encoding='utf-8',
    )

    constrained_rows = [
        ['objective', calibration.get('objective')],
        ['constraint_missed_critical', (calibration.get('constraints') or {}).get('missed_critical_ruptures')],
        ['constraint_critical_recall', (calibration.get('constraints') or {}).get('critical_rupture_recall')],
        ['constraint_false_auto_accept_max', (calibration.get('constraints') or {}).get('false_auto_accept_rate_max')],
    ]
    (root / 'table_constrained_calibration.tex').write_text(
        _latex_table(
            ['item', 'value'],
            constrained_rows,
            'Constrained calibration objective and safety constraints',
            'tab:constrained_calibration',
        ),
        encoding='utf-8',
    )

    abl_rows = [
        [
            row.get('mode'),
            row.get('agreement_proxy'),
            row.get('missed_critical_ruptures'),
            row.get('false_auto_accept_rate'),
            row.get('false_block_rate'),
            row.get('auto_accept_coverage'),
            row.get('mean_reduction_loss'),
        ]
        for row in ablation.get('rows', [])
    ]
    (root / 'table_ablation.tex').write_text(
        _latex_table(
            ['mode', 'agreement_proxy', 'missed_critical', 'false_auto_accept', 'false_block', 'auto_accept_cov', 'mean_Delta'],
            abl_rows,
            'Ablation benchmark for observer safety metrics',
            'tab:ablation',
        ),
        encoding='utf-8',
    )

    baseline_rows = [
        [
            row.get('baseline'),
            row.get('information_access'),
            row.get('agreement_proxy'),
            row.get('agreement_reference'),
            row.get('missed_critical_ruptures'),
            row.get('false_auto_accept_rate'),
            row.get('false_block_rate'),
            row.get('auto_accept_coverage'),
        ]
        for row in baseline.get('rows', [])
    ]
    (root / 'table_baseline_comparison.tex').write_text(
        _latex_table(
            ['baseline', 'information_access', 'agreement_proxy', 'agreement_reference', 'missed_critical', 'false_auto_accept', 'false_block', 'auto_accept_cov'],
            baseline_rows,
            'Baseline comparison (full observer vs threshold vs SHAP/LIME/Anchors guardrails)',
            'tab:baseline_comparison',
        ),
        encoding='utf-8',
    )

    def _pick_row(payload: dict[str, Any], name: str) -> dict[str, Any]:
        for row in payload.get('rows', []):
            if row.get('baseline') == name:
                return row
        return {}

    by_dataset_rows = []
    dataset_payloads = [
        ('breast_cancer', baseline),
        ('diabetes_binary', baseline_diabetes),
        ('wine_risk', baseline_wine),
        ('synthetic_ruptures', baseline_synth),
    ]
    for ds, payload in dataset_payloads:
        row_full = _pick_row(payload, 'full_observer_calibrated')
        row_thr = _pick_row(payload, 'probability_threshold')
        by_dataset_rows.append([
            ds,
            row_full.get('agreement_reference'),
            row_full.get('false_auto_accept_rate'),
            row_full.get('critical_rupture_recall'),
            row_thr.get('agreement_reference'),
            row_thr.get('false_auto_accept_rate'),
            row_thr.get('critical_rupture_recall'),
        ])
    (root / 'table_baseline_comparison_by_dataset.tex').write_text(
        _latex_table(
            [
                'dataset',
                'full_agreement_reference',
                'full_false_auto_accept',
                'full_critical_recall',
                'threshold_agreement_reference',
                'threshold_false_auto_accept',
                'threshold_critical_recall',
            ],
            by_dataset_rows,
            'Baseline comparison by dataset (full calibrated observer vs probability threshold)',
            'tab:baseline_by_dataset',
        ),
        encoding='utf-8',
    )

    structure_rows = [
        [
            row.get('policy'),
            row.get('agreement_reference'),
            row.get('missed_critical_ruptures'),
            row.get('critical_rupture_recall'),
            row.get('false_auto_accept_rate'),
            row.get('false_block_rate'),
        ]
        for row in structure.get('rows', [])
    ]
    (root / 'table_structure_aware_benchmark.tex').write_text(
        _latex_table(
            ['policy', 'agreement_reference', 'missed_critical', 'critical_recall', 'false_auto_accept', 'false_block'],
            structure_rows,
            'Structure-aware benchmark (real rows + controlled perturbations)',
            'tab:structure_aware',
        ),
        encoding='utf-8',
    )

    def _policy_row(payload: dict[str, Any], name: str) -> dict[str, Any]:
        for row in payload.get('rows', []):
            if row.get('policy') == name or row.get('baseline') == name:
                return row
        return {}

    ns_rows: list[list[Any]] = []
    for ds_name, payload in (
        ('breast_cancer', structure),
        ('wine_risk', structure_wine),
        ('diabetes_binary', structure_diabetes),
    ):
        full = _policy_row(payload, 'full_observer_calibrated')
        thr = _policy_row(payload, 'probability_threshold')
        if not full or not thr:
            continue
        full_far = float(full.get('false_auto_accept_rate', 0.0))
        thr_far = float(thr.get('false_auto_accept_rate', 0.0))
        full_agr = float(full.get('agreement_reference', 0.0))
        thr_agr = float(thr.get('agreement_reference', 0.0))
        ns_rows.append([
            ds_name,
            full_agr,
            thr_agr,
            full_agr - thr_agr,
            full_far,
            thr_far,
            thr_far - full_far,
            full.get('critical_rupture_recall'),
            full.get('missed_critical_ruptures'),
        ])
    (root / 'table_non_synthetic_improvements.tex').write_text(
        _latex_table(
            [
                'dataset',
                'full_agreement_ref',
                'threshold_agreement_ref',
                'agreement_gain',
                'full_false_auto_accept',
                'threshold_false_auto_accept',
                'false_auto_accept_drop',
                'critical_recall',
                'missed_critical',
            ],
            ns_rows,
            'Non-synthetic structure-aware improvements on real datasets',
            'tab:non_synthetic_improvements',
        ),
        encoding='utf-8',
    )

    def _rows_by_mode(payload: dict[str, Any], mode: str) -> list[list[Any]]:
        out_rows: list[list[Any]] = []
        for row in payload.get('rows', []):
            if row.get('baseline', '').startswith('full_observer') or row.get('baseline') in {
                'probability_threshold', 'shap_guardrail', 'lime_guardrail', 'anchor_guardrail'
            }:
                out_rows.append([
                    mode,
                    row.get('baseline'),
                    row.get('information_access'),
                    row.get('missed_critical_ruptures'),
                    row.get('critical_rupture_recall'),
                    row.get('false_auto_accept_rate'),
                    row.get('agreement_reference'),
                ])
        return out_rows

    synth_mode_rows = _rows_by_mode(
        baseline_synth_native if baseline_synth_native else baseline_synth, 'native'
    ) + _rows_by_mode(baseline_synth_equal, 'equal_guardrail')
    (root / 'table_synthetic_guardrail_modes.tex').write_text(
        _latex_table(
            [
                'baseline_access',
                'baseline',
                'information_access',
                'missed_critical',
                'critical_recall',
                'false_auto_accept',
                'agreement_reference',
            ],
            synth_mode_rows,
            'Synthetic ruptures: native vs equal-guardrail baseline access',
            'tab:synthetic_guardrail_modes',
        ),
        encoding='utf-8',
    )

    case_rows = [
        [
            c.get('case'),
            c.get('P'),
            c.get('representation'),
            c.get('delta'),
            c.get('I_pre'),
            c.get('rho'),
            c.get('chi_R'),
            c.get('chi_R_crit'),
            c.get('chi_Auto'),
            c.get('action'),
        ]
        for c in defense.get('cases', [])
    ]
    (root / 'table_defense_cases.tex').write_text(
        _latex_table(
            ['case', 'P', 'repr', 'Delta', 'I_pre', 'rho', 'chi_R', 'chi_R_crit', 'chi_Auto', 'action'],
            case_rows,
            'Three end-to-end defense cases',
            'tab:defense_cases',
        ),
        encoding='utf-8',
    )

    params_rows = [
        ['gamma_max', 0.45, 'certified alignment', 'fixed before experiments'],
        ['I_min', plan.i_min, 'interpretability floor', 'ExplainPlan baseline'],
        ['theta_1...theta_4', list(backend.observer.thresholds), 'risk observer', 'calibration'],
        ['w_p,w_u,w_I,w_Delta,w_R', backend.observer.weights, 'rho aggregation', 'calibration'],
        ['eta_1,eta_2,eta_3', plan.eta, 'u_M decomposition', 'ExplainPlan'],
        ['epsilon_int', plan.epsilon, 'numeric tolerance', 'fixed'],
        ['lambda_Delta', plan.beta.get('reduction', 0.10), 'reduction penalty', 'fixed'],
    ]
    (root / 'table_explainplan_params.tex').write_text(
        _latex_table(
            ['parameter', 'value', 'where_used', 'selection'],
            params_rows,
            'ExplainPlan and observer parameters',
            'tab:explainplan_params',
        ),
        encoding='utf-8',
    )

    return {
        'quantiles': str(root / 'table_breast_cancer_quantiles.tex'),
        'calibration': str(root / 'table_observer_calibration.tex'),
        'constrained_calibration': str(root / 'table_constrained_calibration.tex'),
        'ablation': str(root / 'table_ablation.tex'),
        'baseline_comparison': str(root / 'table_baseline_comparison.tex'),
        'baseline_by_dataset': str(root / 'table_baseline_comparison_by_dataset.tex'),
        'structure_aware_benchmark': str(root / 'table_structure_aware_benchmark.tex'),
        'non_synthetic_improvements': str(root / 'table_non_synthetic_improvements.tex'),
        'synthetic_guardrail_modes': str(root / 'table_synthetic_guardrail_modes.tex'),
        'defense_cases': str(root / 'table_defense_cases.tex'),
        'params': str(root / 'table_explainplan_params.tex'),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', default='reports/thesis_tables')
    args = parser.parse_args()
    out = export(out_dir=args.out_dir)
    print(json.dumps({'status': 'ok', 'files': out}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
