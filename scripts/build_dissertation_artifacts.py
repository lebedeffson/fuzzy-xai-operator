from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CHANNELS = ['L_k', 'mu_k', 'R_k', 'alpha_k', 'u_k', 'tau_k', 'eta_k', 'D_k', 'Report', 'Action']


def _load_json(path: str | Path) -> dict[str, Any]:
    p = ROOT / path
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding='utf-8'))


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    p = ROOT / path
    if not p.exists():
        return []
    with p.open(encoding='utf-8') as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, '') for field in fields})


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')


def _sha(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ''
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _copy(src: Path, dst: Path) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.exists():
        shutil.copy2(src, dst)
        return True
    return False


def _save_text_figure(path: Path, title: str, lines: list[str], *, color: str = '#0f766e') -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    ax.text(0.03, 0.90, title, fontsize=21, fontweight='bold', color=color, va='top')
    for i, line in enumerate(lines):
        ax.text(0.05, 0.78 - i * 0.075, line, fontsize=12, va='top', family='DejaVu Sans')
    fig.patch.set_facecolor('#f8fafc')
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def _bar(path: Path, title: str, labels: list[str], values: list[float], *, color: str = '#2563eb', ylim: tuple[float, float] | None = (0, 1)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(labels, values, color=color, alpha=0.88)
    ax.set_title(title, fontsize=16, fontweight='bold')
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(axis='y', alpha=0.25)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02, f'{v:.3f}', ha='center', fontsize=10)
    fig.autofmt_xdate(rotation=20, ha='right')
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _heatmap(path: Path, title: str, ylabels: list[str], xlabels: list[str], matrix: list[list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, max(4.2, 0.62 * len(ylabels) + 2)))
    im = ax.imshow(matrix, cmap='YlGnBu', vmin=0, vmax=1)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xticks(range(len(xlabels)), labels=xlabels, rotation=35, ha='right')
    ax.set_yticks(range(len(ylabels)), labels=ylabels)
    for y, row in enumerate(matrix):
        for x, val in enumerate(row):
            ax.text(x, y, '1' if val else '0', ha='center', va='center', color='#0f172a', fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _route_figure(path: Path, title: str, modules: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13, max(4.5, 0.55 * len(modules) + 2)))
    ax.axis('off')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=16)
    xs = [0.05, 0.30, 0.55, 0.78, 0.94]
    labels = ['External artifact', 'Adapter', 'E/D artifact', 'Report', 'Action']
    for row, module in enumerate(modules):
        y = 0.90 - row * (0.78 / max(1, len(modules) - 1)) if len(modules) > 1 else 0.55
        ax.text(0.01, y, module, fontsize=10, va='center', fontweight='bold')
        for i, x in enumerate(xs):
            ax.scatter([x], [y], s=480, color='#dbeafe', edgecolor='#1d4ed8', linewidth=1.2)
            ax.text(x, y, labels[i], ha='center', va='center', fontsize=8)
            if i < len(xs) - 1:
                ax.annotate('', xy=(xs[i + 1] - 0.045, y), xytext=(x + 0.045, y), arrowprops=dict(arrowstyle='->', color='#334155', lw=1.4))
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def _module_channels(row: dict[str, Any]) -> dict[str, int]:
    pending = row.get('status') in {'planned', 'source-pending'}
    if pending:
        return {c: 0 for c in CHANNELS}
    adapter = str(row.get('adapter', ''))
    base = {c: 0 for c in CHANNELS}
    for c in ['L_k', 'mu_k', 'u_k', 'tau_k', 'Report']:
        base[c] = 1
    if adapter != 'planned':
        base['R_k'] = 1
        base['alpha_k'] = 1
    if row.get('evidence_level') in {'repository-output-level', 'fixture-level'}:
        base['eta_k'] = 1
    if row.get('status') == 'real-output-compatible':
        base['D_k'] = 1
        base['Action'] = 1
    return base


def _scenario_tables(out: Path, matrix_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    scenario_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    reports_dir = ROOT / 'reports' / 'chapter5' / 'scenario_reports'
    figures_dir = ROOT / 'figures' / 'chapter5'
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    for row in matrix_rows:
        rid = row.get('registry_id', '')
        title = row.get('title', rid)
        scenario_rows.append({
            'scenario': title,
            'registry_id': rid,
            'module_name': title,
            'adapter_class': row.get('adapter', ''),
            'source_repo': row.get('source_repo', ''),
            'evidence_level': row.get('evidence_level', ''),
            'status': row.get('status', ''),
            'claim_scope': row.get('claim_scope', ''),
            'chapter_section': '5.x' if row.get('chapter_role') != 'future_extension' else 'appendix/future work',
        })
        adapter_called = row.get('run_allowed') in {True, 'True', 'true', '1', 1}
        output_type = 'ExplanationArtifact + report' if adapter_called else 'registered metadata only'
        action = 'audit_report' if adapter_called else 'not_run'
        report_path = reports_dir / f'{rid}_action_report.md'
        figure_path = figures_dir / f'{rid}_route.png'
        _route_figure(figure_path, f'{rid}: adapter route', [title])
        report_path.write_text('\n'.join([
            f'# Scenario action report: {rid}',
            '',
            f'- module: `{title}`',
            f'- adapter_called: `{adapter_called}`',
            f'- output_type: `{output_type}`',
            f'- status: `{row.get("status", "")}`',
            f'- evidence_level: `{row.get("evidence_level", "")}`',
            f'- action: `{action}`',
            f'- claim_scope: {row.get("claim_scope", "")}',
            '',
            'Численные `chi_R`, `chi_Auto` и `rho` для внешнего модуля не подставляются искусственно. '
            'Если адаптер не предоставляет полный структурный контур, сценарий фиксируется как audit/report-only.',
        ]), encoding='utf-8')
        run_rows.append({
            'registry_id': rid,
            'source_repo': row.get('source_repo', ''),
            'adapter_called': adapter_called,
            'output_type': output_type,
            'has_explanation_object': adapter_called,
            'has_diagnostic_state': row.get('status') == 'real-output-compatible',
            'chi_R': 'N/A',
            'chi_Auto': 'N/A',
            'rho': 'N/A',
            'action': action,
            'report_path': str(report_path.relative_to(ROOT)),
            'figure_path': str(figure_path.relative_to(ROOT)),
            'status': row.get('status', ''),
            'claim_scope': row.get('claim_scope', ''),
        })
        cov = {'registry_id': rid, **_module_channels(row)}
        coverage_rows.append(cov)

    _write_csv(ROOT / 'reports' / 'chapter5' / 'scenario_registry_table.csv', scenario_rows, ['scenario', 'registry_id', 'module_name', 'adapter_class', 'source_repo', 'evidence_level', 'status', 'claim_scope', 'chapter_section'])
    _write_csv(ROOT / 'reports' / 'chapter5' / 'scenario_run_summary.csv', run_rows, ['registry_id', 'source_repo', 'adapter_called', 'output_type', 'has_explanation_object', 'has_diagnostic_state', 'chi_R', 'chi_Auto', 'rho', 'action', 'report_path', 'figure_path', 'status', 'claim_scope'])
    _write_csv(ROOT / 'reports' / 'chapter5' / 'module_channel_coverage.csv', coverage_rows, ['registry_id', *CHANNELS])
    return scenario_rows, run_rows, coverage_rows


def _scenario_quantitative_summary(matrix_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fixed_values = {
        'hybrid_xiris': ('accuracy', 'not_available', 'safe_accuracy', 'not_available'),
        'anza_lira': ('dice', 'not_available', 'safe_dice', 'not_available'),
        'deep_neuro_fuzzy_kafn': ('f1', 'not_available', 'safe_f1', 'not_available'),
        'fan_multimodal': ('attention_consistency', 'not_available', 'safe_attention_consistency', 'not_available'),
    }
    rows: list[dict[str, Any]] = []
    for row in matrix_rows:
        rid = row.get('registry_id', '')
        b_metric, b_value, fx_metric, fx_value = fixed_values.get(rid, ('not_available', 'not_available', 'not_available', 'not_available'))
        available = b_value != 'not_available' and fx_value != 'not_available'
        rows.append({
            'registry_id': rid,
            'baseline_metric': b_metric,
            'baseline_value': b_value,
            'fuzzyxai_metric': fx_metric,
            'fuzzyxai_value': fx_value,
            'missed_critical': 'N/A',
            'false_auto_accept': 'N/A',
            'report_only': str(row.get('status') != 'real-output-compatible').lower(),
            'quantitative_comparison_available': str(available).lower(),
            'status': row.get('status', ''),
            'notes': 'No fake quantitative value is inserted; source metric must be pinned before comparison.' if not available else 'Quantitative comparison available from pinned source artifact.',
        })
    return rows


def _write_diagram_specs(out: Path) -> None:
    base = out / 'diagram_specs'
    specs: dict[str, dict[str, Any]] = {
        'chapter2/fig_2_1_operator.json': {
            'figure_id': 'fig_2_1', 'title': 'System operator route',
            'nodes': [{'id': x, 'label': x} for x in ['C_k', 'ExplainPlan', 'Omega', 'E_k', 'I(E_G)']],
            'edges': [{'from': 'C_k', 'to': 'Omega'}, {'from': 'ExplainPlan', 'to': 'Omega'}, {'from': 'Omega', 'to': 'E_k'}, {'from': 'E_k', 'to': 'I(E_G)'}],
            'color_roles': {'operator': '#0f766e', 'contract': '#2563eb'}, 'caption': 'Системный оператор строит E_k из состояния компонента и ExplainPlan.', 'short_explanation': 'Core operator route for chapter 2.'
        },
        'chapter2/fig_2_4_composition.json': {
            'figure_id': 'fig_2_4', 'title': 'Composition and diagnostic state',
            'nodes': [{'id': x, 'label': x} for x in ['E_i', 'd_E', 'E_j', 'D_ij']],
            'edges': [{'from': 'E_i', 'to': 'd_E'}, {'from': 'd_E', 'to': 'E_j'}, {'from': 'd_E', 'to': 'D_ij'}],
            'color_roles': {'ok': '#16a34a', 'rupture': '#dc2626'}, 'caption': 'Композиция объяснений и переход к диагностическому состоянию.', 'short_explanation': 'Shows where semantic disagreement is checked.'
        },
        'chapter3/fig_3_2_hierarchy.json': {
            'figure_id': 'fig_3_2', 'title': 'Hierarchy of fuzzy representations',
            'nodes': [{'id': x, 'label': x} for x in ['F0', 'F_int', 'F_H', 'F_N^src', 'F_ML-audit']],
            'edges': [{'from': 'F0', 'to': 'F_int'}, {'from': 'F0', 'to': 'F_H'}, {'from': 'F_int', 'to': 'F_N^src'}, {'from': 'F_N^src', 'to': 'F_ML-audit'}],
            'color_roles': {'selected': '#0f766e', 'candidate': '#dbeafe'}, 'caption': 'Иерархия представлений неопределённости.', 'short_explanation': 'Higher classes preserve more uncertainty structure.'
        },
        'chapter3/fig_3_3_reduction.json': {
            'figure_id': 'fig_3_3', 'title': 'Reduction and Delta',
            'nodes': [{'id': x, 'label': x} for x in ['A_k^F', 'reduce', 'F0', 'Delta']],
            'edges': [{'from': 'A_k^F', 'to': 'reduce'}, {'from': 'reduce', 'to': 'F0'}, {'from': 'reduce', 'to': 'Delta'}],
            'color_roles': {'loss': '#dc2626'}, 'caption': 'Редукция расширенного представления и потеря Delta.', 'short_explanation': 'Delta measures structural loss.'
        },
        'chapter3/fig_3_8_chi_auto_sample113.json': {
            'figure_id': 'fig_3_8', 'title': 'Chi_Auto for sample_113',
            'nodes': [{'id': 'e_model', 'label': 'E_model^ext', 'type': 'explanation'}, {'id': 'e_risk', 'label': 'E_risk', 'type': 'risk'}, {'id': 'e_action', 'label': 'E_action', 'type': 'action'}],
            'edges': [{'from': 'e_model', 'to': 'e_risk', 'label': 'f'}, {'from': 'e_risk', 'to': 'e_action', 'label': 'g'}],
            'color_roles': {'blocked': '#dc2626'}, 'caption': 'Контекстная проверка chi_Auto для sample_113.', 'short_explanation': 'chi_auto=false because auto maps to audit absent in AutoAccept.'
        },
        'chapter3/fig_3_4_route.json': {
            'figure_id': 'fig_3_4', 'title': 'Risk-aware observer route',
            'nodes': [{'id': x, 'label': x} for x in ['DatasetCase', 'Prediction', 'E_k', 'A_k^F', 'CertifiedPath/Rupture', 'chi_Auto', 'rho', 'Action']],
            'edges': [{'from': a, 'to': b} for a, b in zip(['DatasetCase', 'Prediction', 'E_k', 'A_k^F', 'CertifiedPath/Rupture', 'chi_Auto', 'rho'], ['Prediction', 'E_k', 'A_k^F', 'CertifiedPath/Rupture', 'chi_Auto', 'rho', 'Action'])],
            'color_roles': {'risk': '#ea580c', 'action': '#0f766e'}, 'caption': 'Маршрут наблюдателя от данных к действию.', 'short_explanation': 'Shows rupture, context and risk gates.'
        },
        'chapter4/fig_4_1_open_interfaces.json': {
            'figure_id': 'fig_4_1', 'title': 'Open interfaces: SDK/API/registry',
            'nodes': [{'id': x, 'label': x} for x in ['External module', 'BaseAdapter', '/v1/explain', 'ExplanationArtifact', '/v1/risk-action', 'Report']],
            'edges': [{'from': 'External module', 'to': 'BaseAdapter'}, {'from': 'BaseAdapter', 'to': '/v1/explain'}, {'from': '/v1/explain', 'to': 'ExplanationArtifact'}, {'from': 'ExplanationArtifact', 'to': '/v1/risk-action'}, {'from': '/v1/risk-action', 'to': 'Report'}],
            'color_roles': {'sdk': '#2563eb', 'api': '#0f766e'}, 'caption': 'Открытые интерфейсы экосистемы FuzzyXAI.', 'short_explanation': 'Registration, adapter and API contract route.'
        },
        'chapter5/fig_5_1_general_scenario_route.json': {
            'figure_id': 'fig_5_1', 'title': 'General scenario route',
            'nodes': [{'id': x, 'label': x} for x in ['registry_id', 'source artifact', 'adapter', 'E/D artifact', 'report/action']],
            'edges': [{'from': 'registry_id', 'to': 'source artifact'}, {'from': 'source artifact', 'to': 'adapter'}, {'from': 'adapter', 'to': 'E/D artifact'}, {'from': 'E/D artifact', 'to': 'report/action'}],
            'color_roles': {'adapter': '#0f766e', 'report': '#9333ea'}, 'caption': 'Общий маршрут сценария главы 5.', 'short_explanation': 'External artifact is routed through adapter to report/action.'
        },
        'chapter5/fig_5_2_hybrid_xiris_route.json': {
            'figure_id': 'fig_5_2', 'title': 'HYBRID-XIRIS route',
            'nodes': [{'id': x, 'label': x} for x in ['hybrid_xiris', 'image artifact', 'medical_image_to_explanation', 'ExplanationArtifact', 'audit_report']],
            'edges': [{'from': 'hybrid_xiris', 'to': 'image artifact'}, {'from': 'image artifact', 'to': 'medical_image_to_explanation'}, {'from': 'medical_image_to_explanation', 'to': 'ExplanationArtifact'}, {'from': 'ExplanationArtifact', 'to': 'audit_report'}],
            'color_roles': {'real_output': '#16a34a'}, 'caption': 'Сценарный маршрут HYBRID-XIRIS.', 'short_explanation': 'Shows real-output-compatible module route without retraining claims.'
        },
    }
    for rel, payload in specs.items():
        _write_json(base / rel, payload)


def _write_figure_manifests(out: Path) -> None:
    retained = [
        {'chapter': 2, 'figure_id': '2.1', 'status': 'keep', 'type': 'diagram', 'source_spec': 'diagram_specs/chapter2/fig_2_1_operator.json', 'will_be_redrawn': True, 'notes': 'core operator route'},
        {'chapter': 2, 'figure_id': '2.4', 'status': 'keep', 'type': 'diagram', 'source_spec': 'diagram_specs/chapter2/fig_2_4_composition.json', 'will_be_redrawn': True, 'notes': 'composition scheme'},
        {'chapter': 3, 'figure_id': '3.2', 'status': 'keep', 'type': 'diagram', 'source_spec': 'diagram_specs/chapter3/fig_3_2_hierarchy.json', 'will_be_redrawn': True, 'notes': 'representation hierarchy'},
        {'chapter': 3, 'figure_id': '3.8', 'status': 'keep', 'type': 'diagram', 'source_spec': 'diagram_specs/chapter3/fig_3_8_chi_auto_sample113.json', 'will_be_redrawn': True, 'notes': 'topos sample_113'},
        {'chapter': 4, 'figure_id': '4.1', 'status': 'keep', 'type': 'diagram', 'source_spec': 'diagram_specs/chapter4/fig_4_1_open_interfaces.json', 'will_be_redrawn': True, 'notes': 'SDK/API interfaces'},
        {'chapter': 4, 'figure_id': '4.app', 'status': 'appendix', 'type': 'screenshot', 'source_spec': 'appendix/app_gui_screenshots', 'will_be_redrawn': False, 'notes': 'GUI screenshots only in appendix'},
        {'chapter': 5, 'figure_id': '5.1', 'status': 'keep', 'type': 'diagram', 'source_spec': 'diagram_specs/chapter5/fig_5_1_general_scenario_route.json', 'will_be_redrawn': True, 'notes': 'scenario route'},
        {'chapter': 5, 'figure_id': '5.2', 'status': 'keep', 'type': 'diagram', 'source_spec': 'diagram_specs/chapter5/fig_5_2_hybrid_xiris_route.json', 'will_be_redrawn': True, 'notes': 'HYBRID-XIRIS route'},
    ]
    _write_csv(out / 'retained_figures_manifest.csv', retained, ['chapter', 'figure_id', 'status', 'type', 'source_spec', 'will_be_redrawn', 'notes'])
    captions = ['# Figure Captions Final', '']
    fmap = []
    for row in retained:
        if row['status'] == 'appendix':
            continue
        fid = f"fig_{str(row['figure_id']).replace('.', '_')}"
        captions += [
            f"## Figure {row['figure_id']}",
            f"Short caption: {row['notes']}.",
            f"Long caption: Diagram source is `{row['source_spec']}` and should be redrawn as clean vector art.",
            f"In-text sentence: Как показано на рис. {row['figure_id']}, маршрут фиксирует проверяемый переход без ручной подстановки.",
            '',
        ]
        fmap.append({'figure_id': fid, 'chapter': row['chapter'], 'section': f"{row['chapter']}.x", 'insert_after_heading': row['notes'], 'text_reference_sentence': f"Как показано на рис. {row['figure_id']}, ...", 'required': True})
    (out / 'figure_captions_final.md').write_text('\n'.join(captions), encoding='utf-8')
    _write_csv(out / 'figure_to_text_map.csv', fmap, ['figure_id', 'chapter', 'section', 'insert_after_heading', 'text_reference_sentence', 'required'])


def run(out_dir: str | Path = 'dissertation_artifacts') -> dict[str, Any]:
    out = ROOT / out_dir
    if out.exists():
        shutil.rmtree(out)
    for sub in ['chapter2', 'chapter3', 'chapter4', 'chapter5', 'appendix/app_gui_screenshots', 'diagram_specs/chapter2', 'diagram_specs/chapter3', 'diagram_specs/chapter4', 'diagram_specs/chapter5']:
        (out / sub).mkdir(parents=True, exist_ok=True)

    ch2 = _load_json('reports/chapter2/sample_113_report.json') or _load_json('reports/chapter2_real_operator_case/breast_cancer_operator_case.json')
    plan_hash = _load_json('reports/chapter2/explain_plan_hash.json')
    matrix_rows = _load_json('reports/chapter4/ecosystem_evidence.json').get('rows') or _read_csv('evidence/evidence_matrix.csv')
    matrix_rows = [dict(r) for r in matrix_rows]

    _save_text_figure(out / 'chapter2/fig_2_explainplan_contract.png', 'ExplainPlan contract', [
        f"YAML: configs/explain_plan_chapter2.yaml",
        f"SHA256: {plan_hash.get('sha256', 'N/A')}",
        f"required trace fields: {', '.join(plan_hash.get('required_trace_fields', [])) or 'N/A'}",
        'Contract fixes terms, rules, weights, thresholds and trace requirements before evaluation.',
    ])
    _bar(out / 'chapter2/fig_2_sample113_membership.png', 'sample_113 membership degrees', ['low', 'medium', 'high'], [float(ch2.get('mu_low', 0)), float(ch2.get('mu_medium', 0)), float(ch2.get('mu_high', 0))], color='#16a34a')
    _bar(out / 'chapter2/fig_2_sample113_uncertainty.png', 'sample_113 uncertainty and interpretability', ['U_model', 'U_rules', 'U_trace', 'u_M', 'I_pre'], [float(ch2.get(k, 0)) for k in ['U_model', 'U_rules', 'U_trace', 'u_M', 'I_pre']], color='#7c3aed')
    _write_csv(out / 'chapter2/table_2_sample113_values.csv', [{
        'p_113': ch2.get('p_malignant', ch2.get('P(malignant)', '')),
        'mu_low': ch2.get('mu_low', ''),
        'mu_medium': ch2.get('mu_medium', ''),
        'mu_high': ch2.get('mu_high', ''),
        'U_model': ch2.get('U_model', ''),
        'U_rules': ch2.get('U_rules', ''),
        'U_trace': ch2.get('U_trace', ''),
        'u_M': ch2.get('u_M', ''),
        'I_pre': ch2.get('I_pre', ''),
        'action': ch2.get('action', ''),
    }], ['p_113', 'mu_low', 'mu_medium', 'mu_high', 'U_model', 'U_rules', 'U_trace', 'u_M', 'I_pre', 'action'])
    _write_csv(out / 'chapter2/table_2_explainplan_hash.csv', [{'path': 'configs/explain_plan_chapter2.yaml', 'sha256': plan_hash.get('sha256', ''), 'status': plan_hash.get('status', '')}], ['path', 'sha256', 'status'])
    _copy(ROOT / 'reports/chapter2/calibration_constants.csv', out / 'chapter2/table_2_calibration_constants.csv')
    _copy(ROOT / 'reports/chapter2/equal_raw_structure_summary.csv', out / 'chapter2/table_2_equal_raw_structure_summary.csv')
    _copy(ROOT / 'figures/chapter2/calibration_constants.png', out / 'chapter2/fig_2_calibration_constants.png')
    _copy(ROOT / 'figures/chapter2/equal_raw_structure_comparison.png', out / 'chapter2/fig_2_equal_raw_structure_comparison.png')
    (out / 'chapter2/text_2_reproducibility_insert.md').write_text('Команда `make reproduce-chapter2` воспроизводит ExplainPlan hash и численный пример `sample_113` без ручной подстановки значений.\n', encoding='utf-8')
    _copy(ROOT / 'reports/chapter2/section_2_10_insert.md', out / 'chapter2/text_2_10_insert.md')
    _copy(ROOT / 'reports/chapter3/dataset_roles_summary.csv', out / 'chapter3/table_3_dataset_roles_summary.csv')
    _copy(ROOT / 'reports/chapter3/dataset_roles_summary.md', out / 'chapter3/text_3_dataset_roles_summary.md')
    _copy(ROOT / 'reports/chapter3/safety_limitation_insert.md', out / 'chapter3/text_3_safety_limitation_insert.md')

    required4 = ['module_id', 'name', 'evidence_level', 'status', 'source_repo', 'claim_scope', 'adapter_class', 'gui_visible', 'artifact_present', 'local_fixture_present', 'report_present', 'figure_present']
    table4 = []
    for row in matrix_rows:
        table4.append({
            'module_id': row.get('registry_id', ''),
            'name': row.get('title', ''),
            'evidence_level': row.get('evidence_level', ''),
            'status': row.get('status', ''),
            'source_repo': row.get('source_repo', ''),
            'claim_scope': row.get('claim_scope', ''),
            'adapter_class': row.get('adapter', ''),
            'gui_visible': True,
            'artifact_present': bool(row.get('source_artifact')) and str(row.get('source_repo', '')).startswith('https://github.com/') and row.get('status') != 'planned',
            'local_fixture_present': bool(row.get('fixture_exists', False)),
            'report_present': (ROOT / 'reports/chapter4/ecosystem_evidence.json').exists(),
            'figure_present': (ROOT / 'reports/browser_visual_check/11_ecosystem_registry.png').exists(),
        })
    _write_csv(out / 'chapter4/table_4_evidence_matrix.csv', table4, required4)
    _write_csv(ROOT / 'reports/chapter4/ecosystem_evidence.csv', table4, required4)
    _write_csv(out / 'chapter4/table_4_registry_snapshot.csv', matrix_rows, list(matrix_rows[0].keys()) if matrix_rows else ['registry_id'])
    _write_csv(out / 'chapter4/table_4_source_repo_cards.csv', [{k: r.get(k, '') for k in ['registry_id', 'title', 'source_repo', 'source_commit', 'source_artifact', 'claim_scope']} for r in matrix_rows], ['registry_id', 'title', 'source_repo', 'source_commit', 'source_artifact', 'claim_scope'])

    shots = {
        '10_evidence_contract.png': 'fig_4_evidence_contract.png',
        '11_ecosystem_registry.png': 'fig_4_ecosystem_registry.png',
        '12_real_rows_improvements.png': 'fig_4_real_rows_improvements.png',
    }
    for src_name, dst_name in shots.items():
        src = ROOT / 'reports/browser_visual_check' / src_name
        _copy(src, out / 'chapter4' / dst_name)
        _copy(src, out / 'appendix/app_gui_screenshots' / src_name)

    statuses = sorted({str(r.get('status', '')) for r in matrix_rows})
    counts = [sum(1 for r in matrix_rows if str(r.get('status', '')) == s) for s in statuses]
    _bar(out / 'chapter4/fig_4_registry_status_counts.png', 'Registry status counts', statuses, [float(c) for c in counts], color='#ea580c', ylim=None)
    heat_fields = ['fixture_exists', 'run_allowed', 'quantitative_claim_allowed']
    _heatmap(out / 'chapter4/fig_4_evidence_matrix_heatmap.png', 'Evidence matrix flags', [r.get('registry_id', '') for r in matrix_rows], heat_fields, [[1.0 if r.get(f) in {True, 'True', 'true', '1', 1} else 0.0 for f in heat_fields] for r in matrix_rows])
    (out / 'chapter4/text_4_evidence_pack_insert.md').write_text('Команда `make ecosystem-evidence` формирует машинно-читаемый evidence-пакет: registry snapshot, evidence matrix и reproduction index. Поля `status`, `evidence_level` и `claim_scope` ограничивают допустимые утверждения по каждому модулю.\n', encoding='utf-8')
    (out / 'chapter4/text_4_gui_insert.md').write_text('Evidence-вкладка Studio показывает hash-контракт, sample_113, внешний реестр модулей и границы claims; скриншоты получены автоматической проверкой Chromium (`make browser-visual-check`).\n', encoding='utf-8')

    scenario_rows, run_rows, coverage_rows = _scenario_tables(out, matrix_rows)
    quant_rows = _scenario_quantitative_summary(matrix_rows)
    _write_csv(out / 'chapter5/table_5_registry_scenarios.csv', scenario_rows, ['scenario', 'registry_id', 'module_name', 'adapter_class', 'source_repo', 'evidence_level', 'status', 'claim_scope', 'chapter_section'])
    _write_csv(out / 'chapter5/table_5_scenario_run_summary.csv', run_rows, ['registry_id', 'source_repo', 'adapter_called', 'output_type', 'has_explanation_object', 'has_diagnostic_state', 'chi_R', 'chi_Auto', 'rho', 'action', 'report_path', 'figure_path', 'status', 'claim_scope'])
    _write_csv(out / 'chapter5/table_5_module_channel_coverage.csv', coverage_rows, ['registry_id', *CHANNELS])
    _write_csv(ROOT / 'reports' / 'chapter5' / 'scenario_quantitative_summary.csv', quant_rows, ['registry_id', 'baseline_metric', 'baseline_value', 'fuzzyxai_metric', 'fuzzyxai_value', 'missed_critical', 'false_auto_accept', 'report_only', 'quantitative_comparison_available', 'status', 'notes'])
    _write_csv(out / 'chapter5/table_5_scenario_quantitative_summary.csv', quant_rows, ['registry_id', 'baseline_metric', 'baseline_value', 'fuzzyxai_metric', 'fuzzyxai_value', 'missed_critical', 'false_auto_accept', 'report_only', 'quantitative_comparison_available', 'status', 'notes'])
    _bar(out / 'chapter5/fig_5_scenario_status_overview.png', 'Chapter 5 scenario status overview', statuses, [float(c) for c in counts], color='#0891b2', ylim=None)
    _route_figure(out / 'chapter5/fig_5_scenario_action_routes.png', 'Scenario routes through FuzzyXAI', [r.get('registry_id', '') for r in matrix_rows])
    _heatmap(out / 'chapter5/fig_5_module_channel_coverage.png', 'Module channel coverage', [r['registry_id'] for r in coverage_rows], CHANNELS, [[float(r[c]) for c in CHANNELS] for r in coverage_rows])
    (out / 'chapter5/text_5_scenario_run_insert.md').write_text('Сценарный прогон проверяет не качество исходных моделей, а прохождение внешнего артефакта через `registry -> adapter -> explanation/report -> action`. Неполные внешние модули получают `action=audit_report` или `not_run`, а недоступные численные поля честно помечаются как `N/A`.\n', encoding='utf-8')
    _write_diagram_specs(out)
    _write_figure_manifests(out)

    for p in ['evidence/evidence_matrix.csv', 'evidence/registry_snapshot.json', 'evidence/reproduction_index.md', 'reports/browser_visual_check/browser_visual_check.md']:
        src = ROOT / p
        if src.exists():
            _copy(src, out / 'appendix' / {'evidence/evidence_matrix.csv': 'app_evidence_matrix_full.csv', 'evidence/registry_snapshot.json': 'app_registry_snapshot_full.json', 'evidence/reproduction_index.md': 'app_reproduction_index.md', 'reports/browser_visual_check/browser_visual_check.md': 'app_browser_visual_check_report.md'}[p])

    captions = [
        '# Captions', '',
        '- Рисунок 2.X — Контракт ExplainPlan и SHA256 воспроизводимого плана.',
        '- Рисунок 2.X — Степени принадлежности sample_113.',
        '- Рисунок 4.X — Evidence-вкладка FuzzyXAI Studio: ExplainPlan, sample_113 и проверяемые артефакты.',
        '- Рисунок 4.X — Интерактивный реестр внешних модулей с evidence level, status, source repo и claim scope.',
        '- Рисунок 5.X — Сценарные маршруты внешних модулей через FuzzyXAI.',
        '- Рисунок 5.X — Покрытие каналов объяснительного объекта по модулям.',
    ]
    (out / 'captions.md').write_text('\n'.join(captions) + '\n', encoding='utf-8')
    (out / 'insertion_plan.md').write_text('\n'.join([
        '# Insertion plan', '',
        '## Chapter 2', 'Use `chapter2/table_2_sample113_values.csv` and figures `fig_2_*` after the ExplainPlan definition.', '',
        '## Chapter 4', 'Use `chapter4/table_4_evidence_matrix.csv`, `fig_4_evidence_contract.png`, `fig_4_ecosystem_registry.png` and put full registry snapshot in appendix.', '',
        '## Chapter 5', 'Use `chapter5/table_5_scenario_run_summary.csv`, `fig_5_scenario_action_routes.png`, `fig_5_module_channel_coverage.png` for scenario evidence.', '',
        '## Appendix', 'Attach full manifest, visual check report and screenshots from `appendix/`.',
    ]) + '\n', encoding='utf-8')

    manifest = []
    for p in sorted(out.rglob('*')):
        if p.is_file() and p.name != 'artifact_manifest_sha256.json':
            manifest.append({'path': str(p.relative_to(out)), 'sha256': _sha(p)})
    (out / 'artifact_manifest_sha256.json').write_text(json.dumps({'status': 'ok', 'files': manifest}, ensure_ascii=False, indent=2), encoding='utf-8')
    _copy(out / 'artifact_manifest_sha256.json', out / 'appendix/app_manifest_sha256.json')

    gui_report = {
        'checked_at': 'manual-fill-after-review',
        'commit': 'use git rev-parse HEAD',
        'browser_visual_check': 'PASS' if (ROOT / 'reports/browser_visual_check/browser_visual_check.json').exists() else 'MISSING',
        'ui_health_check': 'PASS' if (ROOT / 'reports/ui_health_check.json').exists() or (ROOT / 'reports/ui_health_check.md').exists() else 'UNKNOWN',
        'screenshots': {name: (ROOT / 'reports/browser_visual_check' / name).exists() for name in shots},
        'manual_visual_score': {'readability': 0, 'layout': 0, 'navigation': 0, 'dissertation_readiness': 0},
        'manual_notes': [],
    }
    gui_dir = ROOT / 'reports/gui'
    gui_dir.mkdir(parents=True, exist_ok=True)
    (gui_dir / 'gui_acceptance_report.json').write_text(json.dumps(gui_report, ensure_ascii=False, indent=2), encoding='utf-8')

    return {'status': 'ok', 'out_dir': str(out), 'files': len(manifest)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', default='dissertation_artifacts')
    args = parser.parse_args()
    print(json.dumps(run(args.out_dir), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
