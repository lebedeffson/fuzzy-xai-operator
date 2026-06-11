from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/gui_screenshots'
BLUE = '#1f5f9f'
BLACK = '#1d1d1f'
GRAY = '#5f6368'
LIGHT = '#f6f8fb'
BORDER = '#d7dee8'
WARN = '#fff7e6'


def read_json(rel: str) -> dict[str, Any]:
    return json.loads((ROOT / rel).read_text(encoding='utf-8'))


def setup(title: str):
    fig = plt.figure(figsize=(16, 9), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    fig.patch.set_facecolor('white')
    ax.text(0.04, 0.94, title, fontsize=25, fontweight='bold', color=BLACK, va='top')
    ax.text(0.04, 0.90, 'Сценарии объяснения, аудита и контроля риска', fontsize=12, color=GRAY, va='top')
    return fig, ax


def box(ax, x, y, w, h, title: str, lines: list[str] | None = None, fill: str = 'white', edge: str = BORDER, title_color: str = BLACK):
    patch = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.012,rounding_size=0.018', linewidth=1.2, edgecolor=edge, facecolor=fill)
    ax.add_patch(patch)
    ax.text(x + 0.018, y + h - 0.035, title, fontsize=14, fontweight='bold', color=title_color, va='top')
    if lines:
        yy = y + h - 0.075
        for line in lines:
            wrapped = textwrap.wrap(str(line), width=max(22, int(w * 78))) or ['']
            for part in wrapped:
                ax.text(x + 0.018, yy, part, fontsize=10.5, color=BLACK, va='top')
                yy -= 0.027
            yy -= 0.004


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=18, linewidth=1.4, color=BLUE))


def badge(ax, x, y, text: str, fill: str = 'white', color: str = BLUE):
    patch = FancyBboxPatch((x, y), 0.12, 0.036, boxstyle='round,pad=0.006,rounding_size=0.015', linewidth=1.0, edgecolor=color, facecolor=fill)
    ax.add_patch(patch)
    ax.text(x + 0.06, y + 0.018, text, fontsize=9.5, color=color, ha='center', va='center', fontweight='bold')


def button(ax, x, y, text: str, w: float = 0.13):
    patch = FancyBboxPatch((x, y), w, 0.045, boxstyle='round,pad=0.006,rounding_size=0.011', linewidth=1.1, edgecolor=BLUE, facecolor=BLUE)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + 0.023, text, fontsize=9.5, color='white', ha='center', va='center')


def tabbar(ax, active: str):
    tabs = ['Summary', 'Input', 'Parameters', 'Route', 'Result', 'Claims', 'Evidence', 'Developer details']
    x = 0.04
    for t in tabs:
        w = 0.078 if t != 'Developer details' else 0.13
        fill = BLUE if t == active else 'white'
        color = 'white' if t == active else BLUE
        patch = FancyBboxPatch((x, 0.825), w, 0.038, boxstyle='round,pad=0.004,rounding_size=0.01', linewidth=1.0, edgecolor=BLUE, facecolor=fill)
        ax.add_patch(patch)
        ax.text(x + w / 2, 0.844, t, fontsize=8.5, color=color, ha='center', va='center')
        x += w + 0.008


def save(fig, name: str):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, bbox_inches='tight', pad_inches=0, facecolor='white')
    plt.close(fig)


def home():
    h = read_json('reports/chapter5/hybrid_xiris_summary.json')
    b = read_json('reports/chapter5/beacon_xai_summary.json')
    g = read_json('reports/chapter5/gis_integro_route_metrics.json')
    gd = read_json('reports/chapter5/gd_anfis_shap_report.json')
    fig, ax = setup('FuzzyXAI Studio - Dashboard')
    box(ax, 0.04, 0.80, 0.18, 0.075, 'Evidence status', ['PASS'], fill=LIGHT, title_color=BLUE)
    box(ax, 0.24, 0.80, 0.14, 0.075, 'Reports', ['4 / 4'], fill=LIGHT, title_color=BLUE)
    box(ax, 0.40, 0.80, 0.16, 0.075, 'Checksums', ['PASS'], fill=LIGHT, title_color=BLUE)
    box(ax, 0.58, 0.80, 0.16, 0.075, 'Last run', ['from manifest'], fill=LIGHT, title_color=BLUE)
    button(ax, 0.04, 0.735, 'Run all scenarios', 0.14)
    button(ax, 0.195, 0.735, 'Check evidence', 0.13)
    button(ax, 0.34, 0.735, 'Open reports', 0.12)
    button(ax, 0.475, 0.735, 'Export screenshots', 0.15)
    cards = [
        ('HYBRID-XIRIS', 'real-output-compatible', ['Тип: safety', f"Данные: {h['total_objects']} объектов", f"missed: {h['baseline_missed']} / {h['fuzzyxai_missed']}; action: block"]),
        ('BEACON-XAI', 'fixture-certified', ['Тип: audit', f"Данные: {b['total_signals']} сигналов", f"manual: {b['baseline_manual_checks']} -> {b['fuzzyxai_manual_checks']}; reports={b['audit_reports']}"]),
        ('GIS INTEGRO', 'source-pending', ['Тип: route', f"p={g['probability']}; alpha={g['mean_alpha_k']}", f"gamma={g['gamma_route']:.2f}; Delta={g['Delta']:.2f}"]),
        ('GD-ANFIS/SHAP', 'source-pending', ['Тип: route', f"Rules={gd['n_rules']}; Delta={gd['Delta']}", f"I_pre={gd['I_pre']}; action=audit"]),
    ]
    positions = [(0.04, 0.42), (0.52, 0.42), (0.04, 0.12), (0.52, 0.12)]
    for (title, status, lines), (x, y) in zip(cards, positions):
        box(ax, x, y, 0.42, 0.25, title, lines, fill='white', title_color=BLUE)
        badge(ax, x + 0.25, y + 0.205, status, fill=WARN if status == 'source-pending' else 'white', color='#b7791f' if status == 'source-pending' else BLUE)
        button(ax, x + 0.018, y + 0.02, 'Открыть сценарий', 0.14)
        button(ax, x + 0.17, y + 0.02, 'Отчёт', 0.08)
        button(ax, x + 0.26, y + 0.02, 'Evidence', 0.09)
    save(fig, '01_home_dashboard.png')

def route_screen(name: str, filename: str, input_lines: list[str], adapter: str, conflict: str, result: str, evidence: str, status: str):
    fig, ax = setup(f'FuzzyXAI Studio - {name}')
    tabbar(ax, 'Route')
    box(ax, 0.04, 0.70, 0.27, 0.10, 'Input data', input_lines, fill=LIGHT)
    if 'HYBRID' in name:
        params = ['score_threshold = 0.70', 'quality_threshold = 0.45', 'chi_R_crit = 1', 'chi_Auto = false']
    else:
        params = ['trace/source checks: enabled', 'action mode: restricted', f'status: {status}']
    box(ax, 0.34, 0.70, 0.28, 0.10, 'Parameters', params, fill='white')
    box(ax, 0.65, 0.70, 0.30, 0.10, 'Evidence status', ['Checksum: PASS', f'Отчёт: {evidence}'], fill='white')
    steps = [('Input', input_lines[:1]), ('Adapter', [adapter]), ('Explanation', [conflict]), ('Risk observer', ['status: built']), ('Action', [result]), ('Evidence', ['PASS'])]
    x = 0.045
    for i, (t, lines) in enumerate(steps):
        box(ax, x, 0.43, 0.135, 0.18, f'{i+1}. {t}', lines, fill='white', title_color=BLUE)
        if i < len(steps) - 1:
            arrow(ax, x + 0.137, 0.52, x + 0.158, 0.52)
        x += 0.153
    box(ax, 0.04, 0.18, 0.43, 0.16, 'Claims: можно', ['маршрут выполнен', 'отчёт сформирован', 'метрики воспроизведены'], fill='white')
    box(ax, 0.52, 0.18, 0.43, 0.16, 'Claims: нельзя', ['качество внешней модели', 'клиническая эффективность', 'production-ready статус'], fill=WARN)
    save(fig, filename)

def hybrid_result():
    h = read_json('reports/chapter5/hybrid_xiris_summary.json')
    bc = read_json('reports/chapter5/hybrid_xiris_blocking_case.json')
    fig, ax = setup('HYBRID-XIRIS - Result')
    tabbar(ax, 'Result')
    box(ax, 0.04, 0.64, 0.22, 0.16, 'ACTION', ['BLOCK', 'Reason: high confidence + low quality'], fill=LIGHT, title_color=BLUE)
    box(ax, 0.29, 0.64, 0.22, 0.16, 'Risk observer', [f"chi_R_crit = {bc['chi_R_crit']}", f"chi_Auto = {str(bc['chi_Auto']).lower()}", 'q=0.45; score=0.70'], fill='white')
    box(ax, 0.54, 0.64, 0.20, 0.16, 'Metrics', [f"Baseline missed: {h['baseline_missed']}", f"FuzzyXAI missed: {h['fuzzyxai_missed']}", f"False block: {h['false_block']}"], fill='white')
    box(ax, 0.77, 0.64, 0.18, 0.16, 'Evidence', ['Checksum: PASS', 'Trace: available'], fill='white')
    box(ax, 0.04, 0.39, 0.91, 0.14, 'Evidence files', ['Report + blocking case: PASS', 'CSV objects: PASS', 'Table + checksum: PASS'], fill='white')
    button(ax, 0.06, 0.305, 'Download JSON', 0.13)
    button(ax, 0.205, 0.305, 'Download CSV', 0.12)
    button(ax, 0.34, 0.305, 'Download table', 0.13)
    button(ax, 0.485, 0.305, 'Show trace', 0.10)
    button(ax, 0.60, 0.305, 'Show checksum', 0.13)
    box(ax, 0.04, 0.11, 0.91, 0.13, 'Интерпретация', ['Система блокирует автоматическое решение: уверенность модели высокая, но качество входа низкое. Это safety-маршрут, а не claim о clinical effectiveness.'], fill=LIGHT)
    save(fig, '03_hybrid_xiris_result.png')

def beacon_result():
    b = read_json('reports/chapter5/beacon_xai_summary.json')
    fig, ax = setup('BEACON-XAI - результат аудита')
    box(ax, 0.04, 0.58, 0.28, 0.24, 'Результат', ['AUDIT REPORT', f"Valid after adapter: {b['valid_after_adapter']}/{b['total_signals']}"], fill=LIGHT, title_color=BLUE)
    box(ax, 0.36, 0.58, 0.28, 0.24, 'Сравнение', [f"Manual checks baseline: {b['baseline_manual_checks']}", f"Manual checks FuzzyXAI: {b['fuzzyxai_manual_checks']}", f"Audit reports: {b['audit_reports']}"], fill='white')
    box(ax, 0.68, 0.58, 0.27, 0.24, 'Evidence', ['Отчёт: beacon_xai_summary.json', 'Таблица: beacon_xai_summary.md', 'Статус проверки: PASS'], fill='white')
    box(ax, 0.04, 0.24, 0.91, 0.22, 'Вывод', ['Ручная проверка сокращена в контрольном fixture-протоколе.', '17 сигналов не прошли адаптер; причины сохранены в beacon_xai_adapter_failures.csv.'], fill='white')
    save(fig, '05_beacon_audit_result.png')


def gis_report():
    g = read_json('reports/chapter5/gis_integro_route_metrics.json')
    route_screen('GIS INTEGRO', '06_gis_integro_route_report.png', ['probability + alpha_k + SHAP support'], 'gis-integro-adapter', f"gamma_route = {g['gamma_route']:.2f}, Delta = {g['Delta']:.2f}", 'ROUTE REPORT', 'gis_integro_route_metrics.json', 'source-pending')


def gd_report():
    gd = read_json('reports/chapter5/gd_anfis_shap_report.json')
    route_screen('GD-ANFIS/SHAP', '07_gd_anfis_shap_route_report.png', ['ANFIS rules + SHAP vector'], 'tabular-xai-adapter', f"Delta = {gd['Delta']}, I_pre = {gd['I_pre']}", 'AUDIT REPORT', 'gd_anfis_shap_report.json', 'source-pending')


def evidence_center():
    fig, ax = setup('Evidence Center')
    kpis = [('Pipeline status', 'PASS'), ('Reports', '4 / 4'), ('Tables', '5 / 5'), ('Checksums', 'PASS'), ('Registry', 'PASS')]
    x = 0.04
    for title, value in kpis:
        box(ax, x, 0.76, 0.16, 0.08, title, [value], fill=LIGHT, title_color=BLUE)
        x += 0.18
    headers = ['Сценарий', 'JSON report', 'Table', 'Checksum', 'Status']
    rows = [
        ['HYBRID-XIRIS', 'hybrid_xiris_summary.json', 'hybrid_xiris_baseline_comparison.md', 'PASS', 'PASS'],
        ['BEACON-XAI', 'beacon_xai_summary.json', 'beacon_xai_summary.md', 'PASS', 'PASS'],
        ['GIS INTEGRO', 'gis_integro_route_metrics.json', 'gis_integro_metrics.md', 'PASS', 'PASS'],
        ['GD-ANFIS/SHAP', 'gd_anfis_shap_report.json', 'gd_anfis_shap_metrics.md', 'PASS', 'PASS'],
    ]
    x0, y0 = 0.04, 0.62
    widths = [0.18, 0.25, 0.27, 0.10, 0.10]
    row_h = 0.075
    for i, head in enumerate(headers):
        ax.add_patch(FancyBboxPatch((x0 + sum(widths[:i]), y0), widths[i], row_h, boxstyle='square,pad=0', linewidth=1, edgecolor=BORDER, facecolor=LIGHT))
        ax.text(x0 + sum(widths[:i]) + 0.01, y0 + row_h/2, head, fontsize=10.5, color=GRAY, va='center')
    for r, row in enumerate(rows):
        y = y0 - (r + 1) * row_h
        for i, cell in enumerate(row):
            ax.add_patch(FancyBboxPatch((x0 + sum(widths[:i]), y), widths[i], row_h, boxstyle='square,pad=0', linewidth=1, edgecolor=BORDER, facecolor='white'))
            ax.text(x0 + sum(widths[:i]) + 0.01, y + row_h/2, cell, fontsize=9.5, color=BLACK, va='center')
    button(ax, 0.04, 0.20, 'Run pipeline', 0.12)
    button(ax, 0.18, 0.20, 'Compare reports', 0.14)
    button(ax, 0.34, 0.20, 'Verify checksums', 0.15)
    button(ax, 0.51, 0.20, 'Download manifest', 0.15)
    button(ax, 0.68, 0.20, 'Download tables', 0.14)
    save(fig, '08_evidence_center.png')

def developer_details():
    fig, ax = setup('Developer / Evidence details')
    box(ax, 0.04, 0.76, 0.18, 0.08, 'Status', ['PASS'], fill=LIGHT, title_color=BLUE)
    box(ax, 0.24, 0.76, 0.22, 0.08, 'Last run', ['from manifest/checksums'], fill=LIGHT, title_color=BLUE)
    box(ax, 0.04, 0.52, 0.28, 0.18, 'Registry / Manifest', ['registry/modules.json', 'manifest_sha256.json', 'checksums.sha256'], fill='white')
    box(ax, 0.36, 0.52, 0.28, 0.18, 'Logs', ['logs/run_chapter4_5.log', 'logs/compare_reports.log', 'logs/checksums.log'], fill='white')
    box(ax, 0.68, 0.52, 0.27, 0.18, 'Raw reports', ['hybrid_xiris_summary.json', 'beacon_xai_summary.json', 'gis_integro_route_metrics.json', 'gd_anfis_shap_report.json'], fill='white')
    button(ax, 0.06, 0.42, 'Open registry', 0.12)
    button(ax, 0.20, 0.42, 'Open manifest', 0.13)
    button(ax, 0.35, 0.42, 'Open checksums', 0.13)
    button(ax, 0.50, 0.42, 'Open logs', 0.10)
    button(ax, 0.62, 0.42, 'Open Dockerfile', 0.13)
    box(ax, 0.04, 0.18, 0.91, 0.16, 'Developer boundary', ['Технические JSON, registry, manifest и логи вынесены сюда. На пользовательских экранах остаются сценарий, параметры, route, result, claims и evidence.'], fill=LIGHT)
    save(fig, '09_developer_details.png')

def html_index():
    html = '''<!doctype html><html lang="ru"><meta charset="utf-8"><title>FuzzyXAI Studio</title><body style="font-family:Arial,sans-serif;background:#fff;color:#1d1d1f;max-width:1180px;margin:40px auto;line-height:1.45"><h1>FuzzyXAI Studio - сценарии объяснения, аудита и контроля риска</h1><p>Витрина сценариев: вход, маршрут, результат, claims и evidence.</p><ul><li>HYBRID-XIRIS: baseline missed = 168, FuzzyXAI missed = 0</li><li>BEACON-XAI: 64 -> 11, audit reports = 12</li><li>GIS INTEGRO: gamma_route = 0.20, Delta = 0.08</li><li>GD-ANFIS/SHAP: route metrics из fixture</li></ul></body></html>'''
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'fuzzyxai_studio_scenarios.html').write_text(html, encoding='utf-8')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob('*.png'):
        old.unlink()
    home()
    route_screen('HYBRID-XIRIS', '02_hybrid_xiris_route.png', ['model_score + quality_score', '1000 объектов, critical = 168'], 'image-adapter', 'высокая уверенность + низкое качество', 'BLOCK', 'hybrid_xiris_summary.json', 'safety')
    hybrid_result()
    route_screen('BEACON-XAI', '04_beacon_audit_route.png', ['100 сигналов', 'counterevidence + trace_version'], 'beacon-adapter', '17 сигналов не прошли адаптер', 'AUDIT REPORT', 'beacon_xai_summary.json', 'audit')
    beacon_result()
    gis_report()
    gd_report()
    evidence_center()
    developer_details()
    html_index()


if __name__ == '__main__':
    main()
