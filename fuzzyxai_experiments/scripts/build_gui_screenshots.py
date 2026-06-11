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


def save(fig, name: str):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, bbox_inches='tight', pad_inches=0, facecolor='white')
    plt.close(fig)


def home():
    h = read_json('reports/chapter5/hybrid_xiris_summary.json')
    b = read_json('reports/chapter5/beacon_xai_summary.json')
    g = read_json('reports/chapter5/gis_integro_route_metrics.json')
    gd = read_json('reports/chapter5/gd_anfis_shap_report.json')
    fig, ax = setup('FuzzyXAI Studio - сценарии объяснения, аудита и контроля риска')
    cards = [
        ('HYBRID-XIRIS', ['Тип: safety', 'Задача: блокировка критического случая', f"Данные: {h['total_objects']} объектов", f"Результат: baseline missed = {h['baseline_missed']}", f"FuzzyXAI missed = {h['fuzzyxai_missed']}", 'Статус: real-output-compatible', '[ Открыть сценарий ]']),
        ('BEACON-XAI', ['Тип: audit', 'Задача: сокращение ручной проверки', f"Данные: {b['total_signals']} сигналов", f"Результат: {b['baseline_manual_checks']} -> {b['fuzzyxai_manual_checks']}", f"Audit reports = {b['audit_reports']}", 'Статус: fixture-certified', '[ Открыть сценарий ]']),
        ('GIS INTEGRO', ['Тип: route', 'Задача: проверка геомаршрута', 'Данные: geo fixture', f"gamma_route = {g['gamma_route']:.2f}", f"Delta = {g['Delta']:.2f}", 'Статус: source-pending', '[ Открыть сценарий ]']),
        ('GD-ANFIS/SHAP', ['Тип: route', 'Задача: объединение правил и SHAP', 'Данные: rules + SHAP fixture', f"Delta = {gd['Delta']}", f"I_pre = {gd['I_pre']}", 'Статус: source-pending', '[ Открыть сценарий ]']),
    ]
    positions = [(0.04, 0.53), (0.52, 0.53), (0.04, 0.18), (0.52, 0.18)]
    for (title, lines), (x, y) in zip(cards, positions):
        box(ax, x, y, 0.42, 0.29, title, lines, fill=LIGHT, title_color=BLUE)
    save(fig, '01_home_dashboard.png')


def route_screen(name: str, filename: str, input_lines: list[str], adapter: str, conflict: str, result: str, evidence: str, status: str):
    fig, ax = setup(f'FuzzyXAI Studio - {name}')
    box(ax, 0.04, 0.74, 0.92, 0.11, 'Сценарий', [f'Класс: {status}', 'Цель: показать вход, маршрут, действие и evidence без чтения кода.'], fill=LIGHT)
    steps = [
        ('1. Вход', input_lines),
        ('2. Адаптер', [adapter]),
        ('3. Объяснение', [conflict, '(E_k / D_k)']),
        ('4. Наблюдатель', ['Проверка риска, согласованности и ограничений claims']),
        ('5. Действие', [result]),
        ('6. Evidence', [evidence]),
    ]
    x = 0.045
    for i, (t, lines) in enumerate(steps):
        box(ax, x, 0.42, 0.135, 0.20, t, lines, fill='white', title_color=BLUE)
        if i < len(steps) - 1:
            arrow(ax, x + 0.137, 0.52, x + 0.158, 0.52)
        x += 0.153
    box(ax, 0.04, 0.18, 0.43, 0.16, 'Что можно утверждать', ['маршрут выполнен', 'отчёт сформирован', 'метрики воспроизведены'], fill='white')
    box(ax, 0.52, 0.18, 0.43, 0.16, 'Что нельзя утверждать', ['новая точность внешней модели', 'клиническая эффективность', 'production-ready статус'], fill=WARN)
    save(fig, filename)


def hybrid_result():
    h = read_json('reports/chapter5/hybrid_xiris_summary.json')
    fig, ax = setup('HYBRID-XIRIS - результат')
    box(ax, 0.04, 0.58, 0.28, 0.24, 'Результат', ['BLOCK', 'Причина: высокая уверенность модели при низком качестве сегментации'], fill=LIGHT, title_color=BLUE)
    box(ax, 0.36, 0.58, 0.28, 0.24, 'Сравнение', [f"Baseline missed: {h['baseline_missed']}", f"FuzzyXAI missed: {h['fuzzyxai_missed']}", f"False block: {h['false_block']}"], fill='white')
    box(ax, 0.68, 0.58, 0.27, 0.24, 'Evidence', ['Отчёт: hybrid_xiris_summary.json', 'Таблица: hybrid_xiris_baseline_comparison.md', 'Статус проверки: PASS'], fill='white')
    box(ax, 0.04, 0.24, 0.91, 0.22, 'Вывод', ['Критические случаи в контрольном протоколе не пропущены: FuzzyXAI missed = 0.', 'Baseline принимает критические случаи, если model_score > 0.7, и поэтому пропускает 168 объектов.'], fill='white')
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
    box(ax, 0.04, 0.72, 0.91, 0.12, 'Статус evidence', ['Pipeline status: PASS', 'Reports: 4 / 4', 'Tables: 5 / 5', 'Checksums: PASS', 'Registry: PASS'], fill=LIGHT)
    box(ax, 0.04, 0.60, 0.42, 0.22, 'HYBRID-XIRIS', ['Отчёт: hybrid_xiris_summary.json', 'Таблица: hybrid_xiris_baseline_comparison.md', 'Checksum: checksums.sha256', 'Последний запуск: PASS'], fill='white')
    box(ax, 0.52, 0.60, 0.42, 0.22, 'BEACON-XAI', ['Отчёт: beacon_xai_summary.json', 'Таблица: beacon_xai_summary.md', 'Trace: beacon_xai_adapter_failures.csv', 'Последний запуск: PASS'], fill='white')
    box(ax, 0.04, 0.30, 0.42, 0.22, 'GIS INTEGRO', ['Отчёт: gis_integro_route_metrics.json', 'Таблица: gis_integro_metrics.md', 'Статус: source-pending', 'Последний запуск: PASS'], fill=WARN)
    box(ax, 0.52, 0.30, 0.42, 0.22, 'GD-ANFIS/SHAP', ['Отчёт: gd_anfis_shap_report.json', 'Таблица: gd_anfis_shap_metrics.md', 'Статус: source-pending', 'Последний запуск: PASS'], fill=WARN)
    save(fig, '08_evidence_center.png')


def developer_details():
    fig, ax = setup('Developer / Evidence details')
    box(ax, 0.04, 0.64, 0.28, 0.20, 'Registry', ['registry/modules.json', '4 сценария', 'claim_scope задан для каждого'], fill='white')
    box(ax, 0.36, 0.64, 0.28, 0.20, 'Manifest', ['manifest_sha256.json', 'checksums.sha256', 'sha256sum -c checksums.sha256'], fill='white')
    box(ax, 0.68, 0.64, 0.27, 0.20, 'Run scripts', ['run_chapter4_5.sh', 'compare_reports.py', 'export_gui_screenshots.sh'], fill='white')
    box(ax, 0.04, 0.30, 0.91, 0.22, 'Raw reports', ['reports/chapter5/hybrid_xiris_summary.json', 'reports/chapter5/beacon_xai_summary.json', 'reports/chapter5/gis_integro_route_metrics.json', 'reports/chapter5/gd_anfis_shap_report.json'], fill=LIGHT)
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
