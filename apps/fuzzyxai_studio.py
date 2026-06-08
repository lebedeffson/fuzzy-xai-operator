"""Unified single-page FuzzyXAI Studio for presentation and expert work."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from apps.services.layered_case import LayeredCaseService, build_case_state
from fuzzyxai.datasets import list_dataset_modes
from fuzzyxai.ecosystem import build_evidence_matrix, load_ecosystem_registry
from fuzzyxai.studio import (
    PRESETS,
    StudioExplainPlan,
    WhatIfOverrides,
    apply_named_preset,
    build_operator_trace,
    composition_rows,
    explain_action_text,
    export_defense_case,
    representation_rows,
    route_rows,
)

ROOT = Path(__file__).resolve().parents[1]


def _num(v: Any, nd: int = 4) -> str:
    try:
        return f'{float(v):.{nd}f}'
    except Exception:
        return '-'


def _to_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return int(default)


def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return None


def _read_csv_rows(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(dict(row))
                if limit is not None and len(rows) >= limit:
                    break
    except Exception:
        return []
    return rows


def _status_badge(status: str) -> tuple[str, str]:
    s = str(status).upper()
    if s == 'READY':
        return s, '#16a34a'
    if s == 'MISSING':
        return s, '#d97706'
    return s, '#dc2626'


def _action_color(action: str) -> str:
    mapping = {
        'accept': '#16a34a',
        'lower_confidence': '#65a30d',
        'request_more_data': '#d97706',
        'defer_to_human': '#ea580c',
        'block': '#dc2626',
    }
    return mapping.get(str(action), '#334155')


def _chi_auto_for_action(st: dict[str, Any], action: str) -> bool:
    raw = st.get('contexts', {}).get('AutoAccept', {}).get('E_action', True)
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(raw)
    tokens = [x.strip() for x in str(raw).split(',') if x.strip()]
    if tokens:
        return str(action) in set(tokens)
    return bool(raw)


def _membership_option(p: float, mu: dict[str, Any] | None = None) -> dict[str, Any]:
    x = [i / 100 for i in range(101)]
    low = [max(0.0, min(1.0, 2.718281828 ** (-((v - 0.0) ** 2) / (2 * 0.20 * 0.20)))) for v in x]
    med = [max(0.0, min(1.0, 2.718281828 ** (-((v - 0.5) ** 2) / (2 * 0.18 * 0.18)))) for v in x]
    high = [max(0.0, min(1.0, 2.718281828 ** (-((v - 1.0) ** 2) / (2 * 0.20 * 0.20)))) for v in x]
    pv = max(0.0, min(1.0, float(p)))
    mu = mu or {}
    mu_low = float(mu.get('low', max(0.0, min(1.0, (0.5 - pv) / 0.5))))
    mu_med = float(mu.get('medium', max(0.0, 1.0 - abs(pv - 0.5) / 0.5)))
    mu_high = float(mu.get('high', max(0.0, min(1.0, (pv - 0.5) / 0.5))))
    return {
        'tooltip': {'trigger': 'axis'},
        'legend': {'data': ['low curve', 'medium curve', 'high curve', 'case']},
        'xAxis': {'type': 'value', 'min': 0.0, 'max': 1.0},
        'yAxis': {'type': 'value', 'min': 0.0, 'max': 1.0},
        'grid': {'left': 48, 'right': 20, 'top': 36, 'bottom': 46},
        'series': [
            {'name': 'low curve', 'type': 'line', 'smooth': True, 'showSymbol': False, 'areaStyle': {'opacity': 0.08}, 'lineStyle': {'width': 3, 'color': '#3b82f6'}, 'data': [[x[i], low[i]] for i in range(len(x))]},
            {'name': 'medium curve', 'type': 'line', 'smooth': True, 'showSymbol': False, 'areaStyle': {'opacity': 0.08}, 'lineStyle': {'width': 3, 'color': '#84cc16'}, 'data': [[x[i], med[i]] for i in range(len(x))]},
            {'name': 'high curve', 'type': 'line', 'smooth': True, 'showSymbol': False, 'areaStyle': {'opacity': 0.08}, 'lineStyle': {'width': 3, 'color': '#0f172a'}, 'data': [[x[i], high[i]] for i in range(len(x))]},
            {'name': 'case', 'type': 'scatter', 'symbolSize': 13, 'itemStyle': {'color': '#f97316'}, 'data': [[pv, 0.02], [pv, mu_low], [pv, mu_med], [pv, mu_high]], 'markLine': {'symbol': ['none', 'none'], 'lineStyle': {'type': 'dashed', 'color': '#f97316'}, 'data': [{'xAxis': pv}]}},
        ],
        'graphic': [
            {'type': 'text', 'left': '2%', 'top': '4%', 'style': {'text': f'p={pv:.4f} | μ_low={mu_low:.3f} μ_med={mu_med:.3f} μ_high={mu_high:.3f}', 'fill': '#334155', 'fontSize': 12}},
        ],
    }


def _risk_contrib_option(contrib: dict[str, Any], risk: dict[str, Any] | None = None) -> dict[str, Any]:
    labels = ['predicted_risk', 'uncertainty', 'interpretability_gap', 'reduction_loss', 'chi_R']
    c = dict(contrib or {})
    if (not c or sum(float(c.get(k, 0.0)) for k in labels) == 0.0) and risk:
        comp = risk.get('components', {}) or {}
        w = risk.get('weights', {}) or comp.get('weights', {}) or {}
        c = {
            'predicted_risk': float(w.get('predicted_risk', 0.0)) * float(comp.get('predicted_risk', 0.0)),
            'uncertainty': float(w.get('uncertainty', 0.0)) * float(comp.get('uncertainty', 0.0)),
            'interpretability_gap': float(w.get('interpretability_gap', 0.0)) * float(comp.get('interpretability_gap', 0.0)),
            'reduction_loss': float(w.get('reduction_loss', 0.0)) * float(comp.get('reduction_loss', 0.0)),
            'chi_R': float(w.get('diagnostic', 0.0)) * float(comp.get('diagnostic', comp.get('chi_R', 0.0))),
        }
    pairs = [(k, float(c.get(k, 0.0))) for k in labels]
    pairs.sort(key=lambda kv: kv[1], reverse=True)
    labels = [p[0] for p in pairs]
    vals = [p[1] for p in pairs]
    total = sum(vals)
    pct = [0.0 if total <= 1e-12 else (v / total) * 100.0 for v in vals]
    labels_with_abs = [f'{k} ({v:.4f})' for k, v in zip(labels, vals)]
    pct_rounded = [round(v, 2) for v in pct]
    max_pct = max(pct_rounded) if pct_rounded else 0.0
    axis_max = 100.0 if max_pct > 60 else max(25.0, round(max_pct + 8.0, 2))
    return {
        'tooltip': {'trigger': 'axis', 'axisPointer': {'type': 'shadow'}},
        'legend': {'data': ['risk share %']},
        'grid': {'left': 200, 'right': 26, 'top': 36, 'bottom': 30},
        'xAxis': {'type': 'value', 'name': '% share', 'min': 0, 'max': axis_max},
        'yAxis': {'type': 'category', 'data': labels_with_abs},
        'series': [
            {
                'name': 'risk share %',
                'type': 'bar',
                'data': pct_rounded,
                'itemStyle': {'color': '#0b6f8a'},
                'barWidth': 18,
                'label': {'show': True, 'position': 'right', 'formatter': '{c}%'},
            },
        ],
        'graphic': [
            {'type': 'text', 'left': '2%', 'top': '4%', 'style': {'text': f'total_rho={total:.4f}', 'fill': '#334155', 'fontSize': 12}},
        ],
    }


def _baseline_compare_option(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = [str(r.get('baseline', 'n/a')) for r in rows]
    agreement = [float(r.get('agreement_reference', 0.0)) for r in rows]
    missed = [float(r.get('missed_critical_ruptures', 0.0)) for r in rows]
    return {
        'tooltip': {'trigger': 'axis'},
        'legend': {'data': ['agreement_reference', 'missed_critical_ruptures']},
        'xAxis': {'type': 'category', 'data': labels, 'axisLabel': {'interval': 0, 'rotate': 18}},
        'yAxis': [{'type': 'value', 'name': 'agreement'}, {'type': 'value', 'name': 'missed'}],
        'series': [
            {'name': 'agreement_reference', 'type': 'bar', 'data': agreement, 'itemStyle': {'color': '#0f766e'}},
            {'name': 'missed_critical_ruptures', 'type': 'line', 'yAxisIndex': 1, 'data': missed, 'itemStyle': {'color': '#dc2626'}},
        ],
    }


def _non_synthetic_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ds in ['breast_cancer', 'wine_risk', 'diabetes_binary']:
        payload = _load_json(ROOT / 'reports' / 'structure_aware_benchmark' / f'{ds}.json') or {}
        full = next((r for r in payload.get('rows', []) if r.get('policy') == 'full_observer_calibrated'), {})
        threshold = next((r for r in payload.get('rows', []) if r.get('policy') == 'probability_threshold'), {})
        if not full or not threshold:
            continue
        full_ag = float(full.get('agreement_reference', 0.0))
        th_ag = float(threshold.get('agreement_reference', 0.0))
        full_faa = float(full.get('false_auto_accept_rate', 0.0))
        th_faa = float(threshold.get('false_auto_accept_rate', 0.0))
        rows.append(
            {
                'dataset': ds,
                'full_agreement_ref': full_ag,
                'threshold_agreement_ref': th_ag,
                'agreement_gain': full_ag - th_ag,
                'full_false_auto_accept': full_faa,
                'threshold_false_auto_accept': th_faa,
                'false_auto_accept_drop': th_faa - full_faa,
            }
        )
    return rows


def _non_synthetic_option(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = [str(r.get('dataset', '')) for r in rows]
    gain = [round(float(r.get('agreement_gain', 0.0)), 4) for r in rows]
    drop = [round(float(r.get('false_auto_accept_drop', 0.0)), 4) for r in rows]
    return {
        'tooltip': {'trigger': 'axis'},
        'legend': {'data': ['agreement_gain', 'false_auto_accept_drop'], 'bottom': 0},
        'grid': {'left': 52, 'right': 24, 'top': 40, 'bottom': 78},
        'xAxis': {'type': 'category', 'data': labels},
        'yAxis': {'type': 'value', 'min': 0, 'max': 0.5},
        'series': [
            {'name': 'agreement_gain', 'type': 'bar', 'data': gain, 'itemStyle': {'color': '#0f766e'}},
            {'name': 'false_auto_accept_drop', 'type': 'bar', 'data': drop, 'itemStyle': {'color': '#b45309'}},
        ],
    }


def _scenario_status_option(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get('status', 'unknown'))
        counts[status] = counts.get(status, 0) + 1
    labels = list(counts.keys())
    values = [counts[k] for k in labels]
    colors = {
        'real-output-compatible': '#0f766e',
        'fixture-certified': '#0b6f8a',
        'source-pending': '#d97706',
        'planned': '#64748b',
    }
    return {
        'tooltip': {'trigger': 'item'},
        'legend': {'bottom': 0},
        'series': [
            {
                'type': 'pie',
                'radius': ['42%', '72%'],
                'data': [{'name': k, 'value': v, 'itemStyle': {'color': colors.get(k, '#94a3b8')}} for k, v in zip(labels, values)],
            }
        ],
    }


def _operators_flow_option(trace_rows: list[dict[str, Any]]) -> dict[str, Any]:
    names = [str(r.get('operator', '')) for r in trace_rows]
    color_map = {'ok': '#16a34a', 'warning': '#d97706', 'critical': '#dc2626'}
    nodes = []
    for r in trace_rows:
        sev = str(r.get('severity', 'ok'))
        nodes.append(
            {
                'name': str(r.get('operator', '')),
                'itemStyle': {'color': color_map.get(sev, '#16a34a')},
                'value': f"{sev}: {r.get('signal', '')}",
            }
        )
    links = [{'source': names[i], 'target': names[i + 1]} for i in range(len(names) - 1)]
    return {
        'tooltip': {'trigger': 'item'},
        'series': [
            {
                'type': 'graph',
                'layout': 'none',
                'symbolSize': 42,
                'roam': True,
                'label': {'show': True, 'fontSize': 10},
                'edgeSymbol': ['none', 'arrow'],
                'edgeSymbolSize': 8,
                'data': [{'name': n['name'], 'x': 80 + i * 145, 'y': 80 + (i % 2) * 65, 'itemStyle': n['itemStyle'], 'value': n['value']} for i, n in enumerate(nodes)],
                'links': links,
                'lineStyle': {'color': '#0b6f8a', 'width': 2},
            }
        ],
    }


def _dataset_risk_distribution_option(dataset_name: str, case_risk: float) -> dict[str, Any]:
    path = ROOT / 'reports' / 'datasets' / dataset_name / 'predictions.csv'
    risks: list[float] = []
    if path.exists():
        try:
            with path.open(encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    if i >= 5000:
                        break
                    try:
                        risks.append(float(row.get('predicted_risk', 0.0)))
                    except Exception:
                        continue
        except Exception:
            risks = []
    bins = 20
    if not risks:
        return {
            'xAxis': {'type': 'category', 'data': [f'{i / bins:.2f}' for i in range(bins)]},
            'yAxis': {'type': 'value'},
            'series': [{'type': 'bar', 'data': [0] * bins, 'itemStyle': {'color': '#94a3b8'}}],
            'graphic': [{'type': 'text', 'left': 'center', 'top': '45%', 'style': {'text': 'predictions.csv not found', 'fill': '#64748b', 'fontSize': 14}}],
        }
    hist = [0] * bins
    for r in risks:
        rv = max(0.0, min(1.0, float(r)))
        idx = min(bins - 1, int(rv * bins))
        hist[idx] += 1
    labels = [f'{i / bins:.2f}-{(i + 1) / bins:.2f}' for i in range(bins)]
    case_idx = min(bins - 1, int(max(0.0, min(1.0, case_risk)) * bins))
    return {
        'tooltip': {'trigger': 'axis'},
        'grid': {'left': 44, 'right': 24, 'top': 36, 'bottom': 56},
        'xAxis': {'type': 'category', 'data': labels, 'axisLabel': {'interval': 1, 'rotate': 40}},
        'yAxis': {'type': 'value', 'name': 'count'},
        'series': [
            {'name': 'risk_hist', 'type': 'bar', 'data': hist, 'itemStyle': {'color': '#0b6f8a'}},
            {'name': 'case_bin', 'type': 'bar', 'data': [hist[i] if i == case_idx else 0 for i in range(bins)], 'itemStyle': {'color': '#f97316'}},
        ],
        'graphic': [
            {'type': 'text', 'left': '2%', 'top': '4%', 'style': {'text': f'case_risk={case_risk:.4f} | n={len(risks)}', 'fill': '#334155', 'fontSize': 12}},
        ],
    }


def _route_stepper_html(rows: list[dict[str, Any]]) -> str:
    chips: list[str] = []
    for row in rows:
        step = str(row.get('step', 'step'))
        state = str(row.get('state', '-'))
        tone = '#16a34a'
        if 'rupture' in state or 'block' in state:
            tone = '#dc2626'
        elif 'selected' in state or 'decided' in state or 'morphism' in state:
            tone = '#0b6f8a'
        chips.append(
            f"<span style='display:inline-block;margin:4px;padding:6px 10px;border-radius:999px;"
            f"border:1px solid #dbe4ef;background:#fff;'>"
            f"<b>{step}</b>: <span style='color:{tone}'>{state}</span></span>"
        )
    return "<div style='display:flex;flex-wrap:wrap;align-items:center;gap:4px'>" + ''.join(chips) + "</div>"


def _operator_timeline_html(rows: list[dict[str, Any]]) -> str:
    color = {'ok': '#16a34a', 'warning': '#d97706', 'critical': '#dc2626'}
    parts: list[str] = []
    for i, row in enumerate(rows):
        sev = str(row.get('severity', 'ok'))
        tone = color.get(sev, '#334155')
        op = str(row.get('operator', 'op'))
        sig = str(row.get('signal', ''))
        parts.append(
            f"<span style='display:inline-block;margin:4px;padding:6px 10px;border-radius:999px;"
            f"background:{tone};color:#fff;font-weight:700'>{op}</span>"
            f"<span style='font-size:12px;color:#475569'>{sig}</span>"
        )
        if i != len(rows) - 1:
            parts.append("<span style='margin:0 8px;color:#94a3b8'>→</span>")
    if not parts:
        return "<span style='color:#64748b'>No operators</span>"
    return "<div style='display:flex;flex-wrap:wrap;align-items:center;gap:6px'>" + ''.join(parts) + "</div>"


def _action_legend_md(st: dict[str, Any], plan: StudioExplainPlan, chi_auto: bool) -> str:
    risk = st.get('risk', {})
    unc = st.get('uncertainty', {})
    exp = st.get('explanation', {})
    chi_r_crit = int(risk.get('chi_R_crit', 0))
    chi_r = int(risk.get('chi_R', 0))
    rho = float(risk.get('rho', 0.0))
    i_pre = float(exp.get('I_pre', 0.0))
    delta = float(unc.get('delta', 0.0))
    i_min = float(plan.i_min)
    d_max = float(plan.delta_max)
    return (
        f"- `chi_R_crit`: `{chi_r_crit}` ({'forces block' if chi_r_crit else 'not critical'})\n"
        f"- `chi_R`: `{chi_r}`\n"
        f"- `chi_Auto`: `{chi_auto}`\n"
        f"- `rho`: `{_num(rho)}`\n"
        f"- `I_pre >= I_min({i_min})`: `{i_pre >= i_min}`\n"
        f"- `Delta <= Delta_max({d_max})`: `{delta <= d_max}`"
    )


def _save_last_case(case_state: dict[str, Any]) -> Path:
    out_dir = ROOT / 'reports' / 'studio'
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / 'last_case.json'
    p.write_text(json.dumps(case_state, ensure_ascii=False, indent=2), encoding='utf-8')
    return p


def _demo_flow_html(scene: str) -> str:
    scene_names = {'safe': 'Safe', 'audit': 'Audit', 'block': 'Block', 'custom': 'Custom'}
    scene_label = scene_names.get(scene, 'Custom')
    step_style = (
        "display:inline-block;margin:4px;padding:6px 10px;border-radius:999px;"
        "border:1px solid #dbe4ef;background:#fff;color:#0f172a;font-weight:600"
    )
    scene_tone = {'safe': '#16a34a', 'audit': '#d97706', 'block': '#dc2626', 'custom': '#475569'}.get(scene, '#475569')
    scene_badge = (
        f"<span style='display:inline-block;margin:4px;padding:6px 10px;border-radius:999px;"
        f"background:{scene_tone};color:#fff;font-weight:700'>scene: {scene_label}</span>"
    )
    steps = [
        "1) Dataset + Case",
        "2) ExplainPlan / What-if",
        "3) Операторы / Topos / риск",
        "4) Действие + экспорт",
    ]
    step_html = ''.join([f"<span style='{step_style}'>{s}</span>" for s in steps])
    return scene_badge + step_html


def _unified_route_html(st: dict[str, Any] | None) -> str:
    action = '-'
    rho = '-'
    selected = '-'
    chi = '-'
    if st:
        risk = st.get('risk', {})
        unc = st.get('uncertainty', {})
        action = str(risk.get('action', '-'))
        rho = _num(risk.get('rho'))
        selected = str(unc.get('selected_class', '-'))
        chi = f"{risk.get('chi_R', 0)}/{risk.get('chi_R_crit', 0)}"
    steps = [
        ('Входной кейс', 'строка/сценарий'),
        ('Прогноз модели', 'p(y|x)'),
        ('E_k', 'объект объяснения'),
        ('A_k^F', selected),
        ('Путь/разрыв', f'chi_R={chi}'),
        ('Контекст', 'chi_Auto'),
        ('Наблюдатель', f'rho={rho}'),
        ('Действие', action),
        ('Отчет', 'JSON/MD/TEX'),
    ]
    parts = []
    for i, (title, note) in enumerate(steps):
        tone = '#0b6f8a'
        if title == 'Действие' and action == 'block':
            tone = '#dc2626'
        elif title == 'Путь/разрыв' and '/1' in chi:
            tone = '#dc2626'
        parts.append(
            "<span style='display:inline-block;margin:5px;padding:8px 11px;border-radius:14px;"
            f"border:1px solid #cfe2f3;background:#fff;color:#0f172a'>"
            f"<b style='color:{tone}'>{title}</b><br><small>{note}</small></span>"
        )
        if i != len(steps) - 1:
            parts.append("<span style='color:#94a3b8;font-weight:800'>→</span>")
    return "<div style='display:flex;flex-wrap:wrap;gap:6px;align-items:center'>" + ''.join(parts) + "</div>"


def _system_layers_html() -> str:
    layers = [
        ('Операторы ядра', 'SystemOperator, compose, semantic disagreement, I(E_G)'),
        ('Иерархия представлений', 'F0, интервальные, hesitant, neutrosophic, multilevel, Delta'),
        ('Topos / HoTT', 'сертифицированный путь, разрыв, chi_Auto, trace'),
        ('Риск-наблюдатель', 'rho(x), пороги, политика действия'),
        ('Экосистема', 'SDK-адаптеры, реестр, API, пакет доказательности'),
    ]
    cards = []
    for title, body in layers:
        cards.append(
            "<div style='flex:1 1 210px;padding:12px;border-radius:14px;background:#f8fbff;"
            "border:1px solid #dbe7f3'>"
            f"<div style='font-weight:800;color:#0b6f8a'>{title}</div>"
            f"<div style='font-size:12px;color:#334155;margin-top:4px'>{body}</div></div>"
        )
    return "<div style='display:flex;flex-wrap:wrap;gap:10px'>" + ''.join(cards) + "</div>"


def run(port: int = 8097) -> None:
    try:
        from nicegui import ui
    except Exception as exc:  # pragma: no cover
        raise SystemExit('Install nicegui from requirements.txt to run FuzzyXAI Studio.') from exc

    service = LayeredCaseService.create()
    plan = StudioExplainPlan()
    dataset_keys = [d.key for d in list_dataset_modes()]
    state: dict[str, Any] = {'case': None}
    controls: dict[str, Any] = {}

    ui.page_title('FuzzyXAI Studio')
    ui.colors(primary='#0b6f8a', secondary='#0f766e', accent='#b45309')
    ui.add_css(
        """
        .studio-shell {background: linear-gradient(180deg, #f8fbff 0%, #eff6ff 100%); overflow-x:hidden;}
        .nicegui-content {max-width: 100% !important; width: 100% !important; padding: 0 !important;}
        .studio-card {border: 1px solid #dbe7f3; border-radius: 14px; box-shadow: 0 6px 20px rgba(15, 23, 42, 0.05); background:#fff;}
        .studio-title {font-weight: 700; letter-spacing: .01em;}
        .studio-note {color: #334155; font-size: .9rem;}
        .studio-chip {display:inline-block; padding:4px 10px; border-radius:9999px; font-size:.78rem; font-weight:700;}
        .studio-layout {display:grid; grid-template-columns:minmax(340px,31%) minmax(0,1fr); gap:12px; width:100%; box-sizing:border-box;}
        .studio-layout > .nicegui-column:first-child {position: sticky; top: 72px; align-self: start; max-height: calc(100vh - 86px); overflow: auto; padding-right: 4px;}
        .studio-layout .q-tab {font-weight: 700; letter-spacing: .01em;}
        .q-table__container {max-width: 100% !important; overflow-x: auto !important;}
        .q-tab-panels, .q-tab-panel, .q-panel {max-width: 100% !important; overflow-x: hidden !important;}
        .echart, .q-markup-table {background:#fff;}
        .studio-card .q-card__section {padding: 12px 14px;}
        .studio-card .q-table th, .studio-card .q-table td {padding: 8px 10px; font-size: 12px;}
        .studio-card .q-field--dense .q-field__control {min-height: 34px;}
        @media (max-width: 1200px) {.studio-layout {grid-template-columns:1fr;}}
        @media (max-width: 1200px) {.studio-layout > .nicegui-column:first-child {position: static; max-height: none; overflow: visible;}}
        """
    )
    ui.query('.nicegui-content').classes(add='studio-shell')

    with ui.header().classes('items-center justify-between'):
        ui.label('FuzzyXAI Studio').classes('text-h5 studio-title')
        ui.label('Единый маршрут: кейс -> E_k -> A_k^F -> chi_Auto -> rho -> действие').classes('text-sm opacity-80')

    with ui.element('div').classes('studio-layout p-4'):
        left = ui.column().classes('w-full gap-3')
        right = ui.column().classes('w-full gap-3')

    with left:
        with ui.card().classes('w-full studio-card'):
            ui.label('1) Кейс и запуск').classes('text-subtitle1 studio-title')
            dataset_select = ui.select(dataset_keys, value='breast_cancer', label='Dataset mode').props('outlined dense')
            scenario_select = ui.select(['safe', 'ambiguous', 'risky', 'rupture', 'context_block'], value='safe', label='Scenario').props('outlined dense')
            preset_select = ui.select(['none'] + list(PRESETS.keys()), value='none', label='Preset').props('outlined dense')
            sample_idx = ui.number(label='Sample index', value=0, min=0, step=1).props('outlined dense')
            with ui.row().classes('gap-2'):
                run_btn = ui.button('Запустить маршрут', color='primary')
            ui.label('Сценарии показа').classes('text-caption')
            with ui.row().classes('w-full gap-1'):
                scene_safe_btn = ui.button('Безопасный сценарий', color='positive').props('flat dense')
                scene_audit_btn = ui.button('Аудит', color='warning').props('flat dense')
                scene_block_btn = ui.button('Блокировка', color='negative').props('flat dense')
                scene_reset_btn = ui.button('Сбросить', color='secondary').props('flat dense')
            demo_flow = ui.html(_demo_flow_html('custom')).classes('w-full')
            ui.label('Быстрые пресеты').classes('text-caption')
            quick_preset_buttons: dict[str, Any] = {}
            with ui.row().classes('w-full gap-1'):
                for preset_name in PRESETS.keys():
                    label = {
                            'safe_accept': 'безопасно принять',
                            'high_uncertainty': 'высокая неопределенность',
                            'need_more_data': 'нужны данные',
                            'context_forbidden': 'контекст запрещен',
                            'critical_rupture': 'критический разрыв',
                            'source_conflict': 'конфликт источников',
                            'trace_gap': 'разрыв trace',
                            'reduction_loss_too_high': 'потеря редукции',
                        }.get(preset_name, preset_name.replace('_', ' '))
                    quick_preset_buttons[preset_name] = ui.button(label, color='secondary').props('flat dense')
        with ui.card().classes('w-full studio-card'):
            ui.label('Карточка датасета').classes('text-subtitle1 studio-title')
            dataset_preview = ui.markdown('Выбери dataset mode и нажми `Запустить маршрут`.')

        with ui.card().classes('w-full studio-card'):
            ui.label('2) ExplainPlan').classes('text-subtitle1 studio-title')

            def labeled_slider(title: str, value: float):
                with ui.column().classes('w-full gap-1'):
                    ui.label(title).classes('text-caption')
                    return ui.slider(min=0, max=1, step=0.01, value=value).props('label-always')

            w_p = labeled_slider('w_p', plan.w_p)
            w_u = labeled_slider('w_u', plan.w_u)
            w_i = labeled_slider('w_I', plan.w_I)
            w_d = labeled_slider('w_Delta', plan.w_Delta)
            w_r = labeled_slider('w_R', plan.w_R)
            t1 = labeled_slider('theta_1', plan.theta_1)
            t2 = labeled_slider('theta_2', plan.theta_2)
            t3 = labeled_slider('theta_3', plan.theta_3)
            t4 = labeled_slider('theta_4', plan.theta_4)
            i_min = labeled_slider('I_min', plan.i_min)
            d_max = labeled_slider('Delta_max', plan.delta_max)

            with ui.row().classes('w-full gap-2'):
                baseline_access = ui.select(['native', 'equal_guardrail'], value=plan.baseline_access, label='baseline_access').props('outlined dense')
                reduction_strategy = ui.select(['balance', 'midpoint', 'upper', 'lower'], value=plan.reduction_strategy, label='reduction_strategy').props('outlined dense')
            apply_plan_btn = ui.button('Применить план', color='primary')

        with ui.card().classes('w-full studio-card'):
            ui.label('3) What-if').classes('text-subtitle1 studio-title')
            use_overrides = ui.switch('Enable what-if', value=False).props('aria-label=\"Enable what-if\"')

            def labeled_slider_what(title: str, value: float):
                with ui.column().classes('w-full gap-1'):
                    ui.label(title).classes('text-caption')
                    return ui.slider(min=0, max=1, step=0.01, value=value).props('label-always')

            p_r = labeled_slider_what('predicted_risk', 0.70)
            u_m = labeled_slider_what('u_M', 0.35)
            i_p = labeled_slider_what('I_pre', 0.67)
            d_l = labeled_slider_what('Delta', 0.11)
            chi_r = ui.switch('chi_R', value=False).props('aria-label=\"chi_R\"')
            chi_r_crit = ui.switch('chi_R_crit', value=False).props('aria-label=\"chi_R_crit\"')
            chi_auto = ui.switch('chi_Auto', value=True).props('aria-label=\"chi_Auto\"')
            trace_valid = ui.switch('trace_valid', value=True).props('aria-label=\"trace_valid\"')

        with ui.card().classes('w-full studio-card'):
            ui.label('4) Сравнение').classes('text-subtitle1 studio-title')
            bench_dataset = ui.select(dataset_keys, value='synthetic_ruptures', label='Датасет сравнения').props('outlined dense')
            bench_access = ui.select(['native', 'equal_guardrail'], value='native', label='Baseline access').props('outlined dense')
            bench_kind = ui.select(['baseline_comparison', 'structure_aware'], value='baseline_comparison', label='Тип отчета').props('outlined dense')
            bench_load_btn = ui.button('Загрузить отчет сравнения', color='primary')
            bench_box = ui.column().classes('w-full')
            bench_chart = ui.echart(_baseline_compare_option([])).classes('w-full h-56')

        with ui.card().classes('w-full studio-card'):
            ui.label('5) Экспорт').classes('text-subtitle1 studio-title')
            export_btn = ui.button('Экспорт текущего кейса (JSON/MD/TEX)', color='primary')
            export_info = ui.column().classes('w-full')

    with right:
        with ui.tabs().classes('w-full') as right_tabs:
            tab_overview = ui.tab('Маршрут')
            tab_operators = ui.tab('Операторы')
            tab_evidence = ui.tab('Доказательства')
            tab_artifacts = ui.tab('Артефакты')

        with ui.tab_panels(right_tabs, value=tab_overview).classes('w-full'):
            with ui.tab_panel(tab_overview).classes('w-full gap-3'):
                project_map_card = ui.card().classes('w-full studio-card')
                summary = ui.card().classes('w-full studio-card')
                key_panels = ui.row().classes('w-full gap-2')
                viz_row = ui.row().classes('w-full gap-2')
                viz_row_2 = ui.row().classes('w-full gap-2')
                pipeline_table = ui.card().classes('w-full studio-card')
                method_card = ui.card().classes('w-full studio-card')
                with viz_row:
                    with ui.card().classes('w-full lg:w-[49%] studio-card'):
                        ui.label('Функции принадлежности').classes('text-subtitle2 studio-title')
                        membership_chart = ui.echart(_membership_option(0.5)).classes('w-full h-72')
                        membership_meta = ui.markdown('`p`: - | `mu_low`: - | `mu_med`: - | `mu_high`: -').classes('studio-note')
                    with ui.card().classes('w-full lg:w-[49%] studio-card'):
                        ui.label('Вклад в риск').classes('text-subtitle2 studio-title')
                        risk_contrib_chart = ui.echart(_risk_contrib_option({})).classes('w-full h-72')
                        risk_meta = ui.markdown('`total`: - | `top_factor`: -').classes('studio-note')
                with viz_row_2:
                    with ui.card().classes('w-full studio-card'):
                        ui.label('Распределение риска по датасету').classes('text-subtitle2 studio-title')
                        dist_chart = ui.echart(_dataset_risk_distribution_option('breast_cancer', 0.5)).classes('w-full h-64')
                        dist_meta = ui.markdown('`dataset`: - | `n`: - | `case_risk`: -').classes('studio-note')
                with method_card:
                    ui.label('Методология (кратко)').classes('text-subtitle1 studio-title')
                    ui.markdown(
                        '- Система показывает один маршрут принятия решения, а не разрозненные окна.\n'
                        '- Любое изменение в ExplainPlan/What-if сразу влияет на `rho`, `chi_Auto`, `action`.\n'
                        '- Блок сравнения отделяет `native` и `equal_guardrail` режимы доступа.'
                    )

            with ui.tab_panel(tab_operators).classes('w-full gap-3'):
                operators_table = ui.card().classes('w-full studio-card')
                operator_inspector = ui.card().classes('w-full studio-card')
                composition_table = ui.card().classes('w-full studio-card')

            with ui.tab_panel(tab_evidence).classes('w-full gap-3'):
                evidence_contract_card = ui.card().classes('w-full studio-card')
                ecosystem_card = ui.card().classes('w-full studio-card')
                sdk_card = ui.card().classes('w-full studio-card')
                improvements_card = ui.card().classes('w-full studio-card')
                representation_table = ui.card().classes('w-full studio-card')
                raw_trace = ui.expansion('Raw trace JSON', value=False).classes('w-full studio-card')

            with ui.tab_panel(tab_artifacts).classes('w-full gap-3'):
                artifacts_card = ui.card().classes('w-full studio-card')
                dissertation_pack_card = ui.card().classes('w-full studio-card')

    with operator_inspector:
        ui.label('Инспектор операторов').classes('text-subtitle1 studio-title')
        with ui.row().classes('w-full gap-2'):
            op_view_mode = ui.select(['timeline', 'table', 'graph'], value='timeline', label='Вид').props('outlined dense')
            op_severity = ui.select(['all', 'critical', 'warning', 'ok'], value='all', label='Фильтр важности').props('outlined dense')
            op_hide_ok = ui.switch('Скрыть ok', value=False).props('aria-label=\"Hide ok\"')
        op_search = ui.input('Поиск по оператору/сигналу/полям').props('outlined dense clearable')
        operator_select = ui.select([], label='Оператор').props('outlined dense')
        operator_meta = ui.markdown('Запусти pipeline и выбери оператор.')
        operator_details = ui.code('{}', language='json').classes('w-full')
    with artifacts_card:
        ui.label('Визуальные артефакты').classes('text-subtitle1 studio-title')
        ui.label('Последние скриншоты из browser-visual-check').classes('studio-note')
        gallery = ui.row().classes('w-full gap-2 flex-wrap')
    with dissertation_pack_card:
        ui.label('Пакет артефактов диссертации').classes('text-subtitle1 studio-title')
        ui.label('Структура для глав берется отсюда, но главный продукт остается единая Studio.').classes('studio-note')
        artifact_pack_box = ui.column().classes('w-full')

    controls.update(
        {
            'dataset_select': dataset_select,
            'scenario_select': scenario_select,
            'preset_select': preset_select,
            'sample_idx': sample_idx,
            'use_overrides': use_overrides,
            'p_r': p_r,
            'u_m': u_m,
            'i_p': i_p,
            'd_l': d_l,
            'chi_r': chi_r,
            'chi_r_crit': chi_r_crit,
            'chi_auto': chi_auto,
            'trace_valid': trace_valid,
            'bench_dataset': bench_dataset,
            'bench_access': bench_access,
            'bench_kind': bench_kind,
            'op_view_mode': op_view_mode,
            'op_severity': op_severity,
            'op_hide_ok': op_hide_ok,
            'op_search': op_search,
            'plan_widgets': (w_p, w_u, w_i, w_d, w_r, t1, t2, t3, t4, i_min, d_max, baseline_access, reduction_strategy),
        }
    )

    def _filter_operator_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        sev = str(controls['op_severity'].value or 'all')
        hide_ok = bool(controls['op_hide_ok'].value)
        needle = str(controls['op_search'].value or '').strip().lower()
        filtered: list[dict[str, Any]] = []
        for row in rows:
            row_sev = str(row.get('severity', 'ok'))
            if hide_ok and row_sev == 'ok':
                continue
            if sev != 'all' and row_sev != sev:
                continue
            if needle:
                blob = ' '.join(
                    [
                        str(row.get('operator', '')),
                        str(row.get('signal', '')),
                        str(row.get('takes_from', '')),
                        str(row.get('outputs', '')),
                    ]
                ).lower()
                if needle not in blob:
                    continue
            filtered.append(row)
        return filtered

    def _render_operator_inspector() -> None:
        rows = state.get('op_rows_view') or state.get('op_rows') or []
        names = [str(r.get('operator', '')) for r in rows]
        current = str(operator_select.value) if operator_select.value else ''
        operator_select.options = names
        if names and current not in names:
            operator_select.set_value(names[0])
            current = names[0]
        row = next((r for r in rows if str(r.get('operator')) == current), None)
        if not row:
            operator_meta.content = 'Нет данных по оператору.'
            operator_details.content = '{}'
            return
        operator_meta.content = (
            f"**severity:** `{row.get('severity')}`  \n"
            f"**signal:** `{row.get('signal')}`  \n"
            f"**takes_from:** `{row.get('takes_from')}`  \n"
            f"**outputs:** `{row.get('outputs')}`  \n"
            f"**status:** `{row.get('status')}`  \n"
            f"**uses:** `{row.get('uses')}`"
        )
        operator_details.content = json.dumps(row.get('details', {}), ensure_ascii=False, indent=2)

    def _collect_overrides() -> WhatIfOverrides | None:
        if not bool(controls['use_overrides'].value):
            return None
        return WhatIfOverrides(
            predicted_risk=_to_float(controls['p_r'].value, 0.5),
            uncertainty=_to_float(controls['u_m'].value, 0.3),
            i_pre=_to_float(controls['i_p'].value, 0.7),
            reduction_loss=_to_float(controls['d_l'].value, 0.1),
            chi_r=_to_int(controls['chi_r'].value, 0),
            chi_r_crit=_to_int(controls['chi_r_crit'].value, 0),
            chi_auto=bool(controls['chi_auto'].value),
            trace_valid=bool(controls['trace_valid'].value),
        )

    def _apply_plan() -> None:
        (w_pv, w_uv, w_iv, w_dv, w_rv, t1v, t2v, t3v, t4v, i_minv, d_maxv, bav, rsv) = controls['plan_widgets']
        plan.w_p = _to_float(w_pv.value, plan.w_p)
        plan.w_u = _to_float(w_uv.value, plan.w_u)
        plan.w_I = _to_float(w_iv.value, plan.w_I)
        plan.w_Delta = _to_float(w_dv.value, plan.w_Delta)
        plan.w_R = _to_float(w_rv.value, plan.w_R)
        plan.theta_1 = _to_float(t1v.value, plan.theta_1)
        plan.theta_2 = _to_float(t2v.value, plan.theta_2)
        plan.theta_3 = _to_float(t3v.value, plan.theta_3)
        plan.theta_4 = _to_float(t4v.value, plan.theta_4)
        plan.i_min = _to_float(i_minv.value, plan.i_min)
        plan.delta_max = _to_float(d_maxv.value, plan.delta_max)
        plan.baseline_access = str(bav.value)
        plan.reduction_strategy = str(rsv.value)

    def _refresh_operator_views() -> None:
        op_rows = state.get('op_rows') or []
        op_rows_view = _filter_operator_rows(op_rows)
        state['op_rows_view'] = op_rows_view
        view_mode = str(controls['op_view_mode'].value or 'timeline')

        operators_table.clear()
        with operators_table:
            ui.label('Трасса операторов: что берется из системы').classes('text-subtitle1 studio-title')
            ui.label(f"rows: {len(op_rows_view)}/{len(op_rows)} | view: {view_mode}").classes('studio-note')
            if not op_rows_view:
                ui.label('Нет операторов после фильтрации. Ослабь фильтр важности/скрытие/поиск.').classes('text-warning')
                _render_operator_inspector()
                return

            if view_mode == 'graph':
                ui.echart(_operators_flow_option(op_rows_view)).classes('w-full h-72')
            elif view_mode == 'table':
                ui.table(
                    columns=[
                        {'name': 'operator', 'label': 'Оператор', 'field': 'operator'},
                        {'name': 'severity', 'label': 'Severity', 'field': 'severity'},
                        {'name': 'signal', 'label': 'Signal', 'field': 'signal'},
                        {'name': 'takes_from', 'label': 'Takes from', 'field': 'takes_from'},
                        {'name': 'outputs', 'label': 'Outputs', 'field': 'outputs'},
                        {'name': 'status', 'label': 'Статус', 'field': 'status'},
                    ],
                    rows=op_rows_view,
                ).classes('w-full')
            else:
                ui.html(_operator_timeline_html(op_rows_view)).classes('w-full')
                with ui.row().classes('w-full gap-2'):
                    for row in op_rows_view:
                        sev = str(row.get('severity', 'ok'))
                        sev_color = {'ok': '#16a34a', 'warning': '#d97706', 'critical': '#dc2626'}.get(sev, '#334155')
                        with ui.card().classes('w-full lg:w-[24%]'):
                            ui.html(
                                f"<div style='font-weight:700'>{row.get('operator')}</div>"
                                f"<div style='margin-top:6px;font-size:12px'>"
                                f"<span style='padding:2px 8px;border-radius:999px;background:{sev_color};color:white'>{sev}</span>"
                                f"</div>"
                            )
                            ui.label(f"in: {row.get('takes_from')}").classes('text-caption')
                            ui.label(f"out: {row.get('outputs')}").classes('text-caption')
                            ui.label(f"status: {row.get('status')}").classes('text-caption')
                            ui.label(f"signal: {row.get('signal')}").classes('text-caption')
            with ui.expansion('Детали оператора (JSON)', value=False).classes('w-full'):
                ui.code(json.dumps(op_rows_view, ensure_ascii=False, indent=2), language='json').classes('w-full')
        _render_operator_inspector()

    def _render_evidence_pack() -> None:
        contract = _load_json(ROOT / 'reports' / 'chapter2' / 'explain_plan_hash.json') or {}
        sample = _load_json(ROOT / 'reports' / 'chapter2' / 'sample_113_report.json') or {}
        article_path = ROOT / 'reports' / 'reproducibility_artifacts' / 'article_insert.md'
        manifest = _load_json(ROOT / 'reports' / 'reproducibility_artifacts' / 'manifest.json') or {}
        matrix = build_evidence_matrix(load_ecosystem_registry())
        non_synth = _non_synthetic_rows()
        scenario_rows = _read_csv_rows(ROOT / 'reports' / 'chapter5' / 'scenario_run_summary.csv')
        quant_rows = _read_csv_rows(ROOT / 'reports' / 'chapter5' / 'scenario_quantitative_summary.csv')

        evidence_contract_card.clear()
        with evidence_contract_card:
            ui.label('Контракт доказательности').classes('text-subtitle1 studio-title')
            with ui.row().classes('w-full gap-2'):
                for title, value in [
                    ('SHA YAML главы 2', str(contract.get('sha256', '-'))[:16]),
                    ('sample_113 SHA source', str(sample.get('explain_plan_hash', '-'))[:16]),
                    ('Артефакты', len(manifest.get('artifacts', []))),
                    ('Article insert', 'ready' if article_path.exists() else 'missing'),
                ]:
                    with ui.card().classes('w-full lg:w-[23%]'):
                        ui.label(title).classes('text-caption')
                        ui.label(str(value)).classes('text-body2')
            ui.markdown(
                f"- `plan_path`: `{contract.get('plan_path', '-')}`\n"
                f"- `required_trace_fields`: `{contract.get('required_trace_fields', [])}`\n"
                f"- `sample_113 action`: `{sample.get('action', '-')}`\n"
                f"- `sample_113 P(malignant)`: `{_num(sample.get('P(malignant)'), 6)}`"
            )

        ecosystem_card.clear()
        with ecosystem_card:
            ui.label('Реестр внешних модулей').classes('text-subtitle1 studio-title')
            ui.table(
                columns=[
                    {'name': 'registry_id', 'label': 'Модуль', 'field': 'registry_id'},
                    {'name': 'status', 'label': 'Статус', 'field': 'status'},
                    {'name': 'evidence_level', 'label': 'Доказательность', 'field': 'evidence_level'},
                    {'name': 'adapter', 'label': 'Адаптер', 'field': 'adapter'},
                    {'name': 'run_allowed', 'label': 'Запуск', 'field': 'run_allowed'},
                    {'name': 'quantitative_claim_allowed', 'label': 'Колич. claim', 'field': 'quantitative_claim_allowed'},
                ],
                rows=matrix,
            ).classes('w-full')
            with ui.expansion('Карточки исходных репозиториев', value=False).classes('w-full'):
                for row in matrix:
                    with ui.card().classes('w-full'):
                        ui.label(f"{row['registry_id']} | {row['status']}").classes('studio-title')
                        ui.markdown(
                            f"- `source_repo`: `{row['source_repo']}`\n"
                            f"- `source_artifact`: `{row['source_artifact']}`\n"
                            f"- `claim_scope`: {row['claim_scope']}"
                        )

        sdk_card.clear()
        with sdk_card:
            ui.label('Маршрут SDK / API / Docker').classes('text-subtitle1 studio-title')
            ui.label('Один внешний модуль подключается через один контракт: registry -> adapter -> artifact -> report/action.').classes('studio-note')
            with ui.row().classes('w-full gap-2'):
                for title, body in [
                    ('SDK', '`fuzzyxai/sdk/base_adapter.py` + `contracts.py`'),
                    ('API', '`/v1/explain` и `/v1/risk-action` в `api/openapi.yaml`'),
                    ('Deploy', '`deploy/docker-compose.yml`, `Dockerfile.api`, `Dockerfile.studio`'),
                    ('Register', '`scripts/register_external_module.py` + templates'),
                ]:
                    with ui.card().classes('w-full lg:w-[24%]'):
                        ui.label(title).classes('studio-title')
                        ui.markdown(body)
            ui.label('Сценарные прогоны главы 5').classes('text-subtitle2 studio-title')
            if scenario_rows:
                ui.echart(_scenario_status_option(scenario_rows)).classes('w-full h-56')
                display = [
                    {
                        'registry_id': r.get('registry_id'),
                        'adapter_called': r.get('adapter_called'),
                        'output_type': r.get('output_type'),
                        'action': r.get('action'),
                        'status': r.get('status'),
                    }
                    for r in scenario_rows
                ]
                ui.table(columns=[{'name': k, 'label': k, 'field': k} for k in display[0].keys()], rows=display).classes('w-full')
            else:
                ui.label('scenario_run_summary.csv not found').classes('text-warning')
            if quant_rows:
                with ui.expansion('Доступность количественных метрик (без фиктивных чисел)', value=False).classes('w-full'):
                    display_q = [
                        {
                            'registry_id': r.get('registry_id'),
                            'метрика': r.get('baseline_metric'),
                            'доступно': r.get('quantitative_comparison_available'),
                            'status': r.get('status'),
                        }
                        for r in quant_rows
                    ]
                    ui.table(columns=[{'name': k, 'label': k, 'field': k} for k in display_q[0].keys()], rows=display_q).classes('w-full')

        improvements_card.clear()
        with improvements_card:
            ui.label('Улучшения на реальных строках').classes('text-subtitle1 studio-title')
            ui.echart(_non_synthetic_option(non_synth)).classes('w-full h-64')
            if non_synth:
                display_rows = [
                    {
                        'dataset': r['dataset'],
                        'agreement_gain': _num(r['agreement_gain'], 4),
                        'false_auto_accept_drop': _num(r['false_auto_accept_drop'], 4),
                        'full_agreement_ref': _num(r['full_agreement_ref'], 4),
                        'threshold_agreement_ref': _num(r['threshold_agreement_ref'], 4),
                        'full_false_auto_accept': _num(r['full_false_auto_accept'], 4),
                        'threshold_false_auto_accept': _num(r['threshold_false_auto_accept'], 4),
                    }
                    for r in non_synth
                ]
                ui.table(columns=[{'name': k, 'label': k, 'field': k} for k in display_rows[0].keys()], rows=display_rows).classes('w-full')

    def _render_artifact_pack() -> None:
        artifact_pack_box.clear()
        retained = _read_csv_rows(ROOT / 'dissertation_artifacts' / 'retained_figures_manifest.csv')
        figure_map = _read_csv_rows(ROOT / 'dissertation_artifacts' / 'figure_to_text_map.csv')
        scenario = _read_csv_rows(ROOT / 'dissertation_artifacts' / 'chapter5' / 'table_5_scenario_run_summary.csv')
        manifest = _load_json(ROOT / 'dissertation_artifacts' / 'artifact_manifest_sha256.json') or {}
        with artifact_pack_box:
            with ui.row().classes('w-full gap-2'):
                for title, value in [
                    ('итоговые рисунки', len(retained)),
                    ('figure-to-text links', len(figure_map)),
                    ('сценарии главы 5', len(scenario)),
                    ('sha256-файлы', len(manifest)),
                ]:
                    with ui.card().classes('w-full lg:w-[24%]'):
                        ui.label(title).classes('text-caption')
                        ui.label(str(value)).classes('text-body2 studio-title')
            if retained:
                ui.label('Итоговые рисунки: что оставляем и что перерисовываем').classes('text-subtitle2 studio-title')
                rows = [{k: r.get(k, '') for k in ['chapter', 'figure_id', 'status', 'type', 'source_spec', 'will_be_redrawn']} for r in retained]
                ui.table(columns=[{'name': k, 'label': k, 'field': k} for k in rows[0].keys()], rows=rows).classes('w-full')
            if figure_map:
                with ui.expansion('Карта рисунок -> текст', value=False).classes('w-full'):
                    rows = [{k: r.get(k, '') for k in ['figure_id', 'chapter', 'section', 'insert_after_heading', 'required']} for r in figure_map]
                    ui.table(columns=[{'name': k, 'label': k, 'field': k} for k in rows[0].keys()], rows=rows).classes('w-full')
            if scenario:
                with ui.expansion('Маршруты сценариев главы 5', value=False).classes('w-full'):
                    rows = [{k: r.get(k, '') for k in ['registry_id', 'adapter_called', 'output_type', 'action', 'status']} for r in scenario]
                    ui.table(columns=[{'name': k, 'label': k, 'field': k} for k in rows[0].keys()], rows=rows).classes('w-full')
            ui.markdown(
                '- `make dissertation-artifacts` собирает PNG/CSV/MD для вставки.\n'
                '- GUI показывает эти артефакты как результат единого pipeline, а не как отдельный блокнот.\n'
                '- Основные скриншоты остаются в appendix/browser-check, схемы для глав идут через `diagram_specs`.'
            )

    def _render_case(st: dict[str, Any]) -> None:
        risk = st.get('risk', {})
        model = st.get('model', {})
        unc = st.get('uncertainty', {})
        action = str(risk.get('action', ''))
        chi_auto_now = _chi_auto_for_action(st, action)
        color = _action_color(action)
        reason_text = explain_action_text(
            action,
            chi_r_crit=int(risk.get('chi_R_crit', 0)),
            chi_auto=chi_auto_now,
            rho=float(risk.get('rho', 0.0)),
        )

        project_map_card.clear()
        with project_map_card:
            ui.label('Единая карта проекта').classes('text-subtitle1 studio-title')
            ui.label('Все вкладки ниже показывают один маршрут: от входного случая до действия и воспроизводимого отчета.').classes('studio-note')
            ui.html(_unified_route_html(st)).classes('w-full')
            ui.separator()
            ui.html(_system_layers_html()).classes('w-full')

        summary.clear()
        with summary:
            ui.label('Итог по кейсу').classes('text-subtitle1 studio-title')
            ui.html(
                f"<div style='padding:10px 12px;border-radius:10px;background:{color};color:white;font-weight:700;display:inline-block'>"
                f"Действие: {action.upper()}</div>"
            )
            ui.label(f"Причина: {risk.get('reason', '-')}").classes('text-body2')
            ui.label(reason_text).classes('text-body2')
            with ui.expansion('Почему выбрано это действие (правило политики)', value=False).classes('w-full'):
                ui.markdown(_action_legend_md(st, plan, chi_auto_now))
            ui.markdown(
                f"- dataset: `{st.get('dataset', {}).get('name')}` ({st.get('dataset', {}).get('status')})\n"
                f"- predicted_risk: `{_num(model.get('predicted_risk', model.get('p_malignant')) )}`\n"
                f"- I_pre: `{_num(st.get('explanation', {}).get('I_pre'))}`\n"
                f"- selected_class: `{unc.get('selected_class')}`\n"
                f"- Delta: `{_num(unc.get('delta'))}`\n"
                f"- rho: `{_num(risk.get('rho'))}`\n"
                f"- chi_R / chi_R_crit: `{risk.get('chi_R', 0)} / {risk.get('chi_R_crit', 0)}`\n"
                f"- chi_Auto(E_action): `{chi_auto_now}`"
            )

        ds_name = str(st.get('dataset', {}).get('name', 'unknown'))
        ds_status, ds_color = _status_badge(str(st.get('dataset', {}).get('status', 'UNKNOWN')))
        ds_summary = _load_json(ROOT / 'reports' / 'datasets' / ds_name / 'summary.json') or {}
        dataset_preview.content = (
            f"**dataset:** `{ds_name}`  \n"
            f"**status:** <span class='studio-chip' style='background:{ds_color};color:white'>{ds_status}</span>  \n"
            f"- `rows`: `{ds_summary.get('n', '-')}`  \n"
            f"- `domain`: `{ds_summary.get('domain', '-')}`  \n"
            f"- `model_accuracy`: `{_num(ds_summary.get('model_accuracy'))}`  \n"
            f"- `roc_auc`: `{_num(ds_summary.get('model_roc_auc'))}`  \n"
            f"- `agreement_proxy`: `{_num(ds_summary.get('agreement_proxy'))}`  \n"
            f"- `agreement_proxy_applicable`: `{ds_summary.get('agreement_proxy_applicable')}`  \n"
            f"- `missed_critical_ruptures`: `{ds_summary.get('missed_critical_ruptures', '-')}`  \n"
            f"- `critical_rupture_recall`: `{_num(ds_summary.get('critical_rupture_recall'))}`"
        )

        key_panels.clear()
        with key_panels:
            for title, body in [
                ('Модель', f"pred={model.get('prediction')} | true={model.get('true_y')}"),
                ('E_k', f"u={_num(st.get('explanation', {}).get('E_model', {}).get('u'))} | I_pre={_num(st.get('explanation', {}).get('I_pre'))}"),
                ('Topos', f"chi_Auto={chi_auto_now}"),
                ('Наблюдатель', f"rho={_num(risk.get('rho'))} | action={action}"),
            ]:
                with ui.card().classes('min-w-[23%]'):
                    ui.label(title).classes('text-caption')
                    ui.label(body).classes('text-body2')

        membership_chart.options.clear()
        pred_risk = float(model.get('predicted_risk', model.get('p_malignant', 0.5)))
        mu_now = st.get('explanation', {}).get('E_model', {}).get('mu', {}) or {}
        membership_chart.options.update(
            _membership_option(
                pred_risk,
                mu_now,
            )
        )
        membership_chart.update()
        membership_meta.content = (
            f"`p`: `{_num(pred_risk, 4)}` | "
            f"`mu_low`: `{_num(mu_now.get('low', 0.0), 3)}` | "
            f"`mu_med`: `{_num(mu_now.get('medium', 0.0), 3)}` | "
            f"`mu_high`: `{_num(mu_now.get('high', 0.0), 3)}`"
        )
        contrib = risk.get('breakdown', {}).get('contributions', {}) or {}
        risk_contrib_chart.options.clear()
        risk_contrib_chart.options.update(_risk_contrib_option(contrib, risk))
        risk_contrib_chart.update()
        comp = risk.get('components', {}) or {}
        w = risk.get('weights', {}) or comp.get('weights', {}) or {}
        comp_vals = {
            'predicted_risk': float(w.get('predicted_risk', 0.0)) * float(comp.get('predicted_risk', 0.0)),
            'uncertainty': float(w.get('uncertainty', 0.0)) * float(comp.get('uncertainty', 0.0)),
            'interpretability_gap': float(w.get('interpretability_gap', 0.0)) * float(comp.get('interpretability_gap', 0.0)),
            'reduction_loss': float(w.get('reduction_loss', 0.0)) * float(comp.get('reduction_loss', 0.0)),
            'chi_R': float(w.get('diagnostic', 0.0)) * float(comp.get('diagnostic', comp.get('chi_R', 0.0))),
        }
        top_factor = max(comp_vals.items(), key=lambda kv: kv[1])[0]
        risk_meta.content = f"`total`: `{_num(sum(comp_vals.values()), 4)}` | `top_factor`: `{top_factor}`"
        dist_chart.options.clear()
        dist_chart.options.update(_dataset_risk_distribution_option(ds_name, pred_risk))
        dist_chart.update()
        pred_path = ROOT / 'reports' / 'datasets' / ds_name / 'predictions.csv'
        n_rows = '-'
        if pred_path.exists():
            try:
                with pred_path.open(encoding='utf-8') as f:
                    n_rows = str(sum(1 for _ in f) - 1)
            except Exception:
                n_rows = '-'
        dist_meta.content = f"`dataset`: `{ds_name}` | `n`: `{n_rows}` | `case_risk`: `{_num(pred_risk, 4)}`"

        rows = route_rows(st)
        pipeline_table.clear()
        with pipeline_table:
            ui.label('Маршрут pipeline').classes('text-subtitle1 studio-title')
            ui.html(_route_stepper_html(rows)).classes('w-full')
            ui.table(
                columns=[
                    {'name': 'step', 'label': 'Шаг', 'field': 'step'},
                    {'name': 'state', 'label': 'Состояние', 'field': 'state'},
                ],
                rows=rows,
            ).classes('w-full')

        state['op_rows'] = build_operator_trace(st)
        _refresh_operator_views()

        comp_rows = composition_rows(st)
        composition_table.clear()
        with composition_table:
            ui.label('Сертифицированный путь / разрыв').classes('text-subtitle1 studio-title')
            if comp_rows:
                ui.table(columns=[{'name': k, 'label': k, 'field': k} for k in comp_rows[0].keys()], rows=comp_rows).classes('w-full')

        rep_rows = representation_rows(st)
        representation_table.clear()
        with representation_table:
            ui.label('Выбор представления').classes('text-subtitle1 studio-title')
            if rep_rows:
                ui.table(columns=[{'name': k, 'label': k, 'field': k} for k in rep_rows[0].keys()], rows=rep_rows).classes('w-full')
        _render_evidence_pack()
        _render_artifact_pack()

        raw_trace.clear()
        with raw_trace:
            ui.code(json.dumps(st, ensure_ascii=False, indent=2), language='json').classes('w-full')

        gallery.clear()
        shot_dir_candidates = [
            ROOT / 'reports' / 'browser_visual_check_latest',
            ROOT / 'reports' / 'browser_visual_check',
        ]
        shot_dir = next((d for d in shot_dir_candidates if d.exists() and any(d.glob('*.png'))), None)
        shots = sorted(shot_dir.glob('*.png')) if shot_dir is not None else []
        with gallery:
            if not shots:
                ui.label('Скриншоты пока не найдены. Запусти `make browser-visual-check`.').classes('studio-note')
            else:
                ui.label(f'Источник: {shot_dir.name} | кадров: {len(shots)}').classes('studio-note')
                for p in shots[:9]:
                    with ui.card().classes('w-full lg:w-[32%]'):
                        ui.image(str(p)).classes('w-full')
                        ui.label(p.name).classes('text-caption')

    def run_case() -> None:
        try:
            state['scene'] = str(state.get('scene', 'custom') or 'custom')
            _apply_plan()
            ds = str(controls['dataset_select'].value)
            sc = str(controls['scenario_select'].value)
            pr = str(controls['preset_select'].value)
            idx = _to_int(controls['sample_idx'].value, 0)

            base = build_case_state(service, sc, sample_index=idx, dataset_mode=ds)
            tuned = apply_named_preset(base, plan, pr) if pr != 'none' else base
            ov = _collect_overrides()
            if ov is not None:
                from fuzzyxai.studio import recompute_case_state

                tuned = recompute_case_state(tuned, plan, ov)
            state['case'] = tuned
            _render_case(tuned)
            _save_last_case(tuned)
            demo_flow.content = _demo_flow_html(str(state.get('scene', 'custom')))
            load_benchmark()
        except Exception as exc:
            summary.clear()
            with summary:
                ui.label('Ошибка выполнения pipeline').classes('text-subtitle1 text-negative')
                ui.label(str(exc)).classes('text-negative')
            ui.notify('Pipeline error: см. блок Итог по кейсу', type='negative')

    def load_benchmark() -> None:
        bench_box.clear()
        ds = str(controls['bench_dataset'].value)
        access = str(controls['bench_access'].value)
        kind = str(controls['bench_kind'].value)
        fallback_used = False

        # Reset chart first to avoid stale baseline visualization
        bench_chart.options.clear()
        bench_chart.options.update(_baseline_compare_option([]))
        bench_chart.update()

        if kind == 'baseline_comparison':
            path = ROOT / 'reports' / 'datasets' / ds / f'baseline_comparison_{access}.json'
            if not path.exists():
                path = ROOT / 'reports' / 'datasets' / ds / 'baseline_comparison.json'
                fallback_used = True
        else:
            path = ROOT / 'reports' / 'structure_aware_benchmark' / f'{ds}.json'

        with bench_box:
            ui.label(f'Отчет: {path}').classes('text-caption')
            if fallback_used:
                ui.label('Selected access-specific report not found; fallback to default baseline_comparison.json').classes('studio-note')
            if kind != 'baseline_comparison':
                ui.label('Chart is shown for baseline_comparison mode only.').classes('studio-note')
            data = _load_json(path)
            if not data:
                ui.label('report not found').classes('text-negative')
                return
            rows = data.get('rows') or data.get('scenarios') or []
            if isinstance(rows, list) and rows:
                ui.table(columns=[{'name': k, 'label': k, 'field': k} for k in rows[0].keys()], rows=rows).classes('w-full')
                if kind == 'baseline_comparison':
                    bench_chart.options.clear()
                    bench_chart.options.update(_baseline_compare_option(rows))
                    bench_chart.update()
            else:
                ui.code(json.dumps(data, ensure_ascii=False, indent=2), language='json').classes('w-full')
            if kind == 'baseline_comparison':
                ui.markdown('- `native`: baseline only native inputs\n- `equal_guardrail`: baseline also gets `chi_R_crit`')

    def export_current() -> None:
        export_info.clear()
        st = state.get('case')
        with export_info:
            if not st:
                ui.label('run pipeline first').classes('text-negative')
                return
            paths = export_defense_case(st, plan, out_dir=ROOT / 'reports' / 'layered_demo', stem='defense_case')
            ui.label('export done').classes('text-positive')
            for k, v in paths.items():
                ui.label(f'{k}: {v}').classes('text-body2')

    def _apply_scene(scene: str) -> None:
        state['scene'] = scene
        if scene == 'safe':
            controls['dataset_select'].set_value('breast_cancer')
            controls['scenario_select'].set_value('safe')
            controls['preset_select'].set_value('safe_accept')
            controls['op_view_mode'].set_value('timeline')
            controls['bench_dataset'].set_value('breast_cancer')
            controls['bench_access'].set_value('native')
            controls['bench_kind'].set_value('baseline_comparison')
        elif scene == 'audit':
            controls['dataset_select'].set_value('synthetic_ruptures')
            controls['scenario_select'].set_value('ambiguous')
            controls['preset_select'].set_value('need_more_data')
            controls['op_view_mode'].set_value('table')
            controls['bench_dataset'].set_value('synthetic_ruptures')
            controls['bench_access'].set_value('native')
            controls['bench_kind'].set_value('baseline_comparison')
        elif scene == 'block':
            controls['dataset_select'].set_value('synthetic_ruptures')
            controls['scenario_select'].set_value('rupture')
            controls['preset_select'].set_value('critical_rupture')
            controls['op_view_mode'].set_value('graph')
            controls['bench_dataset'].set_value('synthetic_ruptures')
            controls['bench_access'].set_value('equal_guardrail')
            controls['bench_kind'].set_value('baseline_comparison')
        controls['use_overrides'].set_value(False)
        run_case()

    def _reset_default() -> None:
        state['scene'] = 'custom'
        controls['dataset_select'].set_value('breast_cancer')
        controls['scenario_select'].set_value('safe')
        controls['preset_select'].set_value('none')
        controls['sample_idx'].set_value(0)
        controls['use_overrides'].set_value(False)
        controls['op_view_mode'].set_value('timeline')
        controls['bench_dataset'].set_value('synthetic_ruptures')
        controls['bench_access'].set_value('native')
        controls['bench_kind'].set_value('baseline_comparison')
        run_case()

    def _set_custom_and_run() -> None:
        state['scene'] = 'custom'
        run_case()

    run_btn.on_click(run_case)
    for preset_name, btn in quick_preset_buttons.items():
        btn.on_click(lambda _e=None, name=preset_name: (state.__setitem__('scene', 'custom'), controls['preset_select'].set_value(name), controls['use_overrides'].set_value(False), run_case()))
    scene_safe_btn.on_click(lambda _e: _apply_scene('safe'))
    scene_audit_btn.on_click(lambda _e: _apply_scene('audit'))
    scene_block_btn.on_click(lambda _e: _apply_scene('block'))
    scene_reset_btn.on_click(lambda _e: _reset_default())
    apply_plan_btn.on_click(run_case)
    bench_load_btn.on_click(load_benchmark)
    export_btn.on_click(export_current)
    operator_select.on_value_change(lambda _e: _render_operator_inspector())
    op_view_mode.on_value_change(lambda _e: _refresh_operator_views())
    op_severity.on_value_change(lambda _e: _refresh_operator_views())
    op_hide_ok.on_value_change(lambda _e: _refresh_operator_views())
    op_search.on_value_change(lambda _e: _refresh_operator_views())

    controls['dataset_select'].on_value_change(lambda _e: _set_custom_and_run())
    controls['scenario_select'].on_value_change(lambda _e: _set_custom_and_run())
    controls['preset_select'].on_value_change(lambda _e: _set_custom_and_run())
    controls['sample_idx'].on_value_change(lambda _e: _set_custom_and_run())
    controls['bench_dataset'].on_value_change(lambda _e: load_benchmark())
    controls['bench_access'].on_value_change(lambda _e: load_benchmark())
    controls['bench_kind'].on_value_change(lambda _e: load_benchmark())
    controls['use_overrides'].on_value_change(lambda _e: _set_custom_and_run())
    controls['p_r'].on_value_change(lambda _e: _set_custom_and_run() if bool(controls['use_overrides'].value) else None)
    controls['u_m'].on_value_change(lambda _e: _set_custom_and_run() if bool(controls['use_overrides'].value) else None)
    controls['i_p'].on_value_change(lambda _e: _set_custom_and_run() if bool(controls['use_overrides'].value) else None)
    controls['d_l'].on_value_change(lambda _e: _set_custom_and_run() if bool(controls['use_overrides'].value) else None)
    controls['chi_r'].on_value_change(lambda _e: _set_custom_and_run() if bool(controls['use_overrides'].value) else None)
    controls['chi_r_crit'].on_value_change(lambda _e: _set_custom_and_run() if bool(controls['use_overrides'].value) else None)
    controls['chi_auto'].on_value_change(lambda _e: _set_custom_and_run() if bool(controls['use_overrides'].value) else None)
    controls['trace_valid'].on_value_change(lambda _e: _set_custom_and_run() if bool(controls['use_overrides'].value) else None)

    ui.timer(0.1, run_case, once=True)
    ui.run(title='FuzzyXAI Studio', port=port, reload=False, show=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8097)
    args = parser.parse_args()
    run(port=int(args.port))


if __name__ == '__main__':
    main()
