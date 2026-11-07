"""Dependency-free, offline, XSS-safe visualization of a verified FuzzyXAI receipt.

This renderer does not run a model or generate scientific measurements. Every
value is read from the *already verified* structural certificate.
"""
from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any, Mapping

from .doctoral_certification import project_for_audience, verify_certificate


def _s(value: Any) -> str:
    return escape(str(value if value is not None else '—'), quote=True)


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    head = ''.join('<th scope="col">'+_s(name)+'</th>' for name in headers)
    body = '\n'.join('<tr>'+''.join('<td>'+_s(value)+'</td>' for value in row)+'</tr>' for row in rows)
    return f'<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def render_doctoral_report(receipt: Mapping[str, Any], output: str | Path | None = None) -> str:
    """Render a static self-contained HTML report; reject a tampered receipt.

    The audience sections are all projections of the same verified scientific
    evidence; the renderer never extrapolates predictive/clinical accuracy.
    """
    check = verify_certificate(receipt)
    if not check.get('valid'):
        raise ValueError('certificate failed replay/integrity verification')
    auditor = project_for_audience(receipt, 'auditor')
    developer = project_for_audience(receipt, 'developer')
    user = project_for_audience(receipt, 'end_user')
    status = str(user['status'])
    style = {'valid': 'ok', 'invalid': 'bad', 'insufficient_evidence': 'warn'}.get(status, 'warn')
    issue_rows = []
    for i in auditor['issues']:
        issue_rows.append([i.get('kind'), i.get('code'), i.get('path'), i.get('detail')])
    if not issue_rows:
        issue_rows = [['—', 'Замечаний по проверяемым контрактам нет', '—', '—']]
    weights = (auditor.get('risk') or {}).get('rho_recomputed')
    tax = auditor.get('taxonomy') or {}
    selected = auditor.get('representation') or {}
    graph = auditor.get('graph') or {}
    delta = auditor.get('alignment') or {}
    cut = auditor.get('diagnostic_cut') or {}
    statuses = _table(['Контракт', 'Наблюдаемое значение'], [
        ['Координаты метода', f"{tax.get('source', '—')} / {tax.get('object', '—')} / {tax.get('audience', '—')}"],
        ['Точка каталога', tax.get('method_id', '—')],
        ['Класс представления', selected.get('selected_class', '—')],
        ['Профиль неопределённости', ', '.join(selected.get('profile', []))],
        ['Согласование Γ', delta.get('gamma', 'не применяется')],
        ['Редукция Δ', delta.get('delta', 'не применяется')],
        ['Проверенный риск ρ', weights if weights is not None else 'не вычислен'],
        ['Узлы / рёбра / утверждения', f"{graph.get('n_nodes', 0)} / {graph.get('n_edges', 0)} / {graph.get('n_claims', 0)}"],
    ])
    steps = [
        ('01', 'Таксономия', 'источник / объект / адресат'),
        ('02', 'Объяснение', 'правила / активации / происхождение'),
        ('03', 'Согласование', 'Γ / Δ / представление'),
        ('04', 'Наблюдатель', 'ρ / запрет / действие'),
        ('05', 'Доказательство', 'граф / повторная проверка'),
    ]
    route = ''.join('<li><span class="step-number">'+number+'</span><span><b>'+_s(name)+'</b><small>'+_s(desc)+'</small></span></li>' for number, name, desc in steps)
    diagnostic_detail = _table(['Характеристика', 'Результат'], [
        ['Число обнаруженных обязательств', len(auditor['issues'])],
        ['Рекомендуемая диагностическая точка', ', '.join(cut.get('atoms', [])) or '—'],
        ['Точный поиск', 'да' if cut.get('optimality_proven') else 'нет'],
        ['Граница доверия', 'проверен вычислительный маршрут; качество модели не удостоверено'],
    ])
    html = f'''<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FuzzyXAI — доказательный маршрут {_s(status)}</title>
<style>
:root{{--ink:#1d2735;--muted:#526273;--line:#d4dce5;--blue:#174675;--paper:#fff;--light:#f5f7fa}}
*{{box-sizing:border-box}}body{{font-family:system-ui,-apple-system,Segoe UI,Arial,sans-serif;background:var(--light);color:var(--ink);line-height:1.57;margin:0}}
main{{max-width:1020px;margin:38px auto;padding:0 20px 56px}}
header{{background:var(--paper);padding:33px 35px;border:1px solid var(--line);border-radius:12px}}
h1{{font-size:2rem;letter-spacing:-.025em;margin:0 0 8px}}h2{{font-size:1.3rem;margin:0 0 15px}}p{{margin:8px 0 14px}}.muted,small{{color:var(--muted)}}
.panel{{background:var(--paper);border:1px solid var(--line);border-radius:12px;padding:26px 30px;margin-top:18px}}
.badge{{display:inline-block;padding:5px 12px;border-radius:4px;font-weight:700;letter-spacing:.02em;border:1px solid}}
.badge.ok{{background:#eef7f0;border-color:#5c8d6b;color:#1b5e32}}.badge.bad{{background:#fff2f0;border-color:#b65a55;color:#882622}}.badge.warn{{background:#fffae8;border-color:#bba058;color:#755b14}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}.metric{{border:1px solid var(--line);padding:15px 18px;border-radius:8px}}
.metric strong{{display:block;font-size:1.35rem}}.route{{list-style:none;padding:0;display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:8px}}
.route li{{border:1px solid var(--line);border-top:3px solid var(--blue);padding:12px;display:flex;align-items:start;gap:9px;border-radius:6px}}
.route small{{display:block;font-size:.73rem}}.step-number{{color:var(--blue);font-weight:700;font-size:.8rem}}table{{width:100%;border-collapse:collapse;font-size:.89rem}}
th,td{{border-bottom:1px solid var(--line);padding:9px;text-align:left;vertical-align:top;overflow-wrap:anywhere}}th{{background:var(--light);font-weight:700}}
summary{{cursor:pointer;font-weight:700;color:var(--blue)}}details{{padding-top:12px}}.mono{{font-family:ui-monospace,Consolas,monospace;font-size:.78rem;overflow-wrap:anywhere}}
footer{{font-size:.78rem;margin-top:24px;color:var(--muted)}}@media(max-width:750px){{.route{{grid-template-columns:1fr 1fr}}.grid{{grid-template-columns:1fr}}header,.panel{{padding:20px}}}}
@media print{{body{{background:#fff}}main{{margin:0;max-width:none}}.panel,header{{break-inside:avoid}}}}
</style></head>
<body><main>
<header><h1>FuzzyXAI · проверка объяснительного маршрута</h1>
<p class="muted">Структурная проверка методологии глав 1–4. Не является сертификатом качества предсказания или медицинской пригодности.</p>
<span class="badge {style}">{_s(status)}</span>
<div class="grid" style="margin-top:18px"><div class="metric"><small>Действие наблюдателя</small><strong>{_s(user['action'])}</strong></div>
<div class="metric"><small>Автоматическое принятие</small><strong>{'разрешено структурной проверкой' if user['auto_accept_permitted'] else 'не разрешено'}</strong></div></div>
</header>
<section class="panel"><h2>Маршрут доказательства</h2><ol class="route">{route}</ol></section>
<section class="panel"><h2>Метод и численные условия</h2>{statuses}</section>
<section class="panel"><h2>Диагностические нарушения</h2><p class="muted">Несоответствия вычислительных контрактов, не ошибки диагноза или исходной модели.</p>
{_table(['Тип','Код','Место','Причина'], issue_rows)}
<details><summary>Диагностическое сечение и ограничения</summary>{diagnostic_detail}</details></section>
<section class="panel"><h2>Воспроизводимость и аудит</h2>
<p>Повторная проверка целостности: <b>PASS</b></p><p>SHA256 рецепта доказательства:</p><p class="mono">{_s(receipt['sha256'])}</p>
<details><summary>Проверенные идентификаторы источника</summary>{_table(['Поле', 'Значение'], [[k,v] for k,v in (auditor.get('trace') or {{}}).items()])}</details>
<details><summary>Представление разработчика</summary>{_table(['Поле','Значение'], [[k,str(v)] for k,v in developer['risk'].items()])}</details>
</section>
<footer>SHA256 подтверждает согласованность полученного содержимого при повторном расчёте, но не личность автора. { _s(user['limitation']) }. Отсутствующие свидетельства не подменяются нулями.</footer>
</main></body></html>'''
    if output is not None:
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(html, encoding='utf-8')
    return html
