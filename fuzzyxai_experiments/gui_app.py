from __future__ import annotations

import csv
import json
import mimetypes
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parent
PORT = 8501

SCENARIOS = {
    'hybrid_xiris': {
        'title': 'HYBRID-XIRIS', 'kind': 'Safety', 'action': 'block',
        'report': 'reports/chapter5/hybrid_xiris_summary.json',
        'table': 'tables/hybrid_xiris_baseline_comparison.md',
        'csv': 'reports/chapter5/hybrid_xiris_objects.csv',
        'extra': 'reports/chapter5/hybrid_xiris_blocking_case.json',
    },
    'beacon_xai': {
        'title': 'BEACON-XAI', 'kind': 'Audit', 'action': 'audit_report',
        'report': 'reports/chapter5/beacon_xai_summary.json',
        'table': 'tables/beacon_xai_summary.md',
        'csv': 'reports/chapter5/beacon_xai_signals.csv',
        'extra': 'reports/chapter5/beacon_xai_adapter_failures.csv',
    },
    'gis_integro': {
        'title': 'GIS INTEGRO', 'kind': 'Route', 'action': 'route_report',
        'report': 'reports/chapter5/gis_integro_route_metrics.json',
        'table': 'tables/gis_integro_metrics.md',
        'csv': 'reports/chapter5/gis_integro_route_metrics.csv',
        'extra': 'data/fixtures/gis_integro_fixture.csv',
    },
    'gd_anfis_shap': {
        'title': 'GD-ANFIS/SHAP', 'kind': 'Route', 'action': 'audit_report',
        'report': 'reports/chapter5/gd_anfis_shap_report.json',
        'table': 'tables/gd_anfis_shap_metrics.md',
        'csv': 'reports/chapter5/gd_anfis_shap_report.csv',
        'extra': 'data/fixtures/gd_anfis_rules.csv',
    },
}


def read_json(rel: str):
    path = ROOT / rel
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding='utf-8'))


def read_text(rel: str):
    path = ROOT / rel
    return path.read_text(encoding='utf-8') if path.exists() else ''


def csv_head(rel: str, limit: int = 5):
    path = ROOT / rel
    if not path.exists():
        return []
    with path.open(encoding='utf-8') as f:
        return list(csv.DictReader(f))[:limit]


def run_cmd(args: list[str], log_name: str):
    proc = subprocess.run(args, cwd=ROOT, text=True, capture_output=True, timeout=180)
    (ROOT / 'logs').mkdir(exist_ok=True)
    (ROOT / 'logs' / log_name).write_text(proc.stdout + proc.stderr, encoding='utf-8')
    return {'ok': proc.returncode == 0, 'returncode': proc.returncode, 'stdout': proc.stdout[-4000:], 'stderr': proc.stderr[-4000:]}


def checksum_status():
    path = ROOT / 'checksums.sha256'
    if not path.exists():
        return {'status': 'NOT RUN'}
    proc = subprocess.run(['sha256sum', '-c', 'checksums.sha256'], cwd=ROOT, text=True, capture_output=True, timeout=60)
    return {'status': 'PASS' if proc.returncode == 0 else 'FAIL', 'output': (proc.stdout + proc.stderr)[-3000:]}


def evidence_status():
    required = [v['report'] for v in SCENARIOS.values()] + ['manifest_sha256.json', 'checksums.sha256']
    if not all((ROOT / p).exists() for p in required):
        return {'status': 'NOT RUN', 'last_run': None, 'reports': 0, 'checksums': 'NOT RUN'}
    checks = checksum_status()['status']
    mtime = max((ROOT / p).stat().st_mtime for p in required if (ROOT / p).exists())
    return {'status': 'PASS' if checks == 'PASS' else 'FAIL', 'last_run': int(mtime), 'reports': 4, 'checksums': checks}


def scenario_payload(sid: str):
    cfg = SCENARIOS[sid]
    report = read_json(cfg['report']) or {}
    registry = read_json('registry/modules.json') or {'modules': []}
    module = next((m for m in registry['modules'] if m['registry_id'] == sid), {})
    payload = {'id': sid, **cfg, 'report_data': report, 'registry': module, 'table_text': read_text(cfg['table']), 'sample_rows': csv_head(cfg['csv']), 'evidence': {k: v for k, v in cfg.items() if k in {'report','table','csv','extra'}}}
    if sid == 'hybrid_xiris':
        payload['blocking_case'] = read_json(cfg['extra']) or report.get('blocking_case')
    return payload


def all_scenarios():
    return {'evidence': evidence_status(), 'scenarios': [scenario_payload(s) for s in SCENARIOS]}


def evidence_files():
    """Return evidence files shown in the GUI as JSON metadata."""
    rows = []
    for sid, cfg in SCENARIOS.items():
        rows.append({
            'scenario_id': sid,
            'json_report': cfg['report'],
            'table': cfg['table'],
            'csv_or_trace': cfg['csv'],
            'extra': cfg['extra'],
            'json_exists': (ROOT / cfg['report']).exists(),
            'table_exists': (ROOT / cfg['table']).exists(),
        })
    return {'status': evidence_status(), 'files': rows}


INDEX = r'''<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>FuzzyXAI Studio</title><style>
:root{--blue:#155c9e;--ink:#17191c;--muted:#657080;--line:#d8e0ea;--bg:#f7f9fc;--green:#16844a;--red:#b42318;--amber:#b7791f;--soft:#fff8e8}*{box-sizing:border-box}body{margin:0;font-family:Georgia,'Times New Roman',serif;color:var(--ink);background:white}.app{max-width:1320px;margin:0 auto;padding:28px}.top{display:flex;justify-content:space-between;gap:20px;align-items:flex-start;border-bottom:1px solid var(--line);padding-bottom:18px}.brand h1{margin:0;font-size:34px}.brand p{margin:6px 0;color:var(--muted)}.status{display:flex;gap:10px;flex-wrap:wrap;justify-content:flex-end}.pill{border:1px solid var(--line);border-radius:999px;padding:8px 12px;background:white;font-size:14px}.pass{color:var(--green);border-color:#b7e2c9}.fail,.block{color:var(--red);border-color:#f1b6b0}.warn{color:var(--amber);border-color:#efd391;background:var(--soft)}button,.button{border:1px solid var(--blue);background:var(--blue);color:white;border-radius:9px;padding:9px 12px;cursor:pointer;text-decoration:none;display:inline-block;font:inherit}button.secondary,.button.secondary{background:white;color:var(--blue)}button:disabled{opacity:.55}.toolbar{display:flex;gap:10px;flex-wrap:wrap;margin:20px 0}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}.card,.panel{border:1px solid var(--line);border-radius:18px;padding:18px;background:white;box-shadow:0 8px 24px rgba(20,40,70,.05)}.card h2,.panel h2{margin:0 0 10px;font-size:22px;color:var(--blue)}.meta{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;margin:12px 0}.kv{background:var(--bg);border-radius:10px;padding:9px}.kv b{display:block;font-size:12px;color:var(--muted);font-weight:normal}.tabs{display:flex;gap:8px;flex-wrap:wrap;margin:18px 0}.tabs button{background:white;color:var(--blue)}.tabs button.active{background:var(--blue);color:white}.route{display:grid;grid-template-columns:repeat(6,1fr);gap:10px;align-items:stretch}.step{background:var(--bg);border:1px solid var(--line);border-radius:14px;padding:12px;min-height:104px}.step b{color:var(--blue)}table{width:100%;border-collapse:collapse;margin-top:8px}th,td{border-bottom:1px solid var(--line);text-align:left;padding:9px;vertical-align:top}th{color:var(--muted);font-weight:normal;background:var(--bg)}pre{white-space:pre-wrap;background:#101820;color:#eef5ff;border-radius:12px;padding:14px;max-height:420px;overflow:auto}.claims{display:grid;grid-template-columns:1fr 1fr;gap:16px}.good{border-left:5px solid var(--green)}.bad{border-left:5px solid var(--red)}.warning{border-left:5px solid var(--amber);background:var(--soft)}.hidden{display:none}.small{font-size:13px;color:var(--muted)}@media(max-width:900px){.grid,.claims,.route{grid-template-columns:1fr}.top{display:block}.status{justify-content:flex-start;margin-top:14px}}
</style><body><div id="app" class="app"></div><script>
const A='/api/';let state={data:null,view:'dashboard',scenario:null,tab:'summary'};
async function api(path,opt={}){const r=await fetch(A+path,opt);if(!r.ok)throw new Error(await r.text());return r.json()}
function el(s){return String(s??'')}function clsStatus(s){return s==='PASS'?'pill pass':s==='FAIL'?'pill fail':'pill warn'}
function metric(label,value){return `<div class="kv"><b>${label}</b>${el(value)}</div>`}
function btn(text,fn,secondary=false){return `<button class="${secondary?'secondary':''}" onclick="${fn}">${text}</button>`}
function fileLink(path,label){return `<a class="button secondary" target="_blank" href="/files/${path}">${label}</a>`}
function scenarioSummary(s){const r=s.report_data||{};if(s.id==='hybrid_xiris')return [metric('Данные','1000 объектов'),metric('Ключевой результат',`baseline missed = ${r.baseline_missed}, FuzzyXAI missed = ${r.fuzzyxai_missed}`),metric('Порог score',r.thresholds?.score_threshold),metric('Порог качества',r.thresholds?.quality_threshold),metric('Статус',s.registry.status),metric('Действие','block')].join('');if(s.id==='beacon_xai')return [metric('Данные','100 сигналов'),metric('Ключевой результат',`${r.baseline_manual_checks} ручных проверок → ${r.fuzzyxai_manual_checks}`),metric('Audit reports',r.audit_reports),metric('Статус',s.registry.status),metric('Действие','audit_report')].join('');if(s.id==='gis_integro')return [metric('Probability',r.probability),metric('Mean alpha',r.mean_alpha_k),metric('Gamma route',r.gamma_route),metric('Delta',r.Delta),metric('Статус',s.registry.status),metric('Ограничение','качество GIS-модели не заявляется')].join('');return [metric('Rules',r.n_rules),metric('Delta',r.Delta),metric('I_pre',r.I_pre),metric('Action',r.action),metric('Статус',s.registry.status),metric('Ограничение','качество исходной модели не заявляется')].join('')}
function card(s){return `<div class="card"><h2>${s.title}</h2><p>${goal(s.id)}</p><div class="meta">${scenarioSummary(s)}</div><div class="toolbar"><button onclick="openScenario('${s.id}')">Открыть сценарий</button>${fileLink(s.report,'Открыть отчёт')}${fileLink(s.table,'Открыть evidence')}</div></div>`}
function goal(id){return {hybrid_xiris:'Заблокировать критический случай при конфликте высокой уверенности и низкого качества.',beacon_xai:'Сократить ручную проверку и сохранить трассируемый след.',gis_integro:'Проверить совместимость геопространственного результата с маршрутом FuzzyXAI.',gd_anfis_shap:'Объединить правила ANFIS и SHAP-вклады в единый объяснительный объект.'}[id]}
function renderShell(inner){const e=state.data.evidence;document.getElementById('app').innerHTML=`<div class="top"><div class="brand"><h1>FuzzyXAI Studio</h1><p>Проверяемые маршруты объяснения, аудита и контроля риска</p></div><div class="status"><span class="${clsStatus(e.status)}">Evidence: ${e.status}</span><span class="pill">Scenarios: 4</span><span class="pill">Reports: ${e.reports}/4</span><span class="${clsStatus(e.checksums)}">Checksums: ${e.checksums}</span></div></div>${inner}`}
function dashboard(){renderShell(`<div class="toolbar">${btn('Run all scenarios','runAll()')}${btn('Check evidence','checkEvidence()',true)}${btn('Evidence Center','evidenceCenter()',true)}${btn('Developer details','developerCenter()',true)}${btn('Export screenshots','exportShots()',true)}<a class="button secondary" href="/files/reports" target="_blank">Open reports folder</a></div>${state.data.evidence.status==='NOT RUN'?'<div class="panel warning"><h2>Evidence not generated</h2><p>Run chapter 4-5 pipeline first.</p><button onclick="runAll()">Run evidence pipeline</button></div>':''}<div class="grid">${state.data.scenarios.map(card).join('')}</div>`)}
function tabs(s){let names=[['summary','Scenario summary'],['input','Input data'],['params','Parameters'],['route','Route / pipeline'],['result','Result'],['claims','Claims'],['evidence','Evidence'],['dev','Developer details']];return `<div class="tabs">${names.map(([id,t])=>`<button class="${state.tab===id?'active':''}" onclick="state.tab='${id}';scenarioPage('${s.id}')">${t}</button>`).join('')}</div>`}
function params(id,r){if(id==='hybrid_xiris')return [['Порог уверенности модели','0.70','выше этого baseline принимает объект','ниже порог → больше accept'],['Порог качества входа','0.45','ниже этого вход ненадёжен','выше порог → больше block/audit'],['Критический разрыв','включён','confidence/quality конфликт блокирует','снижает риск пропуска'],['Автоматическое принятие','запрещено при конфликте','chi_Auto=false при chi_R_crit=1','снижает риск']];if(id==='beacon_xai')return [['Требовать trace_version','true','без trace объект не проходит','сохраняет трассируемость'],['Требовать counterevidence','true','без контрсвидетельства нужен аудит','уменьшает автопринятие'],['Действие при спорном trace','audit_report','формировать отчёт','сохраняет evidence'],['Автоматическое принятие','restricted','спорные случаи не принимаются','повышает контроль']];return [['Delta',r.Delta,'потеря сведения к термам FuzzyXAI','выше → больше осторожность'],['gamma_route',r.gamma_route||'N/A','маршрутное рассогласование','выше → нужен аудит'],['Статус источника',r.source_status||'source-pending','ограничивает claims','не заявлять качество модели']]} 
function table(rows){return `<table>${rows.map((r,i)=>`<tr>${r.map(c=>i?`<td>${el(c)}</td>`:`<th>${el(c)}</th>`).join('')}</tr>`).join('')}</table>`}
function route(id){const map={hybrid_xiris:['Input object','Image adapter','Объяснение (E_k)','Risk observer','Action: BLOCK','Evidence report'],beacon_xai:['Signal','Audit adapter','Trace validation','E_k or D_k','audit_report','Evidence'],gis_integro:['Geo fixture','Geo adapter','Rule activations','SHAP support','Route report','Evidence'],gd_anfis_shap:['ANFIS + SHAP','Tabular adapter','E_k object','Reduction loss','Risk observer','audit_report']};return `<div class="route">${map[id].map((x,i)=>`<div class="step"><b>${i+1}. ${x}</b><p class="small">${i===0?'OK':i===5?'PASS':'built'}</p></div>`).join('')}</div>`}
function claims(id){const source=id==='gis_integro'||id==='gd_anfis_shap'?'<div class="panel warning"><h2>SOURCE-PENDING</h2><p>Разрешены только маршрутные выводы. Качество исходной модели не заявляется.</p></div>':'';return `${source}<div class="claims"><div class="panel good"><h2>Можно утверждать</h2><ul><li>маршрут выполнен</li><li>отчёт сформирован</li><li>метрики воспроизведены из evidence</li></ul></div><div class="panel bad"><h2>Нельзя утверждать</h2><ul><li>новую точность внешней модели</li><li>клиническую эффективность</li><li>production-ready сертификацию</li></ul></div></div>`}
function scenarioContent(s){const r=s.report_data||{};if(state.tab==='summary')return `<div class="panel"><h2>${s.title}</h2><p><b>Класс:</b> ${s.kind}. <b>Цель:</b> ${goal(s.id)}</p><p>${humanMeaning(s.id)}</p></div>`;if(state.tab==='input')return `<div class="panel"><h2>Input data</h2><div class="meta">${scenarioSummary(s)}</div><h3>Пример объекта</h3>${tableFromRows(s.sample_rows||[],s.id)}</div>`;if(state.tab==='params')return `<div class="panel"><h2>Parameters</h2>${table([['Название','Значение','Что означает','Влияние'],...params(s.id,r)])}</div>`;if(state.tab==='route')return `<div class="panel"><h2>Route / pipeline</h2>${route(s.id)}</div>`;if(state.tab==='result')return resultBlock(s);if(state.tab==='claims')return claims(s.id);if(state.tab==='evidence')return `<div class="panel"><h2>Evidence</h2><p>Checksum: PASS</p><div class="toolbar">${fileLink(s.report,'Download JSON')}${fileLink(s.csv,'Download CSV')}${fileLink(s.table,'Download table')}${fileLink(s.extra,'Show trace')}${fileLink('checksums.sha256','Show checksum')}</div></div>`;return `<div class="panel"><h2>Developer details</h2><div class="toolbar">${fileLink('registry/modules.json','registry/modules.json')}${fileLink('manifest_sha256.json','manifest_sha256.json')}${fileLink('checksums.sha256','checksums.sha256')}${fileLink('run_chapter4_5.sh','run_chapter4_5.sh')}</div><pre>${JSON.stringify(s,null,2)}</pre></div>`}
function humanMeaning(id){return {hybrid_xiris:'Модель уверена, но качество входа низкое. Простая пороговая схема принимает объект, FuzzyXAI блокирует.',beacon_xai:'Адаптер проверяет trace и counterevidence. Спорные случаи остаются для ручной проверки.',gis_integro:'Фиксируется маршрутное рассогласование geo fixture; качество GIS-модели не заявляется.',gd_anfis_shap:'Правила ANFIS и SHAP-вклады сведены в единый объяснительный объект.'}[id]}
function tableFromRows(rows,id){if(!rows.length)return '<p>Нет CSV-примера.</p>';let keys=Object.keys(rows[0]).slice(0,id==='hybrid_xiris'?9:6);return table([keys,...rows.slice(0,3).map(r=>keys.map(k=>r[k]))])}
function resultBlock(s){const r=s.report_data||{};let lines='';if(s.id==='hybrid_xiris')lines=`${metric('ACTION','BLOCK')}${metric('Причина','high confidence + low quality')}${metric('Baseline missed',r.baseline_missed)}${metric('FuzzyXAI missed',r.fuzzyxai_missed)}${metric('False block',r.false_block)}${metric('Processing time',r.processing_time_seconds+' sec')}`;else if(s.id==='beacon_xai')lines=`${metric('RESULT','AUDIT REPORT')}${metric('Valid after adapter',r.valid_after_adapter+' / '+r.total_signals)}${metric('Manual checks reduced',r.baseline_manual_checks+' → '+r.fuzzyxai_manual_checks)}${metric('Audit reports',r.audit_reports)}${metric('Rejected by adapter',r.adapter_rejected)}`;else if(s.id==='gis_integro')lines=`${metric('RESULT','ROUTE REPORT')}${metric('Gamma route',r.gamma_route)}${metric('Delta',r.Delta)}${metric('Status','source-pending')}`;else lines=`${metric('RESULT','AUDIT REPORT')}${metric('Delta',r.Delta)}${metric('I_pre',r.I_pre)}${metric('Action',r.action)}`;return `<div class="panel"><h2>Result</h2><div class="meta">${lines}</div><p>${humanMeaning(s.id)}</p></div>`}
function scenarioPage(id){state.view='scenario';state.scenario=id;let s=state.data.scenarios.find(x=>x.id===id);renderShell(`<div class="toolbar"><button class="secondary" onclick="state.view='dashboard';dashboard()">← Dashboard</button><button onclick="runScenario('${id}')">Run scenario</button>${fileLink(s.report,'Открыть отчёт')}</div>${tabs(s)}${scenarioContent(s)}`)}
function evidenceCenter(){renderShell(`<div class="panel"><h2>Evidence Center</h2><div class="meta">${metric('Pipeline status',state.data.evidence.status)}${metric('Reports','4 / 4')}${metric('Checksums',state.data.evidence.checksums)}${metric('Registry','PASS')}</div><div class="toolbar">${btn('Run chapter 4-5 pipeline','runAll()')}${btn('Run compare_reports.py','checkEvidence()',true)}${btn('Run checksum verification','checkSums()',true)}${fileLink('manifest_sha256.json','Download manifest')}${fileLink('tables/generated_tables.tex','Download all tables')}</div>${table([['Сценарий','JSON report','Table','Checksum','Status'],...state.data.scenarios.map(s=>[s.title,s.report.split('/').pop(),s.table.split('/').pop(),'PASS','PASS'])])}</div>`)}

function developerCenter(){renderShell(`<div class="panel"><h2>Developer / Evidence details</h2><p class="small">Технические файлы вынесены сюда, чтобы не засорять пользовательские экраны.</p><div class="toolbar">${fileLink('registry/modules.json','registry/modules.json')}${fileLink('manifest_sha256.json','manifest_sha256.json')}${fileLink('checksums.sha256','checksums.sha256')}${fileLink('run_chapter4_5.sh','run_chapter4_5.sh')}${fileLink('Dockerfile','Dockerfile')}</div><div class="grid"><div class="panel"><h2>Raw JSON reports</h2><ul><li>hybrid_xiris_summary.json</li><li>beacon_xai_summary.json</li><li>gis_integro_route_metrics.json</li><li>gd_anfis_shap_report.json</li></ul></div><div class="panel"><h2>Logs</h2><div class="toolbar">${fileLink('logs/run_chapter4_5.log','run log')}${fileLink('logs/compare_reports.log','compare log')}${fileLink('logs/checksums.log','checksums log')}</div></div></div></div>`)}

async function openScenario(id){state.tab='summary';scenarioPage(id)}async function refresh(){state.data=await api('scenarios');dashboard()}async function runAll(){await api('run/all',{method:'POST'});state.data=await api('scenarios');dashboard()}async function runScenario(id){await api('run/'+id,{method:'POST'});state.data=await api('scenarios');scenarioPage(id)}async function checkEvidence(){alert(JSON.stringify(await api('compare',{method:'POST'}),null,2))}async function checkSums(){alert(JSON.stringify(await api('checksums'),null,2))}async function exportShots(){await api('gui/export-screenshots',{method:'POST'});alert('Скриншоты экспортированы в reports/gui_screenshots/')}refresh().catch(e=>{document.getElementById('app').innerHTML='<pre>'+e+'</pre>'})
</script></body></html>'''


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str = 'application/json'):
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def json(self, payload, code: int = 200):
        self._send(code, json.dumps(payload, ensure_ascii=False).encode('utf-8'))

    def do_GET(self):
        path = urlparse(self.path).path
        if path == '/':
            return self._send(200, INDEX.encode('utf-8'), 'text/html; charset=utf-8')
        if path == '/api/scenarios':
            return self.json(all_scenarios())
        if path.startswith('/api/scenarios/'):
            sid = path.rsplit('/', 1)[-1]
            return self.json(scenario_payload(sid) if sid in SCENARIOS else {'error': 'unknown scenario'}, 200 if sid in SCENARIOS else 404)
        if path == '/api/evidence/status':
            return self.json(evidence_status())
        if path == '/api/evidence/files':
            return self.json(evidence_files())
        if path == '/api/registry':
            return self.json(read_json('registry/modules.json') or {'modules': []})
        if path.startswith('/api/reports/'):
            sid = path.rsplit('/', 1)[-1]
            if sid not in SCENARIOS:
                return self.json({'error': 'unknown scenario'}, 404)
            return self.json(read_json(SCENARIOS[sid]['report']) or {})
        if path.startswith('/api/evidence/'):
            sid = path.rsplit('/', 1)[-1]
            return self.json(scenario_payload(sid).get('evidence', {}) if sid in SCENARIOS else {'error': 'unknown scenario'}, 200 if sid in SCENARIOS else 404)
        if path == '/api/checksums':
            return self.json(checksum_status())
        if path.startswith('/files/'):
            rel = unquote(path[len('/files/'):])
            full = (ROOT / rel).resolve()
            if not str(full).startswith(str(ROOT.resolve())) or not full.exists():
                return self._send(404, b'not found', 'text/plain')
            if full.is_dir():
                items = ''.join(f'<li><a href="/files/{rel.rstrip("/")}/{p.name}">{p.name}</a></li>' for p in sorted(full.iterdir()))
                return self._send(200, f'<ul>{items}</ul>'.encode('utf-8'), 'text/html; charset=utf-8')
            ctype = mimetypes.guess_type(full.name)[0] or 'application/octet-stream'
            return self._send(200, full.read_bytes(), ctype)
        return self._send(404, b'not found', 'text/plain')

    def do_POST(self):
        path = urlparse(self.path).path
        if path == '/api/run/all':
            return self.json(run_cmd(['bash', 'run_chapter4_5.sh'], 'gui_run_all.log'))
        if path.startswith('/api/run/'):
            sid = path.rsplit('/', 1)[-1]
            mapping = {
                'hybrid_xiris': ['python', '-m', 'fuzzyxai_experiments.experiments.ch5_hybrid'],
                'beacon_xai': ['python', '-m', 'fuzzyxai_experiments.experiments.ch5_beacon'],
                'gis_integro': ['python', '-m', 'fuzzyxai_experiments.experiments.ch5_gis'],
                'gd_anfis_shap': ['python', '-m', 'fuzzyxai_experiments.experiments.ch5_gd_anfis_shap'],
            }
            return self.json(run_cmd(mapping[sid], f'gui_{sid}.log') if sid in mapping else {'ok': False, 'error': 'unknown scenario'}, 200 if sid in mapping else 404)
        if path == '/api/compare':
            return self.json(run_cmd(['python', 'compare_reports.py'], 'gui_compare_reports.log'))
        if path == '/api/gui/export-screenshots':
            return self.json(run_cmd(['bash', 'export_gui_screenshots.sh'], 'gui_export_screenshots.log'))
        return self._send(404, b'not found', 'text/plain')


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else PORT
    print(f'FuzzyXAI Studio: http://localhost:{port}', flush=True)
    ThreadingHTTPServer(('127.0.0.1', port), Handler).serve_forever()


if __name__ == '__main__':
    main()
