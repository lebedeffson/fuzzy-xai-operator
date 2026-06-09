from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = ROOT / 'data' / 'article_fixtures' / 'hybrid_xiris_blocking_fixture.json'
JSON_PATH = ROOT / 'reports' / 'chapter5' / 'hybrid_xiris_blocking_case.json'
CSV_PATH = ROOT / 'reports' / 'chapter5' / 'hybrid_xiris_blocking_case.csv'
FIG_PATH = ROOT / 'figures' / 'chapter5' / 'hybrid_xiris_blocking_case.png'
REPORT_PATH = ROOT / 'reports' / 'chapter5' / 'scenario_reports' / 'hybrid_xiris_blocking_case_report.md'
PLAN_PATH = ROOT / 'configs' / 'explain_plan_chapter2.yaml'


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _write_fixture() -> None:
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'sample_id': 'hybrid_xiris_block_001',
        'image_quality_score': 0.31,
        'segmentation_quality_score': 0.27,
        'model_match_score': 0.88,
        'rule_model_accept_activation': 0.82,
        'rule_quality_block_activation': 0.91,
        'source_conflict': True,
        'notes': 'Fixed numeric fixture for chapter 5.9.2 blocking demonstration.',
    }
    FIXTURE_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')


def build_case() -> dict[str, Any]:
    _write_fixture()
    fixture = json.loads(FIXTURE_PATH.read_text(encoding='utf-8'))
    fixture_sha = sha256_file(FIXTURE_PATH)
    explain_plan_hash = sha256_file(PLAN_PATH)
    chi_R_crit = 1 if fixture['source_conflict'] or fixture['segmentation_quality_score'] < 0.35 else 0
    chi_Auto = False if chi_R_crit else fixture['model_match_score'] >= 0.75
    rho = 0.8 if chi_R_crit else round(0.45 * (1 - fixture['segmentation_quality_score']) + 0.55 * fixture['model_match_score'], 4)
    action = 'block' if chi_R_crit else 'audit_report'
    result = {
        **fixture,
        'chi_R_crit': chi_R_crit,
        'chi_Auto': chi_Auto,
        'rho': rho,
        'action': action,
        'report_path': str(REPORT_PATH.relative_to(ROOT)),
        'explain_plan_hash': explain_plan_hash,
        'fixture_sha256': fixture_sha,
    }
    return result


def _write_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        'sample_id', 'image_quality_score', 'segmentation_quality_score', 'model_match_score',
        'rule_model_accept_activation', 'rule_quality_block_activation', 'source_conflict',
        'chi_R_crit', 'chi_Auto', 'rho', 'action', 'report_path', 'explain_plan_hash', 'fixture_sha256'
    ]
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow({k: row.get(k, '') for k in fields})


def _write_figure(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = ['image quality', 'segmentation', 'match', 'accept rule', 'block rule', 'rho']
    values = [row['image_quality_score'], row['segmentation_quality_score'], row['model_match_score'], row['rule_model_accept_activation'], row['rule_quality_block_activation'], row['rho']]
    colors = ['#ef4444', '#ef4444', '#16a34a', '#2563eb', '#dc2626', '#7c2d12']
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(labels, values, color=colors, alpha=0.88)
    ax.set_ylim(0, 1)
    ax.set_title('HYBRID-XIRIS: блокировка при конфликте качества/источника', fontsize=14, fontweight='bold')
    ax.set_ylabel('значение')
    ax.grid(axis='y', alpha=0.25)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, min(0.97, v + 0.03), f'{v:.2f}', ha='center', fontsize=10)
    ax.text(0.02, 0.95, f"action={row['action']} | chi_R_crit={row['chi_R_crit']} | chi_Auto={row['chi_Auto']}", transform=ax.transAxes, fontsize=11, bbox={'facecolor': '#fee2e2', 'edgecolor': '#dc2626', 'boxstyle': 'round,pad=0.35'})
    fig.autofmt_xdate(rotation=18, ha='right')
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run() -> dict[str, Any]:
    row = build_case()
    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding='utf-8')
    _write_csv(CSV_PATH, row)
    _write_figure(FIG_PATH, row)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text('\n'.join([
        '# HYBRID-XIRIS blocking case',
        '',
        f"- sample_id: `{row['sample_id']}`",
        f"- chi_R_crit: `{row['chi_R_crit']}`",
        f"- chi_Auto: `{row['chi_Auto']}`",
        f"- rho: `{row['rho']}`",
        f"- action: `{row['action']}`",
        f"- fixture_sha256: `{row['fixture_sha256']}`",
        '',
        'Автоматическое применение заблокировано: качество сегментации низкое и есть конфликт источников.',
    ]), encoding='utf-8')
    return {'status': 'ok', 'json': str(JSON_PATH), 'csv': str(CSV_PATH), 'figure': str(FIG_PATH), 'row': row}


if __name__ == '__main__':
    print(json.dumps(run(), ensure_ascii=False, indent=2))
