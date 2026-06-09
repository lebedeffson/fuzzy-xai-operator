from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / 'reports' / 'chapter2_3'
MANIFEST_PATH = ROOT / 'evidence' / 'chapter2_3_manifest_sha256.json'

ROWS: list[dict[str, Any]] = [
    {
        'chapter': 2,
        'object_id': 'ExplainPlan',
        'ru_name': 'Контракт ExplainPlan: термы, правила, веса, пороги и trace',
        'code_paths': ['fuzzyxai/core/explain_plan.py', 'configs/explain_plan_chapter2.yaml'],
        'test_paths': ['tests/test_explain_plan_contract.py'],
        'artifact_paths': ['reports/chapter2/explain_plan_hash.json'],
        'text_insert': 'ExplainPlan фиксируется до применения оператора и проверяется детерминированным SHA256.',
    },
    {
        'chapter': 2,
        'object_id': 'E_k',
        'ru_name': 'Объект объяснения ExplanationObject E_k',
        'code_paths': ['fuzzyxai/core/explanation_object.py', 'fuzzyxai/core/system_operator.py'],
        'test_paths': ['tests/test_operator_trace.py'],
        'artifact_paths': ['reports/chapter2/sample_113_report.json', 'reports/chapter2/sample_113_trace.json'],
        'text_insert': 'E_k содержит термы, правила, активации, неопределённость и трассируемый источник tau_k.',
    },
    {
        'chapter': 2,
        'object_id': 'Omega',
        'ru_name': 'Системный оператор для скалярного риска',
        'code_paths': ['fuzzyxai/core/system_operator.py', 'fuzzyxai/experiments/chapter2_sample113.py'],
        'test_paths': ['tests/test_operator_trace.py', 'tests/test_chapter2_breast_cancer_demo_smoke.py'],
        'artifact_paths': ['reports/chapter2/sample_113_table.csv'],
        'text_insert': 'Оператор переводит risk_score в лингвистические термы, правила и численные каналы отчёта.',
    },
    {
        'chapter': 2,
        'object_id': 'd_E_gamma_Dij',
        'ru_name': 'Семантическое рассогласование d_E, gamma и D_ij',
        'code_paths': ['fuzzyxai/core/trust_evaluator.py', 'fuzzyxai/core/composition.py', 'fuzzyxai/core/diagnostics.py'],
        'test_paths': ['tests/test_semantic_disagreement_pseudometric.py', 'tests/test_composition.py', 'tests/test_composition_well_defined.py'],
        'artifact_paths': ['reports/chapter2/equal_raw_structure_report.json'],
        'text_insert': 'При недопустимом gamma композиция возвращает DiagnosticState, а не скрывает разрыв.',
    },
    {
        'chapter': 2,
        'object_id': 'T_ij_synthesis',
        'ru_name': 'Ограниченный синтез перехода T_ij',
        'code_paths': ['fuzzyxai/core/alignment_synthesis.py', 'experiments/chapter2_alignment_synthesis.py'],
        'test_paths': ['tests/test_alignment_synthesis.py'],
        'artifact_paths': ['reports/chapter2/alignment_synthesis_report.json'],
        'text_insert': 'T_ij синтезируется только по конечному набору кандидатов ExplainPlan; иначе возвращается D_ij.',
    },
    {
        'chapter': 2,
        'object_id': 'I_EG_L_E',
        'ru_name': 'Потеря интерпретируемости L(E) и индекс I(E_G)',
        'code_paths': ['fuzzyxai/core/trust_evaluator.py'],
        'test_paths': ['tests/test_trust.py', 'tests/test_chain_loss_bound.py'],
        'artifact_paths': ['reports/chapter2/chapter2_breast_cancer_summary.json'],
        'text_insert': 'Индекс интерпретируемости вычисляется из компонентной потери и используется в risk-aware контуре.',
    },
    {
        'chapter': 2,
        'object_id': 'cH_cO_cK_calibration',
        'ru_name': 'Калибровка констант c_H, c_O, c_K',
        'code_paths': ['fuzzyxai/experiments/chapter2_calibration.py'],
        'test_paths': ['tests/test_chapter2_chapter3_artifacts.py'],
        'artifact_paths': ['reports/chapter2/calibration_report.json', 'reports/chapter2/calibration_constants.csv', 'figures/chapter2/calibration_constants.png'],
        'text_insert': 'Константы калибруются на C_cal и не подставляются вручную в текст главы.',
    },
    {
        'chapter': 2,
        'object_id': 'equal_raw_structure',
        'ru_name': 'Benchmark native/equal_guardrail/equal_raw_structure',
        'code_paths': ['fuzzyxai/experiments/chapter2_equal_raw_structure.py'],
        'test_paths': ['tests/test_chapter2_chapter3_artifacts.py'],
        'artifact_paths': ['reports/chapter2/equal_raw_structure_summary.csv', 'figures/chapter2/equal_raw_structure_comparison.png'],
        'text_insert': 'Сравнение показывает отличие доступа к сырым структурам от сертифицированного маршрута.',
    },
    {
        'chapter': 3,
        'object_id': 'F_hierarchy',
        'ru_name': 'Иерархия F0, IntervalFS, HesitantFS, NeutrosophicFS, MultiLevelFS',
        'code_paths': ['fuzzyxai/hierarchy/f0.py', 'fuzzyxai/hierarchy/interval.py', 'fuzzyxai/hierarchy/hesitant.py', 'fuzzyxai/hierarchy/neutrosophic.py', 'fuzzyxai/hierarchy/multilevel.py'],
        'test_paths': ['tests/test_reductions.py', 'tests/test_multilevel_reduction_termination.py'],
        'artifact_paths': ['dissertation_artifacts/diagram_specs/chapter3/fig_3_2_hierarchy.json'],
        'text_insert': 'Все классы представлений реализованы как отдельные типы, а не только описаны формально.',
    },
    {
        'chapter': 3,
        'object_id': 'Reduction_Delta',
        'ru_name': 'Редукция к F0 и потеря редукции Delta',
        'code_paths': ['fuzzyxai/hierarchy/reductions.py', 'fuzzyxai/hierarchy/meta_reducer.py'],
        'test_paths': ['tests/test_reduction_approximation_theorem.py', 'tests/test_reduction_graph.py'],
        'artifact_paths': ['dissertation_artifacts/diagram_specs/chapter3/fig_3_3_reduction.json'],
        'text_insert': 'Редукция и Delta считаются программно и участвуют в выборе действия наблюдателя.',
    },
    {
        'chapter': 3,
        'object_id': 'P_sit_D_choice',
        'ru_name': 'Профиль ситуации P_sit, Pareto-выбор и D_choice',
        'code_paths': ['fuzzyxai/selection/profile_builder.py', 'fuzzyxai/selection/pareto_selector.py', 'fuzzyxai/selection/choice_diagnostic.py'],
        'test_paths': ['tests/test_selector.py', 'tests/test_operational_coverage_minimality.py'],
        'artifact_paths': ['reports/chapter3/dataset_roles_summary.csv'],
        'text_insert': 'Класс представления выбирается по покрытию профиля и минимальной достаточности.',
    },
    {
        'chapter': 3,
        'object_id': 'chi_Auto_topos',
        'ru_name': 'Контекстная допустимость chi_Auto и topoi/subpresheaf слой',
        'code_paths': ['fuzzyxai/category/context_topos.py', 'fuzzyxai/category/subpresheaf.py', 'fuzzyxai/risk/context_acceptance.py'],
        'test_paths': ['tests/test_characteristic_morphism_auto.py', 'tests/test_chi_auto_truth_values.py', 'tests/test_subobject_classifier.py', 'tests/test_subpresheaf.py'],
        'artifact_paths': ['dissertation_artifacts/diagram_specs/chapter3/fig_3_8_chi_auto_sample113.json'],
        'text_insert': 'chi_Auto вычисляется как контекстная проверка допустимости автоматического применения.',
    },
    {
        'chapter': 3,
        'object_id': 'CertifiedPath_Rupture',
        'ru_name': 'CertifiedPath и Rupture объяснительного маршрута',
        'code_paths': ['fuzzyxai/category/certified_path.py', 'fuzzyxai/hott/rupture_type.py', 'fuzzyxai/hott/path_certificates.py'],
        'test_paths': ['tests/test_certified_path_category.py', 'tests/test_hott_certified_paths.py', 'tests/test_hott_path_rupture.py'],
        'artifact_paths': ['dissertation_artifacts/diagram_specs/chapter3/fig_3_4_route.json'],
        'text_insert': 'Маршрут либо сертифицируется как путь, либо фиксирует диагностический разрыв.',
    },
    {
        'chapter': 3,
        'object_id': 'rho_action',
        'ru_name': 'rho(x), policy и действия наблюдателя',
        'code_paths': ['fuzzyxai/risk/risk_function.py', 'fuzzyxai/risk/decision_policy.py', 'fuzzyxai/risk/observer_pipeline.py'],
        'test_paths': ['tests/test_risk_function.py', 'tests/test_decision_policy.py', 'tests/test_critical_rupture_blocks.py'],
        'artifact_paths': ['reports/chapter3/synthetic_ruptures_summary.json'],
        'text_insert': 'rho(x) и action вычисляются из структурных признаков, uncertainty, Delta и diagnostic states.',
    },
    {
        'chapter': 3,
        'object_id': 'controlled_critical_ruptures',
        'ru_name': 'Controlled critical ruptures: пять типов нарушений и пять политик',
        'code_paths': ['fuzzyxai/experiments/chapter3_critical_ruptures.py'],
        'test_paths': ['tests/test_chapter3_critical_ruptures.py'],
        'artifact_paths': ['reports/chapter3/synthetic_ruptures_results.csv', 'figures/chapter3/critical_ruptures_comparison.png', 'evidence/critical_ruptures_manifest_sha256.json'],
        'text_insert': 'Диагностический стенд проверяет safety-свойство без ручной подстановки чисел.',
    },
]


def sha256_file(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ''
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _exists_all(paths: list[str]) -> bool:
    return all((ROOT / p).exists() for p in paths)


def _status(row: dict[str, Any]) -> str:
    return 'implemented' if _exists_all(row['code_paths']) and _exists_all(row['test_paths']) and _exists_all(row['artifact_paths']) else 'missing_evidence'


def _missing(row: dict[str, Any]) -> list[str]:
    paths = row['code_paths'] + row['test_paths'] + row['artifact_paths']
    return [p for p in paths if not (ROOT / p).exists()]


def _join(paths: list[str]) -> str:
    return '; '.join(paths)


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def run(out_dir: str | Path = OUT_DIR) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for row in ROWS:
        item = dict(row)
        item['status'] = _status(row)
        item['missing_paths'] = _join(_missing(row))
        item['code_paths'] = _join(row['code_paths'])
        item['test_paths'] = _join(row['test_paths'])
        item['artifact_paths'] = _join(row['artifact_paths'])
        rows.append(item)

    missing = [r for r in rows if r['status'] != 'implemented']
    fieldnames = ['chapter', 'object_id', 'ru_name', 'status', 'code_paths', 'test_paths', 'artifact_paths', 'missing_paths', 'text_insert']
    csv_path = out / 'chapter2_3_implementation_matrix.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({k: r.get(k, '') for k in fieldnames} for r in rows)

    md_path = out / 'chapter2_3_implementation_matrix.md'
    lines = [
        '# Матрица реализации глав 2-3',
        '',
        'Эта таблица фиксирует, что каждый ключевой объект глав 2-3 имеет код, тест и воспроизводимый артефакт.',
        '',
        '| глава | объект | статус | реализация | тесты | артефакты |',
        '| ---: | --- | --- | --- | --- | --- |',
    ]
    for r in rows:
        lines.append(f"| {r['chapter']} | `{r['object_id']}`: {r['ru_name']} | `{r['status']}` | `{r['code_paths']}` | `{r['test_paths']}` | `{r['artifact_paths']}` |")
    md_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')

    insert_path = out / 'chapter2_3_writer_insert.md'
    inserts = ['# Вставка для писателя по главам 2-3', '']
    for chapter in [2, 3]:
        inserts.append(f'## Глава {chapter}')
        for r in rows:
            if r['chapter'] == chapter:
                inserts.append(f"- `{r['object_id']}`: {r['text_insert']}")
        inserts.append('')
    insert_path.write_text('\n'.join(inserts), encoding='utf-8')

    files_for_manifest = [
        'reports/chapter2_3/chapter2_3_implementation_matrix.csv',
        'reports/chapter2_3/chapter2_3_implementation_matrix.md',
        'reports/chapter2_3/chapter2_3_writer_insert.md',
        *[p for row in ROWS for p in row['artifact_paths']],
    ]
    manifest_files = []
    for rel in sorted(set(files_for_manifest)):
        p = ROOT / rel
        manifest_files.append({'path': rel, 'exists': p.exists(), 'sha256': sha256_file(p)})

    payload = {
        'status': 'ok' if not missing else 'failed',
        'checked_at': datetime.now(timezone.utc).isoformat(),
        'chapters': [2, 3],
        'implemented_rows': sum(1 for r in rows if r['status'] == 'implemented'),
        'total_rows': len(rows),
        'missing': [{'object_id': r['object_id'], 'missing_paths': r['missing_paths']} for r in missing],
        'matrix_csv': _display_path(csv_path),
        'matrix_md': _display_path(md_path),
        'writer_insert': _display_path(insert_path),
        'files': manifest_files,
    }
    json_path = out / 'chapter2_3_final_evidence.json'
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    payload['files'].append({'path': _display_path(json_path), 'exists': True, 'sha256': sha256_file(json_path)})
    MANIFEST_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

    if missing:
        raise SystemExit(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


if __name__ == '__main__':
    print(json.dumps(run(), ensure_ascii=False, indent=2))
