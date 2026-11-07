"""One-command reproducible package for dissertation chapter 5.

This runner never fakes a PTB-XL result when its data are missing. The real
WDBC model study runs offline; PTB-XL is opt-in with original WFDB files.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from experiments.chapter5_figure_pack import create_figures
from experiments.chapter5_real_heldout import SEED_DEFAULT, run as run_wdbc
from experiments.chapter5_ptbxl_qc import run as run_ptbxl

ROOT = Path(__file__).resolve().parents[1]


def _commit() -> str | None:
    try:
        p = subprocess.run(['git', 'rev-parse', '--verify', 'HEAD'], cwd=ROOT,
                           text=True, capture_output=True, timeout=4, check=True)
        s = p.stdout.strip()
        return s if len(s) == 40 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _from_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def _write_markdown(path: Path, study: dict[str, Any], rows: list[dict[str, str]], ptb: dict[str, Any] | None) -> None:
    lookup = defaultdict(list)
    for item in rows:
        lookup[(item['perturbation'], item['policy'])].append(item)
    names = [
        ('clean', 'Original holdout data'),
        ('missing_10pct', 'Controlled loss of 10% features'),
        ('missing_40pct', 'Controlled loss of 40% features'),
        ('sensor_noise_2sd', 'Controlled Gaussian noise with 2 training SD'),
    ]
    policy_names = [
        ('confidence_only', 'Probability threshold'),
        ('confidence_equal_guardrail', 'Probability + same critical quality flag'),
        ('canonical_risk_with_qc', 'Canonical FuzzyXAI risk observer + flag'),
    ]
    lines = [
        '# Chapter 5: reproduced practical results (component-level)', '',
        '## Study design and source validity', '',
        '- Dataset: public `sklearn.datasets.load_breast_cancer` (WDBC): 569 records, 30 numerical attributes.',
        f"- Split seeds: `{study['seeds']}`; 60% train, 20% validation, 20% held out (stratified).",
        '- Random Forest trained exclusively on each train partition; medians, standard deviations and range thresholds estimated from train only.',
        '- Acceptance thresholds calibrated from unlabelled model probabilities on validation only; test labels are never used in a decision.',
        '- Missingness and additive noise are controlled perturbations of real held-out observations, **not** observations of naturally occurring biomedical errors.',
        '- Method evaluated: canonical risk observer with explicit external QC inputs. The complete semantic explanation graph, proof synthesis and human comprehension are not tested here.',
        '- `one_minus_I_pre` and `Delta` are explicitly technical proxies for missing/abnormal features; they are not measured semantic interpretability or proven reduction loss.',
        '- Equal-guardrail baseline receives the same critical flag. Only the confidence-only baseline lacks this information.',
        '', '## Predictive model quality on clean held-out data', '',
        '| Split seed | n_test | Accuracy | ROC-AUC | F1 |',
        '|---:|---:|---:|---:|---:|',
    ]
    for split in study['splits']:
        lines.append(f"| {split['seed']} | {split['n_holdout']} | {split['model_accuracy_clean']:.4f} | {split['model_roc_auc_clean']:.4f} | {split['model_f1_clean']:.4f} |")
    lines += [
        '', 'These are **classifier** metrics, not performance metrics for the FuzzyXAI framework.',
        '', '## Application policy results on held-out rows', '',
        'Figures were plotted from `wdbc/group_metrics.csv` without manual substitution.', '',
        '| Test condition | Policy | Mean auto-acceptance coverage | Accepted with critical QC flag |',
        '|---|---|---:|---:|',
    ]
    for condition, title in names:
        for policy, label in policy_names:
            rs = lookup[(condition, policy)]
            coverage = mean(float(r['coverage']) for r in rs)
            bad = sum(int(r['auto_accepted_with_critical_flag']) for r in rs)
            n = sum(int(r['n']) for r in rs)
            lines.append(f'| {title} | {label} | {coverage:.3f} | {bad}/{n} |')
    lines += [
        '', '**Interpretation.** A confidence-only rule can accept an object with a known structural quality failure. '
        'Both the canonical observer and the simpler probability-plus-equal-guardrail baseline reject all objects '
        'with that same explicit flag. This supports the importance of sharing the safety information, '
        '**not** a claim that canonical mathematical operators are uniquely superior.',
        '', 'The mean coverage differences in less severe conditions reflect a trade-off between automatic '
        'acceptance and conservative abstention. No performance gain should be claimed without matched '
        'coverage, equal information and independent, natural-event outcomes.',
        '', '## Controlled uncertainty quantification', '',
    ]
    for split in study['splits']:
        ci = split['paired_bootstrap_critical_flag_false_accept_difference_missing_40pct']
        lines.append(f"- Seed `{split['seed']}`: reduction in accepted critical-flag cases vs confidence-only = "
                     f"`{ci['paired_mean_difference']:.3f}` (paired 95% bootstrap interval "
                     f"`[{ci['ci95_low']:.3f}; {ci['ci95_high']:.3f}]`, n_boot={ci['bootstrap_repetitions']}).")
    lines += [
        '', 'The bootstrap unit is a held-out WDBC row **within a single split**; the three repeated splits '
        'are not independent clinical studies. These intervals describe a controlled failure injection, not a '
        'population confidence interval for medical safety.',
        '', '## ECG study: PTB-XL 1.0.3', '',
        '- The PTB-XL loader enforces patient-disjoint folds and uses the predeclared fold 10 only.',
        '- It accepts unmodified 12-lead 100-Hz waveform files and assesses missing or flatlined leads.',
        '- It separates real source waveforms from artificially injected missing-lead/flatline faults.',
        '- No disease labels enter the technical QC policy. PTB-XL source: '
        'https://physionet.org/content/ptb-xl/1.0.3/ (CC BY 4.0; DOI 10.13026/kfzx-aw45).',
    ]
    if ptb is None:
        lines.append('- **PTB-XL real-waveform run: NOT EXECUTED.** Original WFDB files are not supplied in this environment. Test coverage is limited to synthetic WFDB-shape stubs.')
    else:
        lines.append(f"- **PTB-XL real-waveform run: executed** on `{ptb['n_selected']}` fold-10 records; "
                     f"metadata SHA256 `{ptb['source_metadata_sha256']}`.")
        for condition, counts in ptb['condition_summaries'].items():
            lines.append(f"  - `{condition}`: {counts['actions']}; injected defects detected "
                         f"{counts['detected_injected_faults']}/{counts['n_records']}.")
    lines += [
        '', '## Strict boundaries of the evidence', '',
        '1. There is no measured clinical diagnostic accuracy for ECG, stenosis, or arrhythmia.',
        '2. `0` missed critical flags when both policies have the same guardrail is a policy invariant, not an independent demonstration of improved model accuracy.',
        '3. The WDBC stressors use synthetic corruption on real rows; no unseen real-world fault distribution was examined.',
        '4. Human comprehension, clinical usefulness, end-to-end FuzzyXAI certification and production integrations remain unevaluated.',
        '5. Code/CI reproducibility does not substitute for external datasets, independent annotations or prospective evaluation.',
        '', '## Reproduction', '',
        '```bash',
        'python -m experiments.chapter5_practical_bundle --out reports/chapter5/practical_v2',
        'python -m experiments.chapter5_practical_bundle --out reports/chapter5/practical_v2 --ptbxl /path/to/ptb-xl-1.0.3 --ptbxl-limit 300',
        '```', '',
        'Evidence: `wdbc/summary.json`, `wdbc/group_metrics.csv`, `wdbc/heldout_cases.csv`, '
        '`figures/`, `manifest_sha256.json`, optional `ptbxl/` folder.',
    ]
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def _write_russian_insert(path: Path, study: dict[str, Any], rows: list[dict[str,str]], ptb: dict[str, Any] | None) -> None:
    """Evidence-limited manuscript insert, always derived from result CSV."""
    from statistics import mean
    groups = defaultdict(list)
    for row in rows:
        groups[(row['perturbation'], row['policy'])].append(row)
    def coverage(condition: str, policy: str) -> float:
        return 100.0 * mean(float(x['coverage']) for x in groups[(condition, policy)])
    def bad_count(condition: str, policy: str) -> int:
        return sum(int(x['auto_accepted_with_critical_flag']) for x in groups[(condition, policy)])
    accuracies = [s['model_accuracy_clean'] for s in study['splits']]
    aucs = [s['model_roc_auc_clean'] for s in study['splits']]
    f1s = [s['model_f1_clean'] for s in study['splits']]
    boot_lines = []
    for split in study['splits']:
        ci = split['paired_bootstrap_critical_flag_false_accept_difference_missing_40pct']
        boot_lines.append(
            f"Для seed={split['seed']}: разность долей автоматического принятия объектов с критическим флагом "
            f"{ci['paired_mean_difference']:.3f}, парный 95%-й бутстреп-интервал "
            f"[{ci['ci95_low']:.3f}; {ci['ci95_high']:.3f}]."
        )
    lines = [
        '# Материалы для главы 5: воспроизводимая апробация наблюдателя риска', '',
        '## Постановка вычислительного эксперимента', '',
        'Для проверки способности наблюдающего контура учитывать неполноту входных данных '
        'использован общедоступный табличный набор Wisconsin Diagnostic Breast Cancer (569 объектов, 30 признаков). '
        'Эксперимент относится к методу селективного применения машинного прогноза, а не к клинической диагностике. '
        'На каждом запуске сформированы стратифицированные непересекающиеся обучающая, валидационная и '
        'отложенная тестовая части (приблизительно 60:20:20). Обучение Random Forest, вычисление медиан, '
        'масштабов и порогов выбросов выполнено только на обучающей выборке. Пороги автоматического '
        'применения установлены по выходам модели на валидации без доступа к тестовым меткам. '
        'Проверены три заранее заданных разбиения (42, 123, 2026).', '',
        '## Результаты прогнозной модели', '',
        f"Точность классификатора (accuracy) на отложенной части составила {min(accuracies):.4f}–{max(accuracies):.4f}; "
        f"ROC-AUC — {min(aucs):.4f}–{max(aucs):.4f}; "
        f"F1 — {min(f1s):.4f}–{max(f1s):.4f}. "
        'Это характеристики модели Random Forest, а не метрики эффективности FuzzyXAI.', '',
        '## Селективная политика и контрольные нарушения', '',
        'На тестовых объектах исследованы четыре условия: неизменённые данные, пропуск 10% и 40% '
        'признаков и добавление гауссовского шума с масштабом двух обучающих стандартных отклонений. '
        'Для каждого объекта использованы три политики: простой порог уверенности модели, '
        'тот же порог с идентичным критическим диагностическим флагом и канонический наблюдатель риска '
        'FuzzyXAI с указанным флагом. Последний рассчитан по пятикомпонентному оператору риска; '
        'в данном эксперименте величины, связанные с интерпретируемостью и редукцией, '
        'представлены только техническими прокси неполноты признаков и не должны трактоваться '
        'как семантические показатели глав 2–3.', '',
        '| Условие | Порог уверенности: покрытие, % | Порог + флаг: покрытие, % | Наблюдатель: покрытие, % |',
        '|---|---:|---:|---:|',
    ]
    for condition, label in [('clean','Исходные данные'),('missing_10pct','Пропуски 10%'),('missing_40pct','Пропуски 40%'),('sensor_noise_2sd','Шум 2 SD')]:
        lines.append(f"| {label} | {coverage(condition,'confidence_only'):.1f} | {coverage(condition,'confidence_equal_guardrail'):.1f} | {coverage(condition,'canonical_risk_with_qc'):.1f} |")
    bad_simple=bad_count('missing_40pct','confidence_only')
    bad_equal=bad_count('missing_40pct','confidence_equal_guardrail')
    bad_full=bad_count('missing_40pct','canonical_risk_with_qc')
    total=sum(int(x['n']) for x in groups[('missing_40pct','confidence_only')])
    lines += [
        '', f'При 40%-й потере признаков из {total} повторных тестовых прогонов простой порог уверенности '
        f'автоматически принимал {bad_simple} объектов, для которых была активна критическая проверка полноты. '
        f'Политики с одинаковым защитным флагом принимали соответственно {bad_equal} и {bad_full} таких объектов. '
        'Следовательно, эксперимент подтверждает роль явной проверки входного качества; преимущество '
        'полного фреймворка над равно информированным базовым методом из этого результата не следует.', '',
        '## Оценка неопределённости', '',
        *boot_lines, '',
        'Интервалы получены парным ресэмплированием объектов внутри каждого разбиения при '
        'искусственно заданной потере признаков. Три разбиения не являются независимыми '
        'исследованиями и не могут быть использованы для вывода о клинической безопасности.', '',
        '## Прикладной контур ЭКГ', '',
        'Для 12-отведённой ЭКГ разработан воспроизводимый протокол технической проверки сигнала: '
        'загрузка исходных записей WFDB, контроль пациент-независимого разбиения, расчёт '
        'показателей отсутствующих отсчётов и неизменного сигнала по отведениям, '
        'формирование диагностических действий и эксперименты с контролируемым удалением отведения. '
        'Источник — PTB-XL 1.0.3 (PhysioNet, DOI 10.13026/kfzx-aw45).',
    ]
    if ptb is None:
        lines.append('Исходные записи PTB-XL в данном запуске отсутствовали, поэтому численные показатели на реальных ЭКГ не приводятся; пройдены только автоматические проверки интерфейса и контролируемые тесты.')
    else:
        lines.append(f"Обработано {ptb['n_selected']} исходных 12-отведённых ЭКГ из тестовой складки 10; подробный протокол и SHA256 метаданных сохранены в экспериментальном пакете.")
    lines += [
        '', '## Границы вывода', '',
        'Эксперимент пока устанавливает воспроизводимость компонента риск-ориентированного отбора '
        'и реакцию на известные технические нарушения, но не показывает в полной мере '
        'качество семантических объяснений, человеко-ориентированную понятность, '
        'диагностическую чувствительность по ЭКГ, сравнение с равным baseline по всем '
        'предметным источникам или клиническую эффективность. '
        'Результаты разбиений, векторы решений, исходные ограничения, скрипты и '
        'контрольные суммы приложены для независимого воспроизведения.',
    ]
    path.write_text('\n'.join(lines)+'\n', encoding='utf-8')


def generate(
    out_dir: Path,
    *, seeds: tuple[int, ...] = SEED_DEFAULT,
    n_boot: int = 1500,
    ptbxl: Path | None = None,
    ptbxl_limit: int = 30,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    study = run_wdbc(out_dir / 'wdbc', seeds=seeds, n_boot=n_boot)
    create_figures(out_dir / 'wdbc/group_metrics.csv', out_dir / 'figures')
    ptb_summary = None
    if ptbxl is not None:
        ptb_summary = run_ptbxl(ptbxl, out_dir / 'ptbxl', max_records=ptbxl_limit)
    rows = _from_csv(out_dir / 'wdbc/group_metrics.csv')
    _write_markdown(out_dir / 'CHAPTER5_EMPIRICAL_RESULTS.md', study, rows, ptb_summary)
    _write_russian_insert(out_dir / 'CHAPTER5_EMPIRICAL_RESULTS_RU.md', study, rows, ptb_summary)
    manifest = {
        'schema_version': '1.0',
        'script_repo_head': _commit(),
        'python_version': platform.python_version(),
        'ptbxl_executed': ptb_summary is not None,
        'artifacts': {},
    }
    for file in sorted(out_dir.rglob('*')):
        if file.is_file() and file.name != 'manifest_sha256.json':
            manifest['artifacts'][file.relative_to(out_dir).as_posix()] = hashlib.sha256(file.read_bytes()).hexdigest()
    (out_dir / 'manifest_sha256.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return {
        'status': 'done', 'ptbxl_executed': ptb_summary is not None,
        'wdbc_splits': len(study['splits']), 'wdbc_heldout_cases': study['n_test_cases_total'],
        'evidence_files': len(manifest['artifacts']),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=ROOT / 'reports/chapter5/practical_v2')
    ap.add_argument('--seeds', default=','.join(map(str, SEED_DEFAULT)))
    ap.add_argument('--bootstrap', type=int, default=1500)
    ap.add_argument('--ptbxl', type=Path)
    ap.add_argument('--ptbxl-limit', type=int, default=30)
    args = ap.parse_args()
    seeds = tuple(int(v.strip()) for v in args.seeds.split(',') if v.strip())
    result = generate(args.out, seeds=seeds, n_boot=args.bootstrap, ptbxl=args.ptbxl, ptbxl_limit=args.ptbxl_limit)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
