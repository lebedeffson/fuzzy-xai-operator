from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

TABLE_SPECS: list[tuple[str, str, str, str]] = [
    ('scenarios_s0_s6', 'table_3_13_scenarios.tex', 'Таблица 3.13: Синтетические сценарии S0-S6', 'tab:3_13'),
    ('baseline_comparison', 'table_3_16_baselines.tex', 'Таблица 3.16: Сравнение режимов R0-R3', 'tab:3_16'),
    ('breast_cancer_validation', 'table_3_17_breast_cancer.tex', 'Таблица 3.17: Валидация на breast cancer', 'tab:3_17'),
    ('timing_complexity', 'table_3_18_timing.tex', 'Таблица 3.18: Вычислительная сложность', 'tab:3_18'),
    ('sensitivity_w_R', 'table_3_20_sensitivity_wR.tex', 'Таблица 3.20: Чувствительность к весу w_R', 'tab:3_20'),
    ('sensitivity_theta_high', 'table_3_21_sensitivity_theta.tex', 'Таблица 3.21: Чувствительность к порогу theta_high', 'tab:3_21'),
    ('sensitivity_noise', 'table_3_22_sensitivity_noise.tex', 'Таблица 3.22: Устойчивость к шуму', 'tab:3_22'),
    ('context_logic', 'table_context_logic.tex', 'Контекстная логика предпучка RiskContext', 'tab:context_logic'),
]


def export_latex_tables(in_dir: str | Path = 'reports/chapter5', out_dir: str | Path = 'reports/chapter5/latex_tables') -> list[str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    produced: list[str] = []
    for source_name, target_name, caption, label in TABLE_SPECS:
        df = pd.read_csv(Path(in_dir) / f'{source_name}.csv')
        table = df.to_latex(
            index=False,
            escape=True,
            caption=caption,
            label=label,
            position='htbp',
        )
        path = out / target_name
        path.write_text(table, encoding='utf-8')
        produced.append(str(path))
    return produced


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-dir', default='reports/chapter5')
    parser.add_argument('--out-dir', default='reports/chapter5/latex_tables')
    args = parser.parse_args()
    produced = export_latex_tables(args.in_dir, args.out_dir)
    print(f'LaTeX tables: {len(produced)} files -> {args.out_dir}')


if __name__ == '__main__':
    main()
