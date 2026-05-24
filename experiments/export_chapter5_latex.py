from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

TABLES = [
    'scenarios_s0_s6',
    'baseline_comparison',
    'timing_complexity',
    'breast_cancer_validation',
    'context_logic',
    'sensitivity_w_R',
    'sensitivity_theta_high',
    'sensitivity_noise',
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-dir', default='reports/chapter5')
    parser.add_argument('--out-dir', default='reports/chapter5/latex')
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name in TABLES:
        df = pd.read_csv(Path(args.in_dir) / f'{name}.csv')
        (out / f'{name}.tex').write_text(df.to_latex(index=False, escape=True), encoding='utf-8')
    print(f'LaTeX tables: {out}')


if __name__ == '__main__':
    main()
