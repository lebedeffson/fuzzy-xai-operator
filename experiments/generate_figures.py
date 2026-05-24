from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import plotly.express as px


def _maybe_save_png(fig, png_path: Path) -> bool:
    try:
        fig.write_image(str(png_path))
        return True
    except Exception:
        return False


def generate_figures(
    chapter2_dir: str | Path = 'reports/chapter2',
    chapter5_dir: str | Path = 'reports/chapter5',
    out_dir: str | Path = 'reports/figures',
) -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    produced: dict[str, str] = {}

    c2 = Path(chapter2_dir)
    c5 = Path(chapter5_dir)

    i_pre = pd.read_csv(c2 / 'chapter2_breast_cancer_i_pre.csv')
    fig_i = px.histogram(i_pre, x='i_pre', nbins=20, title='I_pre distribution (breast cancer)')
    i_html = out / 'i_pre_distribution.html'
    fig_i.write_html(i_html, include_plotlyjs='cdn')
    produced['i_pre_html'] = str(i_html)
    i_png = out / 'i_pre_distribution.png'
    if _maybe_save_png(fig_i, i_png):
        produced['i_pre_png'] = str(i_png)

    wr = pd.read_csv(c5 / 'sensitivity_w_R.csv')
    fig_wr = px.line(wr, x='w_R', y='accuracy', markers=True, title='Accuracy vs w_R')
    wr_html = out / 'accuracy_vs_w_R.html'
    fig_wr.write_html(wr_html, include_plotlyjs='cdn')
    produced['wR_html'] = str(wr_html)
    wr_png = out / 'accuracy_vs_w_R.png'
    if _maybe_save_png(fig_wr, wr_png):
        produced['wR_png'] = str(wr_png)

    theta = pd.read_csv(c5 / 'sensitivity_theta_high.csv')
    fig_theta = px.line(theta, x='theta_high', y='accuracy', markers=True, title='Accuracy vs theta_high')
    theta_html = out / 'accuracy_vs_theta_high.html'
    fig_theta.write_html(theta_html, include_plotlyjs='cdn')
    produced['theta_html'] = str(theta_html)
    theta_png = out / 'accuracy_vs_theta_high.png'
    if _maybe_save_png(fig_theta, theta_png):
        produced['theta_png'] = str(theta_png)

    base = pd.read_csv(c5 / 'baseline_comparison.csv')
    fig_base = px.bar(base, x='mode', y='accuracy', title='Mode comparison (accuracy)', text='accuracy')
    base_html = out / 'baseline_accuracy_bar.html'
    fig_base.write_html(base_html, include_plotlyjs='cdn')
    produced['baseline_html'] = str(base_html)
    base_png = out / 'baseline_accuracy_bar.png'
    if _maybe_save_png(fig_base, base_png):
        produced['baseline_png'] = str(base_png)

    return produced


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--chapter2-dir', default='reports/chapter2')
    parser.add_argument('--chapter5-dir', default='reports/chapter5')
    parser.add_argument('--out-dir', default='reports/figures')
    args = parser.parse_args()
    files = generate_figures(args.chapter2_dir, args.chapter5_dir, args.out_dir)
    print(f'Figures generated: {len(files)} -> {args.out_dir}')


if __name__ == '__main__':
    main()
