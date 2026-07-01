from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import quantiles
from typing import Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]


def _memberships(p: float) -> tuple[float, float, float]:
    return max(0.0, 1.0 - p), max(0.0, 1.0 - abs(p - 0.5) / 0.25), max(0.0, p)


def _entropy(vals: tuple[float, float, float]) -> float:
    arr = np.asarray(vals, dtype=float)
    total = float(arr.sum()) or 1.0
    probs = arr / total
    return float(-(probs * np.log2(np.clip(probs, 1e-12, 1.0))).sum())


def _q95(values: list[float]) -> float:
    if not values:
        return 0.0
    if len(values) < 20:
        return max(values)
    return float(quantiles(values, n=20, method='inclusive')[18])


def run(out_dir: str | Path = 'reports/chapter2', seed: int = 42, n_pairs: int = 50) -> dict[str, Any]:
    out = ROOT / out_dir
    fig_dir = ROOT / 'figures/chapter2'
    out.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    data = load_breast_cancer(as_frame=True)
    X_train, X_cal, y_train, _y_cal = train_test_split(data.data, data.target, test_size=0.25, random_state=seed, stratify=data.target)
    model = RandomForestClassifier(n_estimators=80, random_state=seed, max_depth=6)
    model.fit(X_train, y_train)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_cal), size=min(n_pairs, len(X_cal)), replace=False)
    cal = X_cal.iloc[idx]
    scale = X_train.std(axis=0).replace(0, 1.0)

    rows: list[dict[str, Any]] = []
    h_vals: list[float] = []
    o_vals: list[float] = []
    k_vals: list[float] = []
    for pair_id, (_, row) in enumerate(cal.iterrows(), start=1):
        noise = rng.normal(0.0, 0.03, size=row.shape[0]) * scale.to_numpy()
        raw = row.to_numpy(dtype=float)
        pert = raw + noise
        raw_df = row.to_frame().T
        pert_df = raw_df.copy()
        pert_df.iloc[0, :] = pert
        p0 = float(model.predict_proba(raw_df)[0][1])
        p1 = float(model.predict_proba(pert_df)[0][1])
        mu0 = _memberships(p0)
        mu1 = _memberships(p1)
        h = abs(_entropy(mu0) - _entropy(mu1))
        o = abs(max(mu0) - max(mu1)) + abs(p0 - p1)
        top0 = set(np.argsort(np.abs(raw / scale.to_numpy()))[-5:])
        top1 = set(np.argsort(np.abs(pert / scale.to_numpy()))[-5:])
        k = 1.0 - len(top0 & top1) / max(1, len(top0 | top1))
        h_vals.append(float(h)); o_vals.append(float(o)); k_vals.append(float(k))
        rows.append({'pair_id': pair_id, 'p_raw': p0, 'p_perturbed': p1, 'H_distance': h, 'O_distance': o, 'K_distance': k, 'dataset': 'breast_cancer', 'seed': seed})

    constants = [
        {'constant_name': 'c_H_cal', 'value': _q95(h_vals), 'method': 'q95', 'n_pairs': len(rows), 'dataset': 'breast_cancer'},
        {'constant_name': 'c_O_cal', 'value': _q95(o_vals), 'method': 'q95', 'n_pairs': len(rows), 'dataset': 'breast_cancer'},
        {'constant_name': 'c_K_cal', 'value': _q95(k_vals), 'method': 'q95', 'n_pairs': len(rows), 'dataset': 'breast_cancer'},
    ]

    with (out / 'calibration_pairs.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    with (out / 'calibration_constants.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(constants[0].keys()))
        writer.writeheader(); writer.writerows(constants)
    report = {'status': 'ok', 'dataset': 'breast_cancer', 'seed': seed, 'n_pairs': len(rows), 'method': 'q95', 'constants': constants}
    (out / 'calibration_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    (out / 'section_2_10_insert.md').write_text(
        'Раздел 2.10 закрывается калибровочным протоколом: константы `c_H`, `c_O`, `c_K` вычисляются на `C_cal` '
        'из Breast Cancer Wisconsin с фиксированным seed и методом `q95`. Машинно-читаемые результаты находятся в '
        '`reports/chapter2/calibration_report.json` и `reports/chapter2/calibration_constants.csv`.\n',
        encoding='utf-8',
    )

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar([c['constant_name'] for c in constants], [float(c['value']) for c in constants], color=['#0f766e', '#2563eb', '#9333ea'])
    ax.set_title('Chapter 2 calibrated constants (q95)')
    ax.grid(axis='y', alpha=0.25)
    fig.tight_layout(); fig.savefig(fig_dir / 'calibration_constants.png', dpi=180); plt.close(fig)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', default='reports/chapter2')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n-pairs', type=int, default=50)
    args = parser.parse_args()
    print(json.dumps(run(args.out_dir, args.seed, args.n_pairs), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
