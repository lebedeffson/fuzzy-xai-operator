from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from fuzzyxai.datasets import list_dataset_modes, load_dataset_mode


def _load_summary(dataset_mode: str, out_root: Path) -> dict[str, Any] | None:
    p = out_root / dataset_mode / 'summary.json'
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding='utf-8'))


def _fmt(v: Any) -> str:
    if v is None:
        return 'N/A'
    return str(v)


def build_data_card(dataset_mode: str, out_root: Path) -> Path:
    out_dir = out_root / dataset_mode
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = _load_summary(dataset_mode, out_root) or {}
    record, df = load_dataset_mode(dataset_mode)
    target = str(record.target_column)
    vc = df[target].value_counts()
    n_pos = int(vc.get(1, 0))
    n_neg = int(vc.get(0, 0))
    n_total = int(len(df))
    pos_rate = float(n_pos / n_total) if n_total else 0.0
    action_labels = 'available' if bool(summary.get('observer_action_accuracy_applicable')) else 'not available'
    limitations = summary.get('limitations') or []
    applicable_metrics = [
        'model_accuracy',
        'model_roc_auc',
        'rupture_rate',
        'critical_rupture_rate',
        'observer_action_proxy_accuracy',
    ]
    if summary.get('observer_action_accuracy_applicable'):
        applicable_metrics.append('observer_action_accuracy')

    md = [
        f'# Data Card: {dataset_mode}',
        '',
        f"- source: `{record.source}`",
        f"- domain: `{record.metadata.get('domain', summary.get('domain', 'n/a'))}`",
        f"- rows: `{n_total}`",
        f"- target_column: `{target}`",
        f"- class_distribution: `n_negative={n_neg}, n_positive={n_pos}, positive_rate={pos_rate:.6f}`",
        f"- action_labels: `{action_labels}`",
        f"- applicable_metrics: `{', '.join(applicable_metrics)}`",
        f"- recommended_use_in_dissertation: `{_fmt(summary.get('recommended_use_in_dissertation'))}`",
        f"- valid_for_quantitative_claims: `{_fmt(summary.get('valid_for_quantitative_claims'))}`",
        f"- limitations: `{limitations}`",
        f"- notes: `{_fmt(summary.get('notes'))}`",
    ]
    card_path = out_dir / 'data_card.md'
    card_path.write_text('\n'.join(md), encoding='utf-8')
    return card_path


def generate_all_cards(*, out_root: str | Path = 'reports/datasets') -> list[Path]:
    root = Path(out_root)
    paths: list[Path] = []
    for spec in list_dataset_modes():
        paths.append(build_data_card(spec.key, root))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-root', default='reports/datasets')
    args = parser.parse_args()
    cards = generate_all_cards(out_root=args.out_root)
    print(json.dumps({'status': 'ok', 'cards': [str(p) for p in cards]}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

