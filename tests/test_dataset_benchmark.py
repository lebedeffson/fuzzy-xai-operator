from __future__ import annotations

import json

from experiments.dataset_benchmark import run_benchmark


def test_dataset_benchmark_writes_required_files(tmp_path) -> None:
    summary = run_benchmark('breast_cancer', out_root=tmp_path)
    assert summary['status'] == 'READY'
    out_dir = tmp_path / 'breast_cancer'
    assert (out_dir / 'summary.json').exists()
    assert (out_dir / 'summary.md').exists()
    assert (out_dir / 'predictions.csv').exists()
    payload = json.loads((out_dir / 'summary.json').read_text(encoding='utf-8'))
    for key in [
        'dataset', 'status', 'n', 'domain', 'model_accuracy', 'model_roc_auc',
        'observer_action_accuracy', 'mean_I_pre', 'mean_rho', 'rupture_rate',
        'action_distribution', 'selected_representation_distribution', 'notes',
    ]:
        assert key in payload
