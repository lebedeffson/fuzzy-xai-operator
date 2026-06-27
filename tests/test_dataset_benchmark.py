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
        'critical_rupture_rate', 'observer_action_proxy_accuracy',
        'agreement_proxy', 'agreement_proxy_applicable', 'agreement_proxy_reason',
        'i_pre_mean', 'i_pre_std', 'i_pre_median', 'i_pre_p25', 'i_pre_p75', 'i_pre_p05', 'i_pre_p95',
        'rho_mean', 'rho_std', 'rho_median', 'rho_p25', 'rho_p75', 'rho_p05', 'rho_p95',
        'metric_interpretation', 'valid_for_quantitative_claims', 'limitations',
        'recommended_use_in_dissertation',
        'action_distribution', 'selected_representation_distribution', 'notes',
    ]:
        assert key in payload


def test_registry_programs_observer_accuracy_not_applicable(tmp_path) -> None:
    summary = run_benchmark('registry_programs', out_root=tmp_path)
    assert summary['status'] == 'READY'
    assert summary['observer_action_accuracy_applicable'] is False
    assert summary['observer_action_accuracy'] is None
    assert summary['agreement_proxy_applicable'] is False
    assert summary['agreement_proxy'] is None
    assert 'no expert action labels' in str(summary['observer_action_accuracy_reason']).lower()


def test_synthetic_ruptures_has_nonzero_rupture_rate(tmp_path) -> None:
    summary = run_benchmark('synthetic_ruptures', out_root=tmp_path)
    assert summary['status'] == 'READY'
    assert float(summary['rupture_rate']) > 0.0
