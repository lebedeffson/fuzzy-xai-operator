from __future__ import annotations

from experiments.unified_full_demo import run, write_real_data_artifacts


def test_unified_full_demo_real_data_modes_smoke(tmp_path):
    for dataset in ('rikord', 'ruccod'):
        report, i_values, diagnostics = run(dataset=dataset, seed=1, sample_index=0, allow_fallback=True)
        assert 'dataset_metadata' in report
        assert report['dataset_metadata']['requested_dataset'] == dataset
        assert 0.0 <= float(report['risk']['rho']) <= 1.0
        assert len(i_values) > 0
        paths = write_real_data_artifacts(dataset, report, i_values, diagnostics, tmp_path)
        assert paths['json'].exists()
        assert paths['summary_csv'].exists()
        assert paths['i_pre_csv'].exists()
        assert paths['thresholds_csv'].exists()

