from __future__ import annotations

from pathlib import Path

from experiments.ablation_benchmark import run_ablation
from experiments.baseline_comparison import run as run_baseline_comparison
from experiments.calibrate_observer import calibrate
from experiments.defense_cases import generate_defense_cases
from experiments.export_thesis_practice_tables import export


def test_calibration_report_generation(tmp_path) -> None:
    out = calibrate('breast_cancer', out_root=tmp_path)
    assert out['status'] == 'READY'
    assert 'before_calibration' in out and 'after_calibration' in out
    assert (tmp_path / 'breast_cancer' / 'calibration.json').exists()
    assert (tmp_path / 'breast_cancer' / 'calibration.md').exists()


def test_ablation_report_generation(tmp_path) -> None:
    out = run_ablation('breast_cancer', out_root=tmp_path)
    assert out['status'] == 'READY'
    assert len(out['rows']) >= 6
    assert (tmp_path / 'breast_cancer' / 'ablation.json').exists()
    assert (tmp_path / 'breast_cancer' / 'ablation.md').exists()


def test_baseline_comparison_report_generation(tmp_path) -> None:
    out = run_baseline_comparison('breast_cancer', out_root=tmp_path)
    assert out['status'] == 'READY'
    assert len(out['rows']) >= 3
    assert (tmp_path / 'breast_cancer' / 'baseline_comparison.json').exists()
    assert (tmp_path / 'breast_cancer' / 'baseline_comparison.md').exists()


def test_defense_cases_generation(tmp_path) -> None:
    out = generate_defense_cases(out_dir=tmp_path)
    assert out['dataset'] == 'breast_cancer'
    assert len(out['cases']) == 3
    assert (tmp_path / 'accept_case.json').exists()
    assert (tmp_path / 'audit_case.json').exists()
    assert (tmp_path / 'block_case.json').exists()
    assert (tmp_path / 'summary.md').exists()


def test_export_thesis_tables_generation(tmp_path) -> None:
    files = export(out_dir=tmp_path)
    assert all((tmp_path / Path(path).name).exists() for path in files.values())
