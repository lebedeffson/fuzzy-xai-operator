from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest

from experiments.chapter5_ptbxl_qc import (
    CONDITIONS, corrupt_waveform, load_metadata, record_quality, run, safe_record_path,
)


def wave(fs=100, size=1000) -> np.ndarray:
    x = np.arange(size, dtype=float) / fs
    return np.column_stack([np.sin(2 * math.pi * (1.5 + 0.05 * i) * x) + 0.05 * np.sin(2*math.pi*6*x) for i in range(12)])


def create_metadata(root: Path, leaked: bool = False) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / 'ptbxl_database.csv'
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['ecg_id','patient_id','strat_fold','filename_lr','filename_hr'])
        w.writeheader()
        for i, fold in enumerate([1,2,9,10,10,10,10,10,10,10,10], start=1):
            pid = 'same_patient' if leaked and i in (1,4) else f'patient_{i}'
            w.writerow({'ecg_id':i,'patient_id':pid,'strat_fold':fold,'filename_lr':f'records100/00000/{i:05d}_lr','filename_hr':f'records500/00000/{i:05d}_hr'})
    return path


def test_patient_split_leakage_is_fail_closed(tmp_path: Path) -> None:
    create_metadata(tmp_path, leaked=True)
    with pytest.raises(ValueError, match='multiple folds'):
        load_metadata(tmp_path)


def test_path_traversal_and_missing_waveforms_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='WFDB path'):
        safe_record_path(tmp_path, '../../secret')
    with pytest.raises(FileNotFoundError, match='Missing WFDB'):
        safe_record_path(tmp_path, 'records100/00000/00001_lr')


def test_lead_dropout_and_flatline_are_reported_from_raw_data() -> None:
    healthy = record_quality(wave(), 100)
    assert healthy['action'] == 'audit_report'
    for kind in CONDITIONS[1:]:
        altered = record_quality(corrupt_waveform(wave(), kind), 100)
        assert altered['action'] == 'defer_to_human'
        assert altered['n_severe_leads'] >= 1
        assert altered['clinical_diagnosis'] is False


def test_mock_route_disagreement_is_detected() -> None:
    with pytest.raises(RuntimeError, match='route disagrees'):
        record_quality(
            wave(), 100,
            route_verifier=lambda _features: {
                'action': 'block', 'diagnostic_id': 'fake', 'trace_sha256': 'fake',
            },
        )


def test_multi_lead_infinite_values_fail() -> None:
    signal = wave()
    signal[10, 2] = float('inf')
    with pytest.raises(ValueError, match='non-finite'):
        record_quality(signal, 100)


def test_ptbxl_offline_study_generates_readable_evidence(tmp_path: Path) -> None:
    source = tmp_path / 'ptbxl'
    create_metadata(source)
    calls = []
    def stub_loader(root: Path, record: str, fs_expected: int):
        calls.append(record)
        assert fs_expected == 100
        return wave(), 100.0
    report = run(source, tmp_path / 'report', max_records=4, random_state=5, loader=stub_loader, verify_framework_routes=False)
    assert len(calls) == 4
    assert report['n_selected'] == 4
    assert report['n_case_rows'] == 4 * len(CONDITIONS)
    assert report['patients_verified_in_one_fold_only']
    assert report['real_waveform_file_used'] is False
    assert report['framework_routes_verified'] is False
    assert report['framework_proofs_verified'] == 0
    assert report['waveform_source'] == 'test_loader_unverified'
    assert report['clinical_diagnostic_labels_evaluated'] is False
    assert report['condition_summaries']['drop_lead_I']['detected_injected_faults'] == 4
    assert report['condition_summaries']['flatline_lead_I']['detected_injected_faults'] == 4
    assert report['condition_summaries']['original']['detected_injected_faults'] == 0
    cases = (tmp_path / 'report/ptbxl_quality_cases.csv').read_text()
    assert 'patient_' not in cases
    assert 'records100/' not in cases
    assert report == json.loads((tmp_path / 'report/summary.json').read_text())


def test_no_missing_dataset_is_not_silently_replaced_by_fixture(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        run(tmp_path / 'not_downloaded', tmp_path / 'out')
