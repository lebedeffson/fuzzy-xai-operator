from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest

from fuzzyxai.ecg_quality import assess_ecg_quality, extract_ecg_quality
from experiments.chapter5_ecg_batch import read_ecg_records, run_batch


def signal(n=1000):
    return [math.sin(i / 8) for i in range(n)]


def test_raw_good_signal_never_becomes_clinical_accept() -> None:
    features = extract_ecg_quality(signal(), 250)
    action = assess_ecg_quality(features)
    assert features['quality_score'] > 0.90
    assert action['action'] == 'audit_report'
    assert action['automatic_clinical_accept_allowed'] is False


def test_missing_and_noisy_samples_trigger_defer() -> None:
    samples = signal()
    samples[50:201] = [''] * 151
    noise = [0] * 1000
    noise[250:500] = [1] * 250
    features = extract_ecg_quality(samples, 250, noise_mask=noise)
    result = assess_ecg_quality(features)
    assert features['missing_samples'] == 151
    assert features['missing_runs'] == 1
    assert features['noise_segments'] == 1
    assert result['action'] == 'defer_to_human'
    assert 'missing_signal' in result['quality_reasons']
    assert 'marked_noise' in result['quality_reasons']


def test_short_fixture_and_small_missing_are_distinct() -> None:
    short = extract_ecg_quality(signal(10), 250)
    assert assess_ecg_quality(short)['action'] == 'defer_to_human'
    large = signal()
    large[10] = ''
    caution = extract_ecg_quality(large, 250)
    assert assess_ecg_quality(caution)['action'] == 'request_more_data'


def test_constant_signal_never_passes_technical_gate() -> None:
    features = extract_ecg_quality([0.0] * 1000, 250)
    assert features['flatline_fraction'] > 0.99
    assert assess_ecg_quality(features)['action'] == 'defer_to_human'


def test_nonfinite_and_inconsistent_masks_rejected() -> None:
    with pytest.raises(ValueError, match='non-finite'):
        extract_ecg_quality([0.1, float('nan')], 250)
    with pytest.raises(ValueError, match='noise_mask length'):
        extract_ecg_quality([0, 1], 250, noise_mask=[1])
    with pytest.raises(ValueError, match='inconsistent'):
        assess_ecg_quality({**extract_ecg_quality(signal(), 250), 'missing_fraction': 0.3})


def test_batch_records_keep_source_identity_private(tmp_path: Path) -> None:
    source = tmp_path / 'sample.csv'
    with source.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['record_id', 'sample', 'signal', 'sampling_rate'])
        writer.writeheader()
        for rid in ('patient-secret-id-1', 'patient-secret-id-2'):
            for idx, val in enumerate(signal(280)):
                writer.writerow({'record_id': rid, 'sample': idx, 'signal': val, 'sampling_rate': 250})
    records = read_ecg_records(source)
    assert len(records) == 2
    summary = run_batch(source, tmp_path / 'out', source_type='external_unverified')
    assert summary['n_records'] == 2
    assert summary['proofs_verified'] == 2
    assert summary['clinical_metrics'] is None
    published = (tmp_path / 'out/cases.csv').read_text(encoding='utf-8')
    assert 'patient-secret' not in published
    assert 'source_sha256' in published


def test_multiple_record_sampling_rate_mismatch_rejected(tmp_path: Path) -> None:
    source = tmp_path / 'x.csv'
    source.write_text('record_id,sample,signal,sampling_rate\nx,0,0.1,250\nx,1,0.2,360\n', encoding='utf-8')
    with pytest.raises(ValueError, match='inconsistent sampling rates'):
        read_ecg_records(source)


def test_public_route_decides_from_payload_and_verifies_proof() -> None:
    from fuzzyxai.adapters.signal_quality import adapt_medical_ecg_signal
    from fuzzyxai.core.route import build_route
    from fuzzyxai.proof.trace import build_proof_trace
    from fuzzyxai.proof.verifier import verify_proof_trace

    clean_payload = extract_ecg_quality(signal(), 250)
    noisy = extract_ecg_quality(signal(), 250, noise_mask=[1] * 1000)
    a = build_route(adapt_medical_ecg_signal(clean_payload))
    b = build_route(adapt_medical_ecg_signal(noisy))
    assert a.final_action == 'audit_report'
    assert b.final_action == 'defer_to_human'
    for route in [a, b]:
        assert verify_proof_trace(build_proof_trace(route)).valid
        assert route.computed_result['clinical_diagnosis'] is False


def test_prepopulated_quality_score_is_not_used_as_raw_evidence(tmp_path: Path) -> None:
    source = tmp_path / 'fake_score.csv'
    source.write_text(
        'sample,signal,sampling_rate,quality_score\n'
        + ''.join(f'{i},0,250,1.0\n' for i in range(400)),
        encoding='utf-8',
    )
    data = read_ecg_records(source)[source.stem]
    features = extract_ecg_quality(data['samples'], data['sampling_rate'])
    assert features['flatline_fraction'] > 0.99
    assert features['quality_score'] < 0.65
    assert assess_ecg_quality(features)['action'] == 'defer_to_human'
