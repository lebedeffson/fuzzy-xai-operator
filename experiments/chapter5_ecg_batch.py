"""Reproducible ECG technical-QC routes for dissertation chapter 5.

No clinical labels or disease-related performance metrics are inferred.
Example:
  python -m experiments.chapter5_ecg_batch \
    --csv data/raw/ecg/input_ecg_sample_signal.csv \
    --out reports/chapter5/ecg_quality_batch \
    --source-type fixture
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK = ROOT / 'framework' / 'fuzzyxai'
if str(FRAMEWORK) not in sys.path:
    sys.path.insert(0, str(FRAMEWORK))

from fuzzyxai.ecg_quality import assess_ecg_quality, extract_ecg_quality


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def read_ecg_records(path: Path) -> dict[str, dict[str, Any]]:
    """Read one/multiple records in long-form CSV, without clinical labels.

    Required: signal, sampling_rate. Optional: sample, record_id,
    noise_mask, missing_mask. record_id is never exported to public reports.
    """
    with path.open('r', encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        if not {'signal', 'sampling_rate'} <= set(reader.fieldnames or []):
            raise ValueError(f'{path.name}: CSV requires signal and sampling_rate columns')
        grouped: dict[str, dict[str, Any]] = {}
        for row_index, row in enumerate(reader):
            if None in row:
                raise ValueError(f'{path.name}: extra CSV field at row {row_index + 2}')
            rid = str(row.get('record_id') or path.stem)
            entry = grouped.setdefault(rid, {
                'samples': [], 'noise': [], 'missing': [],
                'sampling_rate': None, 'sample_indices': [],
                'noise_present': 'noise_mask' in reader.fieldnames,
                'missing_present': 'missing_mask' in reader.fieldnames,
            })
            try:
                rate = float(row['sampling_rate'])
            except (ValueError, TypeError) as exc:
                raise ValueError(f'{path.name}: invalid sampling rate at row {row_index + 2}') from exc
            if entry['sampling_rate'] is not None and entry['sampling_rate'] != rate:
                raise ValueError(f'{path.name}: inconsistent sampling rates in record')
            entry['sampling_rate'] = rate
            if 'sample' in reader.fieldnames:
                try:
                    sample_idx = int(str(row['sample']))
                except (ValueError, TypeError) as exc:
                    raise ValueError(f'{path.name}: invalid sample index at row {row_index + 2}') from exc
                indices = entry['sample_indices']
                if indices and sample_idx <= indices[-1]:
                    raise ValueError(f'{path.name}: duplicate or unordered sample indices')
                indices.append(sample_idx)
            entry['samples'].append(row['signal'])
            entry['noise'].append(row.get('noise_mask', '0'))
            entry['missing'].append(row.get('missing_mask', '0'))
    if not grouped:
        raise ValueError(f'{path.name}: empty CSV')
    return grouped


def run_batch(csv_path: Path, out_dir: Path, *, source_type: str = 'fixture') -> dict[str, Any]:
    if source_type not in ('fixture', 'external_unverified'):
        raise ValueError('source_type must be fixture or external_unverified')
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    from fuzzyxai.adapters.signal_quality import adapt_medical_ecg_signal
    from fuzzyxai.core.route import build_route
    from fuzzyxai.proof.trace import build_proof_trace
    from fuzzyxai.proof.verifier import verify_proof_trace

    source_digest = _hash(csv_path.read_bytes())
    records = read_ecg_records(csv_path)
    if out_dir.resolve() == csv_path.resolve():
        raise ValueError('out_dir cannot replace the source CSV')
    result_rows: list[dict[str, Any]] = []
    for rid, record in sorted(records.items()):
        computed = extract_ecg_quality(
            record['samples'], record['sampling_rate'],
            noise_mask=record['noise'] if record['noise_present'] else None,
            missing_mask=record['missing'] if record['missing_present'] else None,
        )
        case_id = 'ecg_' + _hash((source_digest + ':' + rid).encode('utf-8'))[:16]
        payload = {
            **computed,
            'record_ref_hash': case_id,
            'source_sha256': source_digest,
            'source_type': source_type,
            'feature_origin': 'computed_from_raw_ecg_csv',
        }
        route = build_route(adapt_medical_ecg_signal(payload))
        proof = build_proof_trace(route)
        verified = verify_proof_trace(proof)
        if not verified.valid:
            raise RuntimeError(f'{case_id}: invalid proof: {verified.errors}')
        expected = assess_ecg_quality(payload)
        if route.final_action != expected['action']:
            raise RuntimeError(f'{case_id}: route/policy mismatch')
        case_dir = out_dir / 'cases' / case_id
        _write_json(case_dir / 'features.json', payload)
        _write_json(case_dir / 'route.json', route.to_dict())
        _write_json(case_dir / 'proof_trace.json', proof.to_dict())
        result_rows.append({
            'case_id': case_id,
            'n_samples': computed['n_samples'],
            'duration_sec': computed['duration_sec'],
            'missing_fraction': computed['missing_fraction'],
            'marked_noise_fraction': computed['marked_noise_fraction'],
            'flatline_fraction': computed['flatline_fraction'],
            'quality_score': computed['quality_score'],
            'action': route.final_action,
            'diagnostic_id': route.computed_result.get('diagnostic_id', ''),
            'proof_valid': True,
            'source_sha256': source_digest,
            'source_type': source_type,
        })

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / 'cases.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result_rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(result_rows)
    counts = Counter(row['action'] for row in result_rows)
    summary = {
        'protocol': 'chapter5_ecg_technical_qc_v1',
        'source_format': 'csv',
        'source_sha256': source_digest,
        'source_type': source_type,
        'n_records': len(result_rows),
        'action_counts': dict(sorted(counts.items())),
        'proofs_verified': len(result_rows),
        'technical_quality_only': True,
        'clinical_labels_verified': False,
        'clinical_metrics': None,
        'model_retrained': False,
        'route_verification': 'FuzzyXAIProofTrace consistency checks',
        'limitations': [
            'No sensitivity/specificity/AUROC for disease or stenosis is computed.',
            'Quality thresholds are not clinically calibrated.',
            'Noise flags are annotations when provided; no medical signal interpretation.',
        ],
    }
    _write_json(out_dir / 'summary.json', summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, default=ROOT / 'data/raw/ecg/input_ecg_sample_signal.csv')
    parser.add_argument('--out', type=Path, default=ROOT / 'reports/chapter5/ecg_quality_batch')
    parser.add_argument('--source-type', choices=['fixture', 'external_unverified'], default='fixture')
    args = parser.parse_args()
    print(json.dumps(run_batch(args.csv, args.out, source_type=args.source_type), indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
