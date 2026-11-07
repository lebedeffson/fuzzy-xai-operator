"""PTB-XL 12-lead ECG technical-quality study (optional local dataset).

No download occurs. Supply an unpacked *original* PTB-XL 1.0.3 directory,
including ptbxl_database.csv and records100/ WFDB files. The study uses fold
10 only; folds 1-8 / 9 are reserved for model development/validation.

This checks technical signal quality, NOT ECG pathology or stenosis.
Missing-lead/flatline perturbations have known technical reference labels;
natural quality or diagnostic ground truth is NOT assumed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK = ROOT / 'framework/fuzzyxai'
if str(FRAMEWORK) not in sys.path:
    sys.path.insert(0, str(FRAMEWORK))

from fuzzyxai.ecg_quality import assess_ecg_quality, extract_ecg_quality

FOLDS = {'train': set(range(1, 9)), 'val': {9}, 'test': {10}}
CONDITIONS = ('original', 'drop_lead_I', 'flatline_lead_I')
REQUIRED = {'ecg_id', 'patient_id', 'strat_fold', 'filename_lr', 'filename_hr'}


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_metadata(root: Path) -> tuple[list[dict[str, str]], str]:
    """Validate original patient-wise fold integrity before selection."""
    path = root / 'ptbxl_database.csv'
    payload = path.read_bytes()
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        missing = REQUIRED - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f'PTB-XL metadata missing fields: {sorted(missing)}')
        rows = list(reader)
    if not rows:
        raise ValueError('Empty PTB-XL metadata')
    seen: set[str] = set()
    patient_fold: dict[str, int] = {}
    for row in rows:
        ecg_id = str(row['ecg_id']).strip()
        patient = str(row['patient_id']).strip()
        if not ecg_id or not patient or patient.lower() == 'nan':
            raise ValueError('Missing ECG or patient identifier; patient split is unverifiable')
        try:
            fold = int(row['strat_fold'])
        except (TypeError, ValueError) as exc:
            raise ValueError('Invalid strat_fold') from exc
        if fold not in range(1, 11):
            raise ValueError('strat_fold outside 1..10')
        if ecg_id in seen:
            raise ValueError(f'Duplicate ecg_id {ecg_id}')
        seen.add(ecg_id)
        previous = patient_fold.setdefault(patient, fold)
        if previous != fold:
            raise ValueError('Patient appears in multiple folds: blocked possible leakage')
    return rows, _digest(payload)


def safe_record_path(root: Path, record: str) -> str:
    """Prevent path traversal if metadata was replaced/corrupted."""
    raw = str(record).strip().replace('\\', '/')
    if not raw.startswith(('records100/', 'records500/')) or raw.endswith(('.dat', '.hea')):
        raise ValueError('Invalid PTB-XL WFDB path')
    p = (root / raw).resolve()
    if not p.is_relative_to(root.resolve()):
        raise ValueError('Unsafe waveform path traversal')
    if not p.with_suffix('.hea').is_file() or not p.with_suffix('.dat').is_file():
        raise FileNotFoundError(f'Missing WFDB pair for {raw}')
    return str(p)


def get_waveform(root: Path, record: str, fs_expected: int = 100) -> tuple[np.ndarray, float]:
    """WFDB is optional; load without ever fetching data from the internet."""
    try:
        import wfdb
    except ImportError as exc:
        raise ImportError('Install optional dependency: pip install wfdb') from exc
    safe_path = safe_record_path(root, record)
    signal, fields = wfdb.rdsamp(safe_path)
    signal = np.asarray(signal, dtype=float)
    rate = float(fields['fs'])
    if signal.ndim != 2 or signal.shape[1] != 12:
        raise ValueError('PTB-XL expected a 12-lead waveform')
    if abs(rate - fs_expected) > 0.5:
        raise ValueError('Unexpected ECG sample rate')
    if len(signal) < 100 or len(signal) > 20000:
        raise ValueError('Unexpected waveform duration/shape')
    if np.any(np.isinf(signal)):
        raise ValueError('Non-finite waveform samples')
    return signal, rate


def corrupt_waveform(signal: np.ndarray, condition: str) -> np.ndarray:
    out = np.array(signal, dtype=float, copy=True)
    if condition == 'original':
        return out
    if condition == 'drop_lead_I':
        out[:, 0] = np.nan
    elif condition == 'flatline_lead_I':
        out[:, 0] = 0.0
    else:
        raise ValueError(f'Unknown condition: {condition}')
    return out


def record_quality(
    signal: np.ndarray, rate: float, *,
    route_verifier: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compute per-lead technical observations without external QC labels."""
    if signal.ndim != 2 or signal.shape[1] != 12:
        raise ValueError('A 12-lead waveform is required')
    lead_reports = []
    proof_digests: list[str] = []
    for j in range(12):
        col = signal[:, j]
        # Internally convert NaN to explicit missing values. Inf remains an error.
        values = [None if np.isnan(v) else float(v) for v in col]
        q = extract_ecg_quality(values, rate)
        decision = assess_ecg_quality(q)
        if route_verifier is not None:
            verified = route_verifier(q)
            if verified['action'] != decision['action'] or verified['diagnostic_id'] != decision['diagnostic_id']:
                raise RuntimeError('Framework route disagrees with technical quality policy')
            proof_digests.append(str(verified['trace_sha256']))
        lead_reports.append({
            'lead_index': j, 'technical_action': decision['action'],
            'missing_fraction': q['missing_fraction'],
            'flatline_fraction': q['flatline_fraction'],
            'quality_score': q['quality_score'],
            'reasons': decision['quality_reasons'],
        })
    severe = [x for x in lead_reports if x['technical_action'] == 'defer_to_human']
    warning = [x for x in lead_reports if x['technical_action'] == 'request_more_data']
    # One bad lead is not assumed to make an entire recording clinically unusable.
    if severe:
        action, diagnostic = 'defer_to_human', 'D_ecg_lead_quality_severe'
    elif warning:
        action, diagnostic = 'request_more_data', 'D_ecg_lead_quality_warning'
    else:
        action, diagnostic = 'audit_report', ''
    return {
        'action': action, 'diagnostic_id': diagnostic,
        'n_severe_leads': len(severe), 'n_warning_leads': len(warning),
        'max_missing_fraction': max(x['missing_fraction'] for x in lead_reports),
        'max_flatline_fraction': max(x['flatline_fraction'] for x in lead_reports),
        'lead_reports': lead_reports,
        'framework_proofs_verified': len(proof_digests),
        'proofs_sha256': _digest('|'.join(proof_digests).encode()) if proof_digests else None,
        'clinical_diagnosis': False,
    }


def verify_ecg_lead_via_framework(features: dict[str, Any]) -> dict[str, Any]:
    """Use the actual public framework route/proof and fail if it diverges.

    Requires the original repository with the chapter-5 dynamic ECG adapter
    patch applied; unit stubs cannot prove an end-to-end framework execution.
    """
    from fuzzyxai.adapters.signal_quality import adapt_medical_ecg_signal
    from fuzzyxai.core.route import build_route
    from fuzzyxai.proof.trace import build_proof_trace
    from fuzzyxai.proof.verifier import verify_proof_trace

    route = build_route(adapt_medical_ecg_signal(features))
    trace = build_proof_trace(route)
    verdict = verify_proof_trace(trace)
    if not verdict.valid:
        raise RuntimeError(f'Invalid FuzzyXAI trace: {verdict.errors}')
    data = trace.to_dict()
    if data.get('computed_result', {}).get('clinical_diagnosis') is not False:
        raise RuntimeError('Technical route incorrectly claims clinical diagnosis')
    return {
        'action': route.final_action,
        'diagnostic_id': route.computed_result.get('diagnostic_id', ''),
        'trace_sha256': _digest(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


def run(
    dataset: Path,
    output: Path,
    *,
    split: str = 'test',
    max_records: int = 30,
    random_state: int = 42,
    loader: Callable[[Path, str, int], tuple[np.ndarray, float]] = get_waveform,
    verify_framework_routes: bool = True,
) -> dict[str, Any]:
    if split != 'test':
        raise ValueError('This release only executes the predeclared fold-10 test protocol')
    if max_records <= 0:
        raise ValueError('max_records must be positive')
    rows, source_sha = load_metadata(dataset)
    fold_ids = FOLDS[split]
    eligible = [row for row in rows if int(row['strat_fold']) in fold_ids]
    if not eligible:
        raise ValueError('No rows for test split')
    sorted_rows = sorted(eligible, key=lambda r: int(r['ecg_id']))
    chosen = np.random.default_rng(random_state).choice(
        len(sorted_rows), size=min(len(sorted_rows), max_records), replace=False)
    selected = [sorted_rows[int(i)] for i in sorted(chosen)]
    per_record: list[dict[str, Any]] = []
    for row in selected:
        # Original metadata's artifact columns are *not* inputs to our policy.
        path = row['filename_lr']
        waveform, rate = loader(dataset, path, 100)
        for condition in CONDITIONS:
            quality = record_quality(
                corrupt_waveform(waveform, condition), rate,
                route_verifier=verify_ecg_lead_via_framework if verify_framework_routes else None,
            )
            per_record.append({
                'record_hash': _digest(f'{source_sha}:{row["ecg_id"]}'.encode())[:16],
                'split': split, 'condition': condition,
                'n_samples': len(waveform), 'n_leads': waveform.shape[1],
                'sampling_rate': rate,
                'n_severe_leads': quality['n_severe_leads'],
                'n_warning_leads': quality['n_warning_leads'],
                'max_missing_fraction': quality['max_missing_fraction'],
                'max_flatline_fraction': quality['max_flatline_fraction'],
                'technical_action': quality['action'],
                'framework_proofs_verified': quality['framework_proofs_verified'],
                'proofs_sha256': quality['proofs_sha256'] or '',
                'diagnostic_id': quality['diagnostic_id'],
                'injected_lead_fault': int(condition != 'original'),
                'known_injected_fault_detected': int(condition != 'original' and quality['action'] != 'audit_report'),
            })
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / 'ptbxl_quality_cases.csv', per_record)
    grouped = {}
    for condition in CONDITIONS:
        selected_rows = [r for r in per_record if r['condition'] == condition]
        count = Counter(r['technical_action'] for r in selected_rows)
        grouped[condition] = {
            'n_records': len(selected_rows),
            'actions': dict(sorted(count.items())),
            'detected_injected_faults': sum(r['known_injected_fault_detected'] for r in selected_rows),
        }
    summary = {
        'protocol': 'chapter5_ptbxl_technical_qc_v1',
        'dataset': 'PTB-XL version 1.0.3 (expectation; verify local dataset version separately)',
        'source_metadata_sha256': source_sha,
        'source_metadata_n_records': len(rows),
        'split': 'fold_10_patient_disjoint_holdout',
        'patients_verified_in_one_fold_only': True,
        'n_selected': len(selected),
        'selection': 'uniform_random_without_replacement_from_fold_10',
        'selection_seed': random_state,
        'n_waveform_conditions': len(CONDITIONS),
        'n_case_rows': len(per_record),
        'condition_summaries': grouped,
        'real_waveform_file_used': loader is get_waveform,
        'framework_routes_verified': verify_framework_routes,
        'framework_proofs_verified': sum(r['framework_proofs_verified'] for r in per_record),
        'waveform_source': 'local_original_WFDB' if loader is get_waveform else 'test_loader_unverified',
        'clinical_diagnostic_labels_evaluated': False,
        'natural_signal_quality_reference_available': False,
        'metric_scope': 'Detection of known artificially injected lead failures; raw-signal quality warnings on original records',
        'external_artifact_flags_not_used_by_policy': True,
        'ethical_scope': 'public de-identified PTB-XL; no patient identifiers in exported case files',
        'limitations': [
            'Raw ECG waveforms come from PTB-XL, but injected faults are controlled synthetic alterations.',
            'A clean original ECG is not verified to be noise-free; original warnings are not false positives.',
            'One or more bad leads is a technical review signal, not a clinical diagnosis or rejection decision.',
            'No claim about sensitivity or specificity for a disease can be made from this protocol.',
            'No pathologies or label metadata are used in the technical QC policy.',
        ],
    }
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset', type=Path, required=True)
    ap.add_argument('--out', type=Path, default=ROOT / 'reports/chapter5/ptbxl_technical_qc')
    ap.add_argument('--max-records', type=int, default=30)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    summary = run(args.dataset, args.out, max_records=args.max_records, random_state=args.seed)
    print(json.dumps({'protocol': summary['protocol'], 'selected': summary['n_selected'], 'out': str(args.out)}, indent=2))


if __name__ == '__main__':
    main()
