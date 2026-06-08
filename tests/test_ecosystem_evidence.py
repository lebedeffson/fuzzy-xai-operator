from __future__ import annotations

import csv
import json

from experiments.ecosystem_evidence_pack import run
from fuzzyxai.ecosystem import build_evidence_matrix, load_ecosystem_registry


def test_ecosystem_registry_has_expected_modules() -> None:
    modules = load_ecosystem_registry()
    ids = {m.registry_id for m in modules}
    assert {
        'hybrid_xiris',
        'gd_anfis_shap',
        'anza_lira',
        'beacon_xai',
        'gis_integro',
        'deep_neuro_fuzzy_kafn',
        'fan_multimodal',
    } <= ids
    assert all(m.source_repo.startswith('https://github.com/') or m.source_repo.startswith('source-not-provided:') for m in modules)


def test_evidence_matrix_keeps_planned_out_of_quant_claims() -> None:
    rows = build_evidence_matrix()
    planned = [r for r in rows if r['status'] == 'planned']
    assert planned
    assert all(not r['quantitative_claim_allowed'] for r in planned)
    assert any(r['run_allowed'] for r in rows)


def test_ecosystem_evidence_pack_writes_files(tmp_path) -> None:
    paths = run(evidence_dir=tmp_path / 'evidence', report_dir=tmp_path / 'chapter4')
    payload = json.loads((tmp_path / 'chapter4' / 'ecosystem_evidence.json').read_text(encoding='utf-8'))
    assert payload['module_count'] >= 5
    assert payload['runnable_count'] >= 3
    with (tmp_path / 'evidence' / 'evidence_matrix.csv').open(encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    assert rows
    assert paths['index'].endswith('reproduction_index.md')
