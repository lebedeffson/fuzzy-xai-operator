from __future__ import annotations

import csv
import json

from scripts.build_chapter2_3_final_evidence import run


def test_chapter2_3_final_evidence_matrix_is_complete(tmp_path) -> None:
    payload = run(tmp_path / 'chapter2_3')
    assert payload['status'] == 'ok'
    assert payload['implemented_rows'] == payload['total_rows'] == 15
    assert payload['missing'] == []

    rows = list(csv.DictReader((tmp_path / 'chapter2_3/chapter2_3_implementation_matrix.csv').open(encoding='utf-8')))
    object_ids = {row['object_id'] for row in rows}
    assert {'ExplainPlan', 'T_ij_synthesis', 'F_hierarchy', 'chi_Auto_topos', 'controlled_critical_ruptures'} <= object_ids
    assert all(row['status'] == 'implemented' for row in rows)
    assert all(row['code_paths'] and row['test_paths'] and row['artifact_paths'] for row in rows)

    manifest = json.loads(__import__('pathlib').Path('evidence/chapter2_3_manifest_sha256.json').read_text(encoding='utf-8'))
    assert manifest['implemented_rows'] == 15
    assert all(item['exists'] and item['sha256'] for item in manifest['files'])
