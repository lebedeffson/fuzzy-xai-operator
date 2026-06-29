from __future__ import annotations

import json
from pathlib import Path

from scripts.register_external_module import validate_entry, write_stub_report


def test_registration_template_validates_and_writes_report(tmp_path) -> None:
    entry = {
        'registry_id': 'tmp_module',
        'title': 'Temporary Module',
        'role': 'test module',
        'source_repo': 'https://github.com/example/tmp-module',
        'source_artifact': 'results/report.json',
        'fixture_path': 'data/fixtures/tmp_module_output.json',
        'adapter': 'tabular_to_explanation',
        'evidence_level': 'fixture-level',
        'status': 'fixture-certified',
        'chapter_role': 'chapter5_scenario',
        'claim_scope': 'test route only',
    }
    validation = validate_entry(entry)
    assert validation['ok'] is True
    assert validation['warnings']
    report = write_stub_report(entry, validation, out_dir=tmp_path)
    payload = json.loads(report.read_text(encoding='utf-8'))
    assert payload['entry']['registry_id'] == 'tmp_module'
