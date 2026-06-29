from __future__ import annotations

import csv
import json

from experiments.integration_effort_report import CSV_PATH, SUMMARY_PATH, run


def test_integration_effort_report_has_protocol_fields() -> None:
    payload = run()
    assert payload['status'] == 'ok'
    with CSV_PATH.open(encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    required = {'method', 'module_id', 'operator_id', 'start_time', 'end_time', 'duration_minutes', 'steps_completed', 'semantic_gap_detected', 'notes'}
    assert rows
    assert required <= set(rows[0])
    assert json.loads(SUMMARY_PATH.read_text(encoding='utf-8'))['measurement_count'] == len(rows)
