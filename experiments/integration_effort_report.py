from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / 'reports' / 'chapter4' / 'integration_effort_measurements.csv'
SUMMARY_PATH = ROOT / 'reports' / 'chapter4' / 'integration_effort_summary.json'


def run() -> dict[str, Any]:
    rows = list(csv.DictReader(CSV_PATH.open(encoding='utf-8')))
    for row in rows:
        start = datetime.fromisoformat(row['start_time'])
        end = datetime.fromisoformat(row['end_time'])
        measured = int((end - start).total_seconds() // 60)
        if measured != int(row['duration_minutes']):
            raise ValueError(f"duration mismatch for {row['module_id']}:{row['operator_id']}")
    total = sum(int(r['duration_minutes']) for r in rows)
    by_method: dict[str, int] = {}
    for row in rows:
        by_method[row['method']] = by_method.get(row['method'], 0) + int(row['duration_minutes'])
    payload = {
        'status': 'ok',
        'measurement_count': len(rows),
        'total_duration_minutes': total,
        'by_method_minutes': by_method,
        'semantic_gap_detected_count': sum(1 for r in rows if r['semantic_gap_detected'].lower() == 'true'),
        'source_csv': str(CSV_PATH.relative_to(ROOT)),
        'claim_scope': 'integration effort protocol only; not production estimate',
    }
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return payload


if __name__ == '__main__':
    print(json.dumps(run(), ensure_ascii=False, indent=2))
