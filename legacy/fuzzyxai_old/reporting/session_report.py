from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


@dataclass
class SessionReport:
    """Serializable report for dashboard sessions.

    If ``log_path`` is set, every step is also appended to a CSV audit log.
    The CSV file is intentionally flat and reproducible: name, timestamp and a
    compact JSON payload are written for each dashboard action.
    """

    title: str
    steps: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    log_path: Optional[str | Path] = None

    def add_step(self, name: str, payload: Mapping[str, Any]) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        record = {
            'name': name,
            'timestamp': timestamp,
            'payload': dict(payload),
        }
        self.steps.append(record)
        self._append_csv(record)

    def _append_csv(self, record: Mapping[str, Any]) -> None:
        if not self.log_path:
            return
        path = Path(self.log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists()
        with path.open('a', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['timestamp', 'name', 'payload_json'])
            if write_header:
                writer.writeheader()
            writer.writerow({
                'timestamp': record.get('timestamp'),
                'name': record.get('name'),
                'payload_json': json.dumps(record.get('payload', {}), ensure_ascii=False, sort_keys=True),
            })

    def to_dict(self) -> Dict[str, Any]:
        return {
            'title': self.title,
            'created_at': datetime.now(timezone.utc).isoformat(),
            'metadata': dict(self.metadata),
            'steps': list(self.steps),
            'log_path': str(self.log_path) if self.log_path else None,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
