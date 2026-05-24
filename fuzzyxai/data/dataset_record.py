from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetRecord:
    name: str
    source: str
    local_path: Path | None = None
    url: str | None = None
    file_format: str | None = None
    target_column: str | None = None
    task_type: str | None = None
    description: str = ''
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_trace(self) -> dict[str, Any]:
        return {
            'dataset_name': self.name,
            'source': self.source,
            'url': self.url,
            'local_path': str(self.local_path) if self.local_path else None,
            'file_format': self.file_format,
            'target_column': self.target_column,
            'task_type': self.task_type,
            'description': self.description,
            'metadata': dict(self.metadata),
        }
