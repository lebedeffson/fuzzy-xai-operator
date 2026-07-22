from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any


def canonical_trace(payload: dict[str, Any]) -> bytes:
    def default(value: Any) -> Any:
        try:
            return asdict(value)
        except TypeError:
            return str(value)

    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=default).encode("ascii")
