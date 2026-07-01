#!/usr/bin/env python3
from __future__ import annotations

from fuzzyxai.adapters import get_adapter, list_adapters


def main() -> int:
    adapters = list_adapters()
    ids = {item["adapter_id"] for item in adapters}
    if "tabular_classification" not in ids:
        raise SystemExit("tabular_classification adapter is not registered")
    adapter = get_adapter("tabular_classification")
    description = adapter.describe()
    for key in ("adapter_id", "task_type", "supported_payload_schema", "required_fields"):
        if key not in description:
            raise SystemExit(f"adapter description missing {key}")
    invalid = adapter.validate_payload({})
    if invalid.valid or not invalid.errors:
        raise SystemExit("adapter validation should fail on empty payload")
    print("fuzzyxai-adapter-sdk-check: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
