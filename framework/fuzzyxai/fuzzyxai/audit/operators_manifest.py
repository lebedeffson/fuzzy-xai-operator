from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any

import yaml

FRAMEWORK_ROOT = Path(__file__).resolve().parents[2]
SOURCE_REPOSITORY = Path.cwd() if (Path.cwd() / "framework/fuzzyxai/operators_manifest.yaml").exists() else FRAMEWORK_ROOT.parents[1]
REPOSITORY_ROOT = SOURCE_REPOSITORY
DEFAULT_MANIFEST = SOURCE_REPOSITORY / "framework/fuzzyxai/operators_manifest.yaml"


def _resolve_callable(reference: str) -> Any:
    module_name, separator, attribute = reference.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError(f"invalid callable reference: {reference}")
    value = importlib.import_module(module_name)
    for part in attribute.split("."):
        value = getattr(value, part)
    if not callable(value):
        raise TypeError(f"manifest target is not callable: {reference}")
    return value


def validate_manifest(path: str | Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    """Validate operator traceability against importable code and repository evidence."""

    manifest_path = Path(path)
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    rows = payload.get("operators", []) if isinstance(payload, dict) else []
    errors: list[str] = []
    ids: set[str] = set()
    required = {"id", "dissertation_ref", "callable", "input_schema", "output_schema", "tests", "artifacts", "visualization"}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append(f"row[{index}]:not_mapping")
            continue
        missing = sorted(required - set(row))
        if missing:
            errors.append(f"row[{index}]:missing:{','.join(missing)}")
            continue
        operator_id = str(row["id"])
        if operator_id in ids:
            errors.append(f"duplicate:{operator_id}")
        ids.add(operator_id)
        try:
            _resolve_callable(str(row["callable"]))
        except Exception as exc:
            errors.append(f"{operator_id}:callable:{exc}")
        for field in ("tests", "artifacts"):
            values = row.get(field)
            if not isinstance(values, list) or not values:
                errors.append(f"{operator_id}:{field}:empty")
                continue
            for value in values:
                if not (REPOSITORY_ROOT / str(value)).exists():
                    errors.append(f"{operator_id}:{field}:missing:{value}")
    return {
        "schema_version": payload.get("schema_version") if isinstance(payload, dict) else None,
        "manifest": str(manifest_path),
        "operator_count": len(rows),
        "operator_ids": sorted(ids),
        "status": "PASS" if rows and not errors else "FAIL",
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output")
    args = parser.parse_args()
    report = validate_manifest(args.manifest)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
