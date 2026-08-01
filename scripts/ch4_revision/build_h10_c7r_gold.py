#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

HUNK = re.compile(r"^@@ -(\d+)(?:,\d+)? \+\d+(?:,\d+)? @@")


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def changed_locations(patch: str) -> list[tuple[str, int]]:
    current: str | None = None
    locations: list[tuple[str, int]] = []
    for line in patch.splitlines():
        if line.startswith("+++ b/"):
            current = line[6:]
            continue
        match = HUNK.match(line)
        if current is not None and match is not None:
            locations.append((current, int(match.group(1))))
    return locations


def symbol_for_location(
    graph: dict[str, object],
    file_path: str,
    line: int,
) -> str | None:
    candidates = []
    for node in graph.get("nodes", []):
        if not isinstance(node, dict) or node.get("file_path") != file_path:
            continue
        symbol = node.get("symbol")
        if not symbol:
            continue
        attributes = node.get("attributes", {})
        if not isinstance(attributes, dict):
            attributes = {}
        start = int(attributes.get("lineno", 0) or 0)
        end = int(attributes.get("end_lineno", start) or start)
        contains = start <= line <= max(start, end)
        distance = 0 if contains else min(abs(line - start), abs(line - end))
        candidates.append((not contains, distance, -start, str(symbol)))
    return min(candidates)[-1] if candidates else None


def build_gold(
    sealed_source: Path,
    manifest: Path,
    output: Path,
) -> dict[str, object]:
    manifest_rows = {
        str(row["incident_id"]): row for row in read_jsonl(manifest)
    }
    gold_rows = []
    for source in read_jsonl(sealed_source):
        identifier = str(source["incident_id"])
        if identifier not in manifest_rows:
            continue
        row = manifest_rows[identifier]
        graph_path = Path(str(row["graph_path"]))
        if not graph_path.is_absolute():
            graph_path = manifest.parent / graph_path
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
        atoms = []
        seen = set()
        for file_path, line in changed_locations(str(source["patch"])):
            symbol = symbol_for_location(graph, file_path, line)
            key = (file_path, symbol)
            if key in seen:
                continue
            seen.add(key)
            atoms.append(
                {
                    "file_path": file_path,
                    "symbol": symbol,
                    "contract": "NOT_SCORED",
                }
            )
        if not atoms:
            raise ValueError(f"no Gold localization atoms for {identifier}")
        gold_rows.append({"incident_id": identifier, "atoms": atoms})
    if set(manifest_rows) != {str(row["incident_id"]) for row in gold_rows}:
        raise ValueError("Gold and observable incident sets differ")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in gold_rows),
        encoding="utf-8",
    )
    output.chmod(0o600)
    return {
        "incidents": len(gold_rows),
        "status": "H10_C7R_GOLD_MATERIALIZED_AFTER_RUNTIME_LOCK",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sealed-source", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            build_gold(
                args.sealed_source.resolve(),
                args.manifest.resolve(),
                args.output.resolve(),
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
