#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path

FORBIDDEN_KEYS = {
    "changed_files",
    "changed_symbols",
    "diff",
    "fix_commit",
    "gold",
    "gold_contract",
    "gold_file",
    "gold_patch",
    "gold_symbol",
    "patch",
}
SEALED_GOLD = (
    "fuzzyxai-h10-c7r-evidence/operation/sealed-gold/"
    "HELD_OUT_GOLD.jsonl"
)
SEALED_SOURCE = (
    "fuzzyxai-h10-c7r-evidence/operation/selection/"
    "SEALED_GOLD_SOURCE.jsonl"
)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _zip_jsonl(
    archive: zipfile.ZipFile,
    name: str,
) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in archive.read(name).decode().splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _forbidden_paths(value: object, path: str = "$") -> list[str]:
    findings: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if str(key).lower() in FORBIDDEN_KEYS:
                findings.append(child_path)
            findings.extend(_forbidden_paths(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            findings.extend(_forbidden_paths(child, f"{path}[{index}]"))
    return findings


def _changed_definitions(
    patch: str,
    graph_path: Path,
) -> list[dict[str, object]]:
    changed: dict[str, set[int]] = {}
    current_file = ""
    old_line = 0
    in_hunk = False
    for line in patch.splitlines():
        if line.startswith("--- a/"):
            current_file = line[6:]
            changed.setdefault(current_file, set())
            in_hunk = False
            continue
        match = re.match(r"@@ -(\d+)(?:,\d+)? \+\d+(?:,\d+)? @@", line)
        if match:
            old_line = int(match.group(1))
            in_hunk = True
            continue
        if not in_hunk or not current_file:
            continue
        if line.startswith("-") and not line.startswith("---"):
            changed[current_file].add(old_line)
            old_line += 1
        elif line.startswith("+") and not line.startswith("+++"):
            continue
        else:
            old_line += 1

    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    atoms: list[dict[str, object]] = []
    for node in graph["nodes"]:
        file_path = str(node.get("file_path") or "")
        attributes = node.get("attributes") or {}
        start = int(attributes.get("lineno", 0) or 0)
        end = int(attributes.get("end_lineno", start) or start)
        if (
            node.get("symbol") is not None
            and file_path in changed
            and any(start <= line <= end for line in changed[file_path])
        ):
            atoms.append(
                {
                    "file_path": file_path,
                    "symbol": str(node["symbol"]),
                    "contract": "NOT_SCORED",
                }
            )
    if not atoms:
        raise RuntimeError("sealed patch did not map to a source definition")
    return atoms


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recollection-root", type=Path, required=True)
    parser.add_argument("--v1-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.recollection_root.resolve()
    manifest_path = root / "HELD_OUT_MANIFEST.jsonl"
    readiness_path = root / "R10_RUNTIME_READINESS.json"
    manifest = _read_jsonl(manifest_path)
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    identifiers = [str(row["incident_id"]) for row in manifest]
    repositories = {str(row["repository"]) for row in manifest}
    if len(manifest) != 40 or len(set(identifiers)) != 40:
        raise RuntimeError("R10M development requires exactly 40 unique cases")
    if len(repositories) < 12:
        raise RuntimeError("R10M development requires at least 12 repositories")
    if not readiness["summary"]["all_incidents_ready"]:
        raise RuntimeError("R10M development requires 40/40 runtime readiness")
    findings = [
        path
        for index, row in enumerate(manifest)
        for path in _forbidden_paths(row, f"$[{index}]")
    ]
    if findings:
        raise RuntimeError(f"Gold leakage in observable manifest: {findings}")

    with zipfile.ZipFile(args.v1_evidence) as archive:
        old_gold = {
            str(row["incident_id"]): row
            for row in _zip_jsonl(archive, SEALED_GOLD)
        }
        source = {
            str(row["incident_id"]): row
            for row in _zip_jsonl(archive, SEALED_SOURCE)
        }

    gold = []
    derived = []
    for row in manifest:
        identifier = str(row["incident_id"])
        if identifier in old_gold:
            gold.append(old_gold[identifier])
            continue
        source_row = source.get(identifier)
        if source_row is None:
            raise RuntimeError(f"no sealed Gold source for {identifier}")
        atoms = _changed_definitions(
            str(source_row["patch"]),
            root / str(row["graph_path"]),
        )
        gold.append({"incident_id": identifier, "atoms": atoms})
        derived.append(
            {
                "incident_id": identifier,
                "derivation": "changed_old_lines_to_enclosing_source_definition",
                "atom_count": len(atoms),
                "atoms": atoms,
            }
        )

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    gold_path = output / "DEVELOPMENT_GOLD.jsonl"
    gold_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in gold),
        encoding="utf-8",
    )
    manifest_copy = output / "DEVELOPMENT_MANIFEST.jsonl"
    manifest_copy.write_bytes(manifest_path.read_bytes())
    audit = {
        "protocol_id": "H10-C7R-R10M-v1",
        "status": "PASS",
        "incident_count": len(manifest),
        "repository_count": len(repositories),
        "runtime_ready_count": readiness["summary"]["ready_incidents"],
        "observable_gold_leakage": 0,
        "observable_manifest_sha256": _sha256(manifest_path),
        "runtime_readiness_sha256": _sha256(readiness_path),
        "v1_evidence_sha256": _sha256(args.v1_evidence),
        "development_gold_sha256": _sha256(gold_path),
        "reused_sealed_gold_count": len(gold) - len(derived),
        "deterministically_derived_gold_count": len(derived),
        "derived_gold": derived,
        "scientific_result": "NOT_EVALUATED",
        "held_out_created": False,
        "held_out_scored": False,
    }
    (output / "DEVELOPMENT_INPUT_AUDIT.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
