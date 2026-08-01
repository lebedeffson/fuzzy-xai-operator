#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from fuzzyxai.experiments.h10_c5b import (
    _public_incident,
    _read_sources,
    load_manifest,
)
from fuzzyxai.gold_repository import extract_gold
from fuzzyxai.repository_diagnostics.guided_retrieval import (
    documents_from_graph,
)
from fuzzyxai.repository_diagnostics.importer_v2 import (
    EvidenceGroundedRepositoryImporter,
)
from fuzzyxai.repository_diagnostics.runtime_events import load_runtime_events


def _event_paths(manifest: Path) -> dict[str, Path]:
    base = manifest.parent.resolve()
    values = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        path = Path(str(item["runtime_events_path"]))
        if not path.is_absolute():
            path = (base / path).resolve()
        values[str(item["incident_id"])] = path
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-manifest",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument("--observable-output", type=Path, required=True)
    parser.add_argument("--gold-output", type=Path, required=True)
    args = parser.parse_args()
    observable = []
    gold_rows = []
    seen = set()
    for manifest in args.source_manifest:
        events = _event_paths(manifest)
        for record in load_manifest(manifest):
            if record.incident_id in seen:
                raise ValueError(
                    f"duplicate H10-C7 incident: {record.incident_id}"
                )
            seen.add(record.incident_id)
            runtime = load_runtime_events(events[record.incident_id])
            public = _public_incident(record)
            graph = EvidenceGroundedRepositoryImporter().build(
                public,
                runtime_events=runtime,
            )
            documents = documents_from_graph(graph)
            observable.append(
                {
                    "incident_id": record.incident_id,
                    "repository": record.repository,
                    "split": "development",
                    "runtime_evidence_status": (
                        record.runtime_evidence_status
                    ),
                    "repository_symbol_count": len(documents),
                    "query": {
                        "issue": (
                            f"{public.stderr} {public.stdout}".strip()
                        ),
                        "failing_tests": record.failing_tests,
                        "traceback": public.traceback,
                        "assertion": public.assertion_difference,
                    },
                    "graph": asdict(graph),
                }
            )
            gold = extract_gold(
                record.patch_path.read_text(encoding="utf-8"),
                _read_sources(record.before_sources_path),
                _read_sources(record.after_sources_path),
            )
            if not gold.atoms:
                raise ValueError(
                    f"independent Gold is empty: {record.incident_id}"
                )
            gold_rows.append(
                {
                    "incident_id": record.incident_id,
                    "atoms": [
                        {
                            "file_path": atom.file_path,
                            "symbol": atom.symbol,
                            "contract": atom.contract,
                        }
                        for atom in gold.atoms
                    ],
                    "scorer_version": gold.scorer_version,
                }
            )
    args.observable_output.parent.mkdir(parents=True, exist_ok=True)
    args.gold_output.parent.mkdir(parents=True, exist_ok=True)
    args.observable_output.write_text(
        "".join(
            json.dumps(item, sort_keys=True) + "\n"
            for item in observable
        ),
        encoding="utf-8",
    )
    args.gold_output.write_text(
        "".join(
            json.dumps(item, sort_keys=True) + "\n"
            for item in gold_rows
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "incident_count": len(observable),
                "repository_count": len(
                    {item["repository"] for item in observable}
                ),
                "gold_channel_separate": True,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
