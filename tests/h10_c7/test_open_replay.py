from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from fuzzyxai.experiments.h10_c7 import load_development_inputs
from fuzzyxai.experiments.h10_c7_replay import verify_h10_c5c_baseline
from fuzzyxai.repository_diagnostics.graph import RepositoryGraph
from fuzzyxai.repository_diagnostics.guided_diagnosis import (
    GuidedNaturalDiagnosisEngine,
)
from fuzzyxai.repository_diagnostics.guided_retrieval import IncidentQuery


def test_published_h10_c5c_baseline_is_replayable() -> None:
    result = verify_h10_c5c_baseline(
        Path("results/h10_c5c/DEVELOPMENT_PER_INCIDENT.csv"),
        Path("results/h10_c5c/H10_C5C_DEVELOPMENT_STATUS.json"),
    )
    assert result["status"] == "H10_C7_OPEN_REPLAY_BASELINE_PASS"
    assert result["published_metrics"]["candidate_recall_at_10"] == 17 / 30
    assert result["published_metrics"]["coverage"] == 0.8


def test_compact_manifest_loads_graph_by_relative_path(
    tmp_path: Path,
    route_graph: RepositoryGraph,
) -> None:
    graph_path = tmp_path / "repository_graphs/fixture.json"
    graph_path.parent.mkdir()
    graph_path.write_text(
        json.dumps(asdict(route_graph), sort_keys=True),
        encoding="utf-8",
    )
    manifest = tmp_path / "incidents.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "incident_id": "fixture",
                "repository": "example/repository",
                "split": "development",
                "query": {
                    "issue": "shape mismatch",
                    "failing_tests": ["tests/test_loader.py::test_shape"],
                    "traceback": "ValueError in load_schema",
                    "assertion": "expected shape 4 observed shape 3",
                },
                "graph_path": "repository_graphs/fixture.json",
                "repository_symbol_count": 10,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    gold = tmp_path / "gold.jsonl"
    gold.write_text(
        json.dumps(
            {
                "incident_id": "fixture",
                "atoms": [
                    {
                        "file_path": "src/loader.py",
                        "symbol": "load_schema",
                        "contract": "DATA_CONTRACT",
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    incidents, labels = load_development_inputs(
        manifest,
        gold,
        minimum_incidents=1,
        minimum_repositories=1,
    )
    assert incidents[0].graph.repository == route_graph.repository
    assert labels["fixture"].atoms[0].contract == "DATA_CONTRACT"


def test_structural_r3_does_not_require_neural_weights(
    route_graph: RepositoryGraph,
) -> None:
    result = GuidedNaturalDiagnosisEngine(structural_only=True).diagnose(
        route_graph,
        IncidentQuery(
            "fixture",
            "schema shape mismatch",
            ("tests/test_loader.py::test_shape",),
            "ValueError in load_schema",
            "expected shape 4 observed shape 3",
        ),
        "R3",
    )
    assert result.status != "VARIANT_UNAVAILABLE"
    assert result.candidates
    assert all(
        "dense:" not in source
        for candidate in result.candidates
        for source in candidate.rank_sources
    )


def test_unknown_contract_remains_candidate_not_confirmed(
    route_graph: RepositoryGraph,
) -> None:
    result = GuidedNaturalDiagnosisEngine().diagnose(
        route_graph,
        IncidentQuery(
            "fixture",
            "",
            ("tests/test_loader.py::test_unknown",),
            "",
            "",
        ),
        "R1",
    )
    assert result.candidates[0].contract.family == "UNKNOWN_CONTRACT"
    assert result.status == "DIAGNOSIS_CANDIDATES"
