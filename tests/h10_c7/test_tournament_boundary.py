from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pytest
import yaml
from fuzzyxai.experiments.h10_c7 import (
    load_development_inputs,
    run_development_tournament,
    validate_confirmatory_manifest,
)
from fuzzyxai.repository_diagnostics.graph import (
    EvidenceRef,
    RepositoryEdge,
    RepositoryGraph,
    RepositoryNode,
)
from fuzzyxai.repository_diagnostics.guided_diagnosis import (
    GuidedNaturalDiagnosisEngine,
)
from fuzzyxai.repository_diagnostics.guided_retrieval import (
    HashingCodeEncoder,
    IncidentQuery,
)


def _payload(index: int) -> tuple[dict[str, object], dict[str, object]]:
    repository = f"fixture/repository-{index % 10}"
    runtime = RepositoryNode(
        "runtime",
        "runtime_exception",
        repository,
        symbol="tests/test_schema.py::test_shape",
        attributes={"obligation": "shape_failure"},
    )
    graph = RepositoryGraph(
        repository,
        "buggy",
        (
            RepositoryNode(
                "cause",
                "function",
                repository,
                "src/loader.py",
                "load_schema",
                {
                    "line_count": 12,
                    "semantic_tokens": ("shape", "schema", "load"),
                },
            ),
            RepositoryNode(
                "decoy",
                "function",
                repository,
                "src/render.py",
                "render",
                {"line_count": 30, "semantic_tokens": ("render",)},
            ),
            RepositoryNode("test", "test", repository),
            runtime,
        ),
        (
            RepositoryEdge("fail", "test", "runtime", "fails_in"),
            RepositoryEdge(
                "trace",
                "cause",
                "runtime",
                "produces",
                ("trace",),
            ),
        ),
        (
            EvidenceRef(
                "trace",
                "traceback",
                "src/loader.py",
                "ValueError expected shape 4 observed shape 3",
            ),
        ),
        ("shape_failure",),
    )
    observable = {
        "incident_id": f"fixture-{index:03d}",
        "repository": repository,
        "split": "development",
        "repository_symbol_count": 100,
        "query": {
            "issue": "schema shape mismatch in load_schema",
            "failing_tests": ["tests/test_schema.py::test_shape"],
            "traceback": "ValueError in load_schema",
            "assertion": "expected shape 4 observed shape 3",
        },
        "graph": asdict(graph),
    }
    gold = {
        "incident_id": f"fixture-{index:03d}",
        "file_path": "src/loader.py",
        "symbol": "load_schema",
        "contract": "DATA_SHAPE",
    }
    return observable, gold


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    values = [_payload(index) for index in range(40)]
    manifest = tmp_path / "development.jsonl"
    gold = tmp_path / "gold.jsonl"
    manifest.write_text(
        "".join(json.dumps(value[0]) + "\n" for value in values),
        encoding="utf-8",
    )
    gold.write_text(
        "".join(json.dumps(value[1]) + "\n" for value in values),
        encoding="utf-8",
    )
    return manifest, gold


def test_gold_patch_is_rejected_from_observable_manifest(
    tmp_path: Path,
) -> None:
    manifest, gold = _write_inputs(tmp_path)
    value = json.loads(manifest.read_text().splitlines()[0])
    value["gold_patch"] = "forbidden"
    manifest.write_text(json.dumps(value) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Gold keys"):
        load_development_inputs(
            manifest,
            gold,
            minimum_incidents=1,
            minimum_repositories=1,
        )


def test_repository_level_development_tournament_is_fail_closed(
    tmp_path: Path,
) -> None:
    manifest, gold = _write_inputs(tmp_path)
    result = run_development_tournament(
        manifest,
        gold,
        tmp_path / "output",
        Path.cwd(),
        GuidedNaturalDiagnosisEngine(
            dense_encoders=(
                HashingCodeEncoder(128),
                HashingCodeEncoder(256),
            )
        ),
    )
    assert result["scientific_result"] == "NOT_EVALUATED"
    assert result["held_out_created"] is False
    assert result["held_out_scored"] is False
    assert result["development_incidents"] == 40
    assert result["development_repositories"] == 10
    assert (tmp_path / "output/DEVELOPMENT_MODEL_MATRIX.csv").is_file()
    assert (tmp_path / "output/DEVELOPMENT_GATES.json").is_file()


def test_bounded_explorer_logs_no_more_than_twelve_actions(
    route_graph: RepositoryGraph,
) -> None:
    query = _payload(0)[0]["query"]
    result = GuidedNaturalDiagnosisEngine(
        dense_encoders=(
            HashingCodeEncoder(128),
            HashingCodeEncoder(256),
        )
    ).diagnose(
        route_graph,
        IncidentQuery(
            "fixture",
            str(query["issue"]),
            tuple(query["failing_tests"]),
            str(query["traceback"]),
            str(query["assertion"]),
        ),
        "R7",
    )
    assert len(result.trajectory) <= 12
    assert all("gold" not in item.observation.lower() for item in result.trajectory)


def test_held_out_creation_is_blocked_without_passed_development(
    tmp_path: Path,
) -> None:
    held_out = tmp_path / "held-out.jsonl"
    held_out.write_text("", encoding="utf-8")
    gates = tmp_path / "gates.json"
    gates.write_text(
        json.dumps({"status": "H10_C7_BLOCKED_DEVELOPMENT_GATE"}),
        encoding="utf-8",
    )
    method_lock = tmp_path / "method-lock.json"
    method_lock.write_text(
        json.dumps({"status": "METHOD_LOCKED_BEFORE_HELD_OUT_SELECTION"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="development gate"):
        validate_confirmatory_manifest(
            held_out,
            gates,
            method_lock,
            Path.cwd(),
        )


def test_operator_addendum_is_bound_to_immutable_parent_manifest() -> None:
    addendum = yaml.safe_load(
        Path(
            "framework/fuzzyxai/operators_manifest_h10_c7_addendum.yaml"
        ).read_text(encoding="utf-8")
    )
    parent = Path(str(addendum["parent_manifest"]))
    assert hashlib.sha256(parent.read_bytes()).hexdigest() == addendum[
        "parent_manifest_sha256"
    ]
    assert addendum["status"] == "prospective_not_evaluated"


def test_model_registry_uses_exact_revisions_and_local_only_policy() -> None:
    registry = json.loads(
        Path("protocol/h10_c7/MODEL_REGISTRY.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(registry["dense_encoders"]) >= 2
    for item in (
        *registry["dense_encoders"],
        *registry["cross_encoders"],
    ):
        assert len(item["revision"]) == 40
        int(item["revision"], 16)
    assert registry["inference_policy"]["allow_network"] is False
    assert registry["inference_policy"]["local_files_only"] is True
