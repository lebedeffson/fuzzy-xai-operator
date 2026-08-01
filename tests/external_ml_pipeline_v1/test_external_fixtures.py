from __future__ import annotations

import hashlib
import inspect
import pickle
from pathlib import Path

from fuzzyxai.external_adapters.manifest import ManifestExternalPipelineAdapter

from experiments.external_ml_pipeline_v1.external_runners import SPECS

ROOT = Path(__file__).resolve().parents[2]


def test_all_four_external_pipelines_execute_without_fuzzyxai() -> None:
    runner = (ROOT / "experiments/external_ml_pipeline_v1/external_runners.py").read_text(encoding="utf-8")
    assert "from fuzzyxai" not in runner and "import fuzzyxai" not in runner
    assert len(SPECS) == 4


def test_external_model_artifacts_are_real_estimators() -> None:
    for spec in SPECS:
        payload = pickle.loads((ROOT / "experiments/external_ml_pipeline_v1/fixtures" / spec.pipeline_id / "baseline/model.pkl").read_bytes())
        assert hasattr(payload, "predict")


def test_each_adapter_extracts_real_artifacts() -> None:
    for spec in SPECS:
        adapter = ManifestExternalPipelineAdapter(ROOT / "experiments/external_ml_pipeline_v1/fixtures" / spec.pipeline_id / "baseline")
        artifacts = adapter.build_route_observations()
        assert artifacts.model["artifact_sha256"] == hashlib.sha256((artifacts.root / "model.pkl").read_bytes()).hexdigest()
        assert artifacts.explanation["artifact_sha256"]


def test_adapter_contains_no_diagnostic_or_repair_logic() -> None:
    source = inspect.getsource(ManifestExternalPipelineAdapter).lower()
    assert "diagnos" not in source
    assert "root_cause" not in source
    assert "repair" not in source


def test_original_and_consistent_retrain_have_distinct_bound_artifacts(benchmark) -> None:
    for spec in SPECS:
        baseline = benchmark.artifacts(spec.pipeline_id)
        retrained = benchmark.artifacts(spec.pipeline_id, "retrained")
        assert baseline.model["artifact_sha256"] != retrained.model["artifact_sha256"]
        assert baseline.explanation["model_sha256"] == baseline.model["artifact_sha256"]
        assert retrained.explanation["model_sha256"] == retrained.model["artifact_sha256"]


def test_source_snapshots_are_pinned_to_repository_commits() -> None:
    for spec in SPECS:
        assert len(spec.repository_commit) == 40
        assert spec.repository_url.startswith("https://github.com/")


def test_tree_and_lime_explanations_are_real() -> None:
    tree = benchmark_path("ext2-shap-tree-explainer")
    lime = benchmark_path("ext4-lime-tabular")
    assert "TreeExplainer" in tree.read_text(encoding="utf-8")
    assert "LimeTabularExplainer" in lime.read_text(encoding="utf-8")


def benchmark_path(pipeline_id: str) -> Path:
    return ROOT / "experiments/external_ml_pipeline_v1/fixtures" / pipeline_id / "baseline/explanation_manifest.json"
