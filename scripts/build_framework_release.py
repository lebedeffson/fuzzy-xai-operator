"""Build and verify a clean source archive from the committed Git tree."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "release_artifacts"
FORBIDDEN_PARTS = {
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "site/dubnaxai",
}
FORBIDDEN_SUFFIXES = {".docx", ".pdf", ".pyc", ".pyo", ".zip"}
ALLOWED_ROOTS = {
    ".github",
    "configs",
    "data_manifests",
    "docs",
    "examples",
    "framework",
    "research",
    "scripts",
    "study",
    "tests",
}
ALLOWED_EXACT = {
    "AGENTS.md",
    ".dockerignore",
    "CHANGELOG.md",
    "CITATION.cff",
    "CONTRIBUTING.md",
    "Dockerfile",
    "Dockerfile.q1",
    "Dockerfile.q1-final",
    "Dockerfile.practical",
    "LICENSE",
    "Makefile",
    "PROJECT_MEMORY.md",
    "README.md",
    "docker-compose.yml",
    "docker-compose.q1.yml",
    "docker-compose.q1-final.yml",
    "RELEASE_NOTES.md",
    "RELEASE_STATUS.md",
    "TEST_REPORT.txt",
    "pyproject.toml",
    "requirements.txt",
    "requirements.lock",
    "uv.lock",
}
ALLOWED_PREFIXES = (
    "experiments/real_training_experiment/",
    "release_evidence/explanation_experience/",
    "release_evidence/chapter4_explanation_experience/",
    "release_evidence/controlled_fixtures/",
    "release_evidence/empirical_experiments/",
    "release_evidence/user_study/",
    "release_evidence/chapter4_empirical_validation/",
    "release_evidence/model_universality/",
    "release_evidence/explanation_quality/",
    "release_evidence/domain_language_review/",
    "release_evidence/chapter4_final_candidate/",
    "release_evidence/final_confirmatory_closure/",
    "experiments/model_universality/",
    "reports/release/universal_model_integration_completion.md",
)
REQUIRED_PATHS = {
    "fuzzy-xai-operator/PROJECT_MEMORY.md",
    "fuzzy-xai-operator/RELEASE_STATUS.md",
    "fuzzy-xai-operator/TEST_REPORT.txt",
    "fuzzy-xai-operator/framework/fuzzyxai/operators_manifest.yaml",
    "fuzzy-xai-operator/framework/fuzzyxai/fuzzyxai/evidence/contracts.py",
    "fuzzy-xai-operator/framework/fuzzyxai/fuzzyxai/evidence/claims.py",
    "fuzzy-xai-operator/framework/fuzzyxai/fuzzyxai/evidence/graph.py",
    "fuzzy-xai-operator/framework/fuzzyxai/fuzzyxai/evidence/human.py",
    "fuzzy-xai-operator/framework/fuzzyxai/fuzzyxai/runtime.py",
    "fuzzy-xai-operator/framework/fuzzyxai/fuzzyxai/schemas/human_explanation.schema.json",
    "fuzzy-xai-operator/framework/fuzzyxai/fuzzyxai/schemas/explanation_visual_spec.schema.json",
    "fuzzy-xai-operator/framework/fuzzyxai/fuzzyxai/visualization/explanation_dashboard.py",
    "fuzzy-xai-operator/framework/fuzzyxai/fuzzyxai/visualization/matplotlib_renderer.py",
    "fuzzy-xai-operator/framework/fuzzyxai/fuzzyxai/visualization/plotly_renderer.py",
    "fuzzy-xai-operator/framework/fuzzyxai/fuzzyxai/visualization/spec.py",
    "fuzzy-xai-operator/framework/fuzzyxai/explanation_experience_manifest.json",
    "fuzzy-xai-operator/framework/fuzzyxai/matlab/+fuzzyxai/dashboard.m",
    "fuzzy-xai-operator/framework/fuzzyxai/matlab/+fuzzyxai/explanationStory.m",
    "fuzzy-xai-operator/examples/object_85_training_trace.py",
    "fuzzy-xai-operator/scripts/verify_explanation_experience.py",
    "fuzzy-xai-operator/scripts/score_comprehension_pilot.py",
    "fuzzy-xai-operator/scripts/build_chapter4_explanation_evidence.py",
    "fuzzy-xai-operator/docs/user_comprehension_study.md",
    "fuzzy-xai-operator/docs/human_explanation_layer.md",
    "fuzzy-xai-operator/docs/AI_PRE_REVIEW_FINAL_BLINDING.md",
    "fuzzy-xai-operator/tests/test_human_explanation_layer.py",
    "fuzzy-xai-operator/framework/fuzzyxai/fuzzyxai/ai_pre_review_final/contracts.py",
    "fuzzy-xai-operator/framework/fuzzyxai/fuzzyxai/ai_pre_review_final/generator.py",
    "fuzzy-xai-operator/framework/fuzzyxai/fuzzyxai/ai_pre_review_final/audit.py",
    "fuzzy-xai-operator/scripts/ai_pre_review_final/audit_blinding.py",
    "fuzzy-xai-operator/scripts/ai_pre_review_final/build_public_bundle.py",
    "fuzzy-xai-operator/scripts/ai_pre_review_final/validate_evidence.py",
    "fuzzy-xai-operator/study/ai_pre_review_final/reviewer_case_schema_v2.json",
    "fuzzy-xai-operator/study/ai_pre_review_final/public_formative/manifest.json",
    "fuzzy-xai-operator/study/ai_pre_review_final/public_formative/reviewer_cases.jsonl",
    "fuzzy-xai-operator/tests/ai_pre_review_final/test_final_blinding.py",
    "fuzzy-xai-operator/release_evidence/explanation_experience/object_85_human_explanation.json",
    "fuzzy-xai-operator/release_evidence/explanation_experience/medical_research_human_explanation.json",
    "fuzzy-xai-operator/release_evidence/explanation_experience/comprehension_pilot/response_template.csv",
    "fuzzy-xai-operator/release_evidence/chapter4_explanation_experience/manifest_sha256.json",
    "fuzzy-xai-operator/experiments/real_training_experiment/run_empirical_validation.py",
    "fuzzy-xai-operator/scripts/build_empirical_chapter4_evidence.py",
    "fuzzy-xai-operator/scripts/verify_empirical_validation.py",
    "fuzzy-xai-operator/release_evidence/empirical_experiments/breast_cancer_checkpoint/empirical_summary.json",
    "fuzzy-xai-operator/release_evidence/chapter4_empirical_validation/manifest_sha256.json",
    "fuzzy-xai-operator/release_evidence/user_study/comprehension_pilot/scoring_report.json",
    "fuzzy-xai-operator/framework/fuzzyxai/fuzzyxai/adapters/contracts_v2.py",
    "fuzzy-xai-operator/framework/fuzzyxai/fuzzyxai/adapters/model_v2.py",
    "fuzzy-xai-operator/framework/fuzzyxai/fuzzyxai/adapters/model_registry.py",
    "fuzzy-xai-operator/framework/fuzzyxai/fuzzyxai/adapters/sklearn_v2.py",
    "fuzzy-xai-operator/framework/fuzzyxai/fuzzyxai/adapters/optional_v2.py",
    "fuzzy-xai-operator/framework/fuzzyxai/fuzzyxai/adapter_conformance.py",
    "fuzzy-xai-operator/framework/fuzzyxai/fuzzyxai/planner.py",
    "fuzzy-xai-operator/framework/fuzzyxai/fuzzyxai/explanation_quality.py",
    "fuzzy-xai-operator/framework/fuzzyxai/fuzzyxai/schemas/adapter_validation_report.schema.json",
    "fuzzy-xai-operator/experiments/model_universality/run_benchmark.py",
    "fuzzy-xai-operator/experiments/model_universality/runtime_validation.py",
    "fuzzy-xai-operator/scripts/merge_model_validation_reports.py",
    "fuzzy-xai-operator/scripts/verify_model_universality.py",
    "fuzzy-xai-operator/scripts/verify_explanation_quality.py",
    "fuzzy-xai-operator/release_evidence/model_universality/summary.json",
    "fuzzy-xai-operator/release_evidence/model_universality/model_support_matrix.csv",
    "fuzzy-xai-operator/release_evidence/model_universality/model_support_matrix.json",
    "fuzzy-xai-operator/release_evidence/model_universality/public_api_verification.json",
    "fuzzy-xai-operator/release_evidence/model_universality/manifest.json",
    "fuzzy-xai-operator/release_evidence/explanation_quality/explanation_quality_report.json",
    "fuzzy-xai-operator/release_evidence/explanation_quality/checksums.sha256",
    "fuzzy-xai-operator/scripts/build_external_validation_package.py",
    "fuzzy-xai-operator/scripts/verify_external_release_gates.py",
    "fuzzy-xai-operator/release_evidence/domain_language_review/review_record.json",
    "fuzzy-xai-operator/scripts/build_chapter4_final_candidate.py",
    "fuzzy-xai-operator/scripts/verify_chapter4_final_candidate.py",
    "fuzzy-xai-operator/release_evidence/chapter4_final_candidate/manifest_sha256.json",
    "fuzzy-xai-operator/configs/full_empirical_validation.json",
    "fuzzy-xai-operator/configs/calibration_grid.yaml",
    "fuzzy-xai-operator/data_manifests/full_empirical_validation.json",
    "fuzzy-xai-operator/framework/fuzzyxai/fuzzyxai/experiments/contracts.py",
    "fuzzy-xai-operator/framework/fuzzyxai/fuzzyxai/experiments/protocols.py",
    "fuzzy-xai-operator/framework/fuzzyxai/fuzzyxai/experiments/statistics.py",
    "fuzzy-xai-operator/scripts/run_full_empirical_validation.py",
    "fuzzy-xai-operator/scripts/run_optional_multimodal_models.py",
    "fuzzy-xai-operator/scripts/verify_full_empirical_validation.py",
    "fuzzy-xai-operator/scripts/reproduce_all.py",
    "fuzzy-xai-operator/scripts/verify_reproduction.py",
    "fuzzy-xai-operator/scripts/verify_empirical_archives.py",
    "fuzzy-xai-operator/tests/test_full_empirical_contracts.py",
    "fuzzy-xai-operator/study/expert_review/reviewer_form.md",
    "fuzzy-xai-operator/research/preregistration/q1_hypotheses.yaml",
    "fuzzy-xai-operator/research/preregistration/q1_analysis_plan.md",
    "fuzzy-xai-operator/research/preregistration/q1_baseline_snapshot.json",
    "fuzzy-xai-operator/research/preregistration/q1_external_gates.json",
    "fuzzy-xai-operator/framework/fuzzyxai/fuzzyxai/q1_validation/schemas.py",
    "fuzzy-xai-operator/framework/fuzzyxai/fuzzyxai/q1_validation/protocols.py",
    "fuzzy-xai-operator/scripts/q1/reproduce_all.py",
    "fuzzy-xai-operator/scripts/q1/verify_all.py",
    "fuzzy-xai-operator/tests/test_q1_validation_contracts.py",
    "fuzzy-xai-operator/Dockerfile.q1",
    "fuzzy-xai-operator/docker-compose.q1.yml",
    "fuzzy-xai-operator/Dockerfile.q1-final",
    "fuzzy-xai-operator/docker-compose.q1-final.yml",
    "fuzzy-xai-operator/configs/q1_final/protocol.json",
    "fuzzy-xai-operator/data_manifests/q1_final_datasets.json",
    "fuzzy-xai-operator/framework/fuzzyxai/fuzzyxai/q1_final/contracts.py",
    "fuzzy-xai-operator/framework/fuzzyxai/fuzzyxai/q1_final/multiclass.py",
    "fuzzy-xai-operator/framework/fuzzyxai/fuzzyxai/q1_final/hypotheses.py",
    "fuzzy-xai-operator/scripts/q1_final/reproduce_all.py",
    "fuzzy-xai-operator/scripts/q1_final/verify_all.py",
    "fuzzy-xai-operator/scripts/q1_final/verify_external_gates.py",
    "fuzzy-xai-operator/docs/reproduction/Q1_FINAL_REPRODUCTION.md",
    "fuzzy-xai-operator/tests/q1_final/test_final_contracts.py",
    "fuzzy-xai-operator/framework/fuzzyxai/fuzzyxai/practical_controller/contracts.py",
    "fuzzy-xai-operator/framework/fuzzyxai/fuzzyxai/practical_controller/runtime.py",
    "fuzzy-xai-operator/framework/fuzzyxai/fuzzyxai/practical_controller/deployment.py",
    "fuzzy-xai-operator/scripts/final_practical_closure/build_protocol.py",
    "fuzzy-xai-operator/scripts/final_practical_closure/verify_formative.py",
    "fuzzy-xai-operator/scripts/final_practical_closure/lock_protocol.py",
    "fuzzy-xai-operator/scripts/final_practical_closure/run_confirmatory.py",
    "fuzzy-xai-operator/study/final_practical_closure/practical_protocol.json",
    "fuzzy-xai-operator/tests/practical_controller/test_practical_controller.py",
    "fuzzy-xai-operator/docs/PRACTICAL_CONTROLLER.md",
    "fuzzy-xai-operator/.github/workflows/final-practical-closure.yml",
    "fuzzy-xai-operator/.github/workflows/final-confirmatory-closure.yml",
    "fuzzy-xai-operator/framework/fuzzyxai/fuzzyxai/final_closure/contracts.py",
    "fuzzy-xai-operator/framework/fuzzyxai/fuzzyxai/final_closure/datasets.py",
    "fuzzy-xai-operator/framework/fuzzyxai/fuzzyxai/final_closure/faults.py",
    "fuzzy-xai-operator/framework/fuzzyxai/fuzzyxai/final_closure/ablation.py",
    "fuzzy-xai-operator/scripts/final_closure/build_protocol.py",
    "fuzzy-xai-operator/scripts/final_closure/seal_datasets.py",
    "fuzzy-xai-operator/scripts/final_closure/lock_protocol.py",
    "fuzzy-xai-operator/scripts/final_closure/verify_prelock.py",
    "fuzzy-xai-operator/scripts/final_closure/build_prelock_bundle.py",
    "fuzzy-xai-operator/tests/final_closure/test_final_closure.py",
    "fuzzy-xai-operator/docs/FINAL_CONFIRMATORY_CLOSURE.md",
    "fuzzy-xai-operator/Dockerfile.practical",
    "fuzzy-xai-operator/uv.lock",
    "fuzzy-xai-operator/configs/ai_pre_review/config.json",
    "fuzzy-xai-operator/framework/fuzzyxai/fuzzyxai/ai_pre_review/contracts.py",
    "fuzzy-xai-operator/framework/fuzzyxai/fuzzyxai/ai_pre_review/generator.py",
    "fuzzy-xai-operator/framework/fuzzyxai/fuzzyxai/ai_pre_review/review_io.py",
    "fuzzy-xai-operator/scripts/ai_pre_review/build_log.py",
    "fuzzy-xai-operator/scripts/ai_pre_review/verify_pipeline.py",
    "fuzzy-xai-operator/study/ai_pre_review/rubric_v1.yaml",
    "fuzzy-xai-operator/study/ai_pre_review/ai_review_schema.json",
    "fuzzy-xai-operator/study/ai_pre_review/human_response_schema.json",
    "fuzzy-xai-operator/study/ai_pre_review/source_case_evidence.jsonl",
    "fuzzy-xai-operator/tests/ai_pre_review/test_pipeline.py",
}


def run_git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_artifact_paths(export_root: Path) -> set[str]:
    """Return repository-relative evidence paths declared by the operator manifest."""

    manifest_path = export_root / "framework/fuzzyxai/operators_manifest.yaml"
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    rows = payload.get("operators", []) if isinstance(payload, dict) else []
    artifacts: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        for value in row.get("artifacts", []):
            relative = Path(str(value))
            if relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError(f"operator manifest contains unsafe artifact path: {value}")
            if not (export_root / relative).is_file():
                raise RuntimeError(f"operator manifest artifact is missing from HEAD: {value}")
            artifacts.add(relative.as_posix())
    if not artifacts:
        raise RuntimeError("operator manifest does not declare any evidence artifacts")
    return artifacts


def validate_archive(path: Path, manifest_artifacts: set[str]) -> tuple[int, list[str]]:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        name_set = set(names)
        manifest_paths = {f"fuzzy-xai-operator/{item}" for item in manifest_artifacts}
        missing = sorted((REQUIRED_PATHS | manifest_paths) - name_set)
        if missing:
            raise RuntimeError(f"release archive is missing required files: {missing}")
        forbidden = []
        for name in names:
            normalized = name.rstrip("/")
            if any(part in normalized for part in FORBIDDEN_PARTS):
                forbidden.append(name)
            if Path(normalized).suffix in FORBIDDEN_SUFFIXES:
                forbidden.append(name)
        if forbidden:
            raise RuntimeError(f"release archive contains generated or quarantined files: {sorted(set(forbidden))}")
        with tempfile.TemporaryDirectory() as temp_dir:
            archive.extractall(temp_dir)
            extracted_root = Path(temp_dir) / "fuzzy-xai-operator"
            if not (extracted_root / "pyproject.toml").is_file():
                raise RuntimeError("archive does not extract to one installable root")
    return len(names), missing


def include_in_source_release(path: Path, manifest_artifacts: set[str]) -> bool:
    relative = path.as_posix()
    if path.suffix.lower() in FORBIDDEN_SUFFIXES:
        return False
    if relative in ALLOWED_EXACT:
        return True
    if relative in manifest_artifacts:
        return True
    if path.parts and path.parts[0] in ALLOWED_ROOTS:
        return True
    return relative.startswith(ALLOWED_PREFIXES)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    commit = run_git("rev-parse", "HEAD")
    head_tree = run_git("rev-parse", "HEAD^{tree}")
    index_tree = run_git("write-tree")
    if index_tree != head_tree:
        raise RuntimeError("release index differs from HEAD; commit staged changes before packaging")
    short_commit = commit[:12]
    archive_path = OUTPUT_DIR / f"fuzzyxai-source-release-{short_commit}.zip"
    with tempfile.TemporaryDirectory() as temp_dir:
        export_root = Path(temp_dir) / "fuzzy-xai-operator"
        export_root.mkdir()
        subprocess.run(
            ["git", "checkout-index", "--all", f"--prefix={export_root}{os.sep}"],
            cwd=ROOT,
            check=True,
        )
        # The historical website and editor state are deliberately outside the
        # framework release, while tracked evidence remains available to tests.
        shutil.rmtree(export_root / "site" / "dubnaxai", ignore_errors=True)
        shutil.rmtree(export_root / ".vscode", ignore_errors=True)
        manifest_artifacts = manifest_artifact_paths(export_root)
        with zipfile.ZipFile(
            archive_path,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as archive:
            for source in sorted(path for path in export_root.rglob("*") if path.is_file()):
                if not include_in_source_release(source.relative_to(export_root), manifest_artifacts):
                    continue
                mode = 0o100755 if source.stat().st_mode & 0o111 else 0o100644
                info = zipfile.ZipInfo(
                    source.relative_to(export_root.parent).as_posix(),
                    date_time=(1980, 1, 1, 0, 0, 0),
                )
                info.external_attr = mode << 16
                archive.writestr(
                    info,
                    source.read_bytes(),
                    compress_type=zipfile.ZIP_DEFLATED,
                    compresslevel=9,
                )
    file_count, _ = validate_archive(archive_path, manifest_artifacts)
    archive_hash = sha256(archive_path)
    checksum_path = archive_path.with_suffix(".zip.sha256")
    checksum_path.write_text(f"{archive_hash}  {archive_path.name}\n", encoding="ascii")
    manifest = {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "commit": commit,
        "branch": run_git("branch", "--show-current"),
        "archive": archive_path.name,
        "sha256": archive_hash,
        "file_count": file_count,
        "source": "git checkout-index at HEAD; quarantined site and editor state pruned",
        "quarantined_site_included": False,
    }
    manifest_path = OUTPUT_DIR / "source_release_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: source_release_archive {archive_path}")
    print(f"PASS: source_release_sha256 {archive_hash}")
    print(f"PASS: source_release_cleanliness files={file_count}")


if __name__ == "__main__":
    main()
