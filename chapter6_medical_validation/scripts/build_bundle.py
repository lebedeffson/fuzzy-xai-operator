"""Create a self-contained reviewer bundle without raw datasets or checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

from chapter6_medical_validation.shared.hashing import sha256_file
from chapter6_medical_validation.shared.reproducibility import environment_facts, hash_if_exists

ROOT = Path(__file__).resolve().parents[1]
BUNDLES = ROOT / "bundles"
EXCLUDED_RUNTIME_SUBTREES = frozenset({
    "ai_pre_review", "audit", "experiments", "final_closure", "practice",
    "q1_final", "q1_validation", "realdata", "strong_confirmatory",
})


def copy_tree(source: Path, target: Path, *, patterns: tuple[str, ...] = ("*",)) -> None:
    if source.is_file():
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, target); return
    for pattern in patterns:
        for path in source.rglob(pattern):
            if (
                path.is_file() and "__pycache__" not in path.parts and path.suffix not in {".pyc", ".pt"}
                and not (set(path.relative_to(source).parts) & EXCLUDED_RUNTIME_SUBTREES)
            ):
                destination = target / path.relative_to(source); destination.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, destination)


def selected_case_names(domain: str, output_name: str) -> set[str]:
    selected = json.loads((ROOT / domain / output_name / "cases" / "selected_cases.json").read_text(encoding="utf-8"))["cases"]
    return {str(value["object_id"]) for value in selected.values() if isinstance(value, dict) and "object_id" in value}


def stage_domain(stage: Path, domain: str, output_name: str = "outputs") -> None:
    source = ROOT / domain; target = stage / "chapter6_medical_validation" / domain
    for name in ("README_RU.md", "SOURCES.md", "CORE_GAP_REPORT.md", "DATA_ACCESS.md", "DATA_ACCESS_REQUIRED_IDRID.md"):
        if (source / name).is_file(): copy_tree(source / name, target / name)
    for name in ("configs", "src", "scripts", "tests"):
        if (source / name).exists(): copy_tree(source / name, target / name)
    if domain == "ophthalmology": return
    outputs = source / output_name
    for name in ("runs", "controls", "training"):
        if (outputs / name).exists():
            # Model metadata and predictions are enough for review; never ship checkpoints.
            copy_tree(outputs / name, target / output_name / name, patterns=("*.json", "*.npz", "*.png", "*.txt"))
    cases = outputs / "cases"; names = selected_case_names(domain, output_name)
    for filename in ("selected_candidates.json", "selected_cases.json", "case_summaries.json"):
        if (cases / filename).is_file(): copy_tree(cases / filename, target / output_name / "cases" / filename)
    for name in names:
        if (cases / name).is_dir(): copy_tree(cases / name, target / output_name / "cases" / name)


def stage_papila(stage: Path, data_root: Path) -> None:
    """Stage reproducible PAPILA metadata and derived evidence, never raw data/checkpoints."""
    source = data_root / "eyes" / "papila"
    target = stage / "chapter6_medical_validation" / "ophthalmology" / "papila_artifacts"
    for path in (
        source / "selected_cases_eye_v3.json", source / "xai_sweep_fold5.json",
        source / "xai_sweep_suspect.json", source / "suspect_predictions.json",
        source / "papila_lightweight_public_route_v1.json",
        source / "verified" / "papila_cv_folds_seed2026.json",
        source / "verified" / "papila_eye_labels.csv",
        source / "controls_v4" / "controls_summary.json",
    ):
        if path.is_file(): copy_tree(path, target / path.name)
    run = source / "runs" / "papila-resnet50-fold5-seed2026-d2caccba5926"
    for name in ("run.json", "test_predictions.json", "validation_predictions.json"):
        if (run / name).is_file(): copy_tree(run / name, target / "canonical_run" / name)
    selected = json.loads((source / "selected_cases_eye_v3.json").read_text(encoding="utf-8"))
    samples = {row["sample_id"] for group in (selected["cases"], selected["suspect_cases"]) for row in group.values()}
    for sample in sorted(samples):
        case = source / "cases_v2" / sample
        for name in ("result.json", "audit.json", "reader_ru.txt", "audit_ru.txt", "inspect_action.json", "provenance_action.png", "provenance_action.json", "lime.json", "lime_signed_map.npy", "lime_positive_map.npy", "lime_superpixels.npy", "grad_cam_raw.npy", "strict_slm.json"):
            if (case / name).is_file(): copy_tree(case / name, target / "cases" / sample / name)
    for control in sorted((source / "controls_v4").glob("CONTROL_*")):
        for name in ("result.json", "audit.json", "reader_ru.txt", "inspect_action.json", "provenance_action.png"):
            if (control / name).is_file(): copy_tree(control / name, target / "controls" / control.name / name)


def reproducibility() -> dict[str, object]:
    data_root = Path(os.environ["FUZZYXAI_CH6_DATA_ROOT"])
    payload: dict[str, object] = {"environment": environment_facts(), "raw_data_in_bundle": False, "runs": []}
    for domain, dataset_path, plan, output_name, config in (
        ("ECG", data_root / "ecg" / "ptb-xl-1.0.3" / "prepared", ROOT / "ecg_ptbxl" / "configs" / "explain_plan_ecg.yaml", "outputs", "model_ecg_resnet1d.yaml"),
        ("BRAIN_V1_PILOT", data_root / "brain" / "allen_ccf_25um" / "prepared", ROOT / "brain_allen" / "configs" / "explain_plan_brain.yaml", "outputs", "model_inceptionv3.yaml"),
        ("BRAIN_V2_CONFIRMATORY", data_root / "brain" / "allen_ccf_25um" / "prepared_v2_confirmatory", ROOT / "brain_allen" / "configs" / "explain_plan_brain.yaml", "outputs_v2_confirmatory", "model_inceptionv3_v2_confirmatory.yaml"),
    ):
        folder = ROOT / ("ecg_ptbxl" if domain == "ECG" else "brain_allen") / output_name / "runs"
        for run_path in sorted(folder.glob("*/run.json")):
            run = json.loads(run_path.read_text(encoding="utf-8"))
            payload["runs"].append({
                "domain": domain, "run_id": run["run_id"], "seed": run["seed"],
                "dataset_manifest_sha256": hash_if_exists(dataset_path / "dataset_manifest.json"),
                "split_manifest_sha256": hash_if_exists(dataset_path / "patches.json") or hash_if_exists(dataset_path / "folds.npy"),
                "model_config_sha256": hash_if_exists(ROOT / ("ecg_ptbxl" if domain == "ECG" else "brain_allen") / "configs" / config),
                "checkpoint_sha256": run.get("checkpoint_sha256"), "explain_plan_sha256": sha256_file(plan),
                "result_artifact_paths": [str(path.relative_to(ROOT)) for path in (ROOT / ("ecg_ptbxl" if domain == "ECG" else "brain_allen") / output_name / "cases").glob("*/result.json")],
            })
    papila = data_root / "eyes" / "papila"
    payload["papila"] = {
        "dataset": "PAPILA Figshare 14798004 v2", "raw_data_in_bundle": False,
        "split_manifest_sha256": hash_if_exists(papila / "verified" / "papila_cv_folds_seed2026.json"),
        "selected_cases_sha256": hash_if_exists(papila / "selected_cases_eye_v3.json"),
        "canonical_run_id": "papila-resnet50-fold5-seed2026-d2caccba5926",
    }
    regression_log = BUNDLES / "ch6_final_full_regression.log"
    regression_summary = None
    if regression_log.is_file():
        for line in reversed(regression_log.read_text(encoding="utf-8", errors="replace").splitlines()):
            if " passed" in line and " skipped" in line and " in " in line:
                regression_summary = line.strip()
                break
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT.parent,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    ).stdout.strip()
    payload["final_regression"] = {
        "head_commit": head or None,
        "command": "PYTHONPATH=framework/fuzzyxai:. /home/lebedeffson/Code/venv/bin/python -m pytest -q",
        "summary": regression_summary,
        "log_sha256": hash_if_exists(regression_log),
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-name", default="CH6_MEDICAL_FUZZYXAI_VALIDATION_FINAL")
    args = parser.parse_args()
    if "FUZZYXAI_CH6_DATA_ROOT" not in os.environ: raise FileNotFoundError("FUZZYXAI_CH6_DATA_ROOT is not set")
    stage = BUNDLES / args.bundle_name
    if stage.exists(): shutil.rmtree(stage)
    target = stage / "chapter6_medical_validation"; target.mkdir(parents=True)
    for name in ("README_RU.md", "CH6_PROTOCOL.md", "ENVIRONMENT_RU.md", "requirements_ch6.txt"):
        copy_tree(ROOT / name, target / name)
    for name in ("reports", "tables", "figures", "shared", "scripts"):
        copy_tree(ROOT / name, target / name)
    stage_domain(stage, "ophthalmology")
    stage_papila(stage, Path(os.environ["FUZZYXAI_CH6_DATA_ROOT"]))
    stage_domain(stage, "ecg_ptbxl")
    stage_domain(stage, "brain_allen")
    stage_domain(stage, "brain_allen", "outputs_v2_confirmatory")
    regression_log = BUNDLES / "ch6_final_full_regression.log"
    if regression_log.is_file():
        copy_tree(regression_log, stage / "final_full_regression.log")
    wheel_log = BUNDLES / "ch6_final_wheel_smoke.log"
    if wheel_log.is_file():
        copy_tree(wheel_log, stage / "wheel_smoke_test.log")
    repo = ROOT.parent
    copy_tree(repo / "pyproject.toml", stage / "pyproject.toml")
    copy_tree(repo / "framework" / "fuzzyxai" / "pyproject.toml", stage / "framework" / "fuzzyxai" / "pyproject.toml")
    copy_tree(repo / "framework" / "fuzzyxai" / "fuzzyxai", stage / "framework" / "fuzzyxai" / "fuzzyxai")
    if (repo / "final_transparency_validation" / "semantic_audit.md").is_file():
        copy_tree(repo / "final_transparency_validation" / "semantic_audit.md", stage / "final_transparency_validation" / "semantic_audit.md")
    wheel_dir = repo / "framework" / "fuzzyxai" / "dist"
    for wheel in sorted(wheel_dir.glob("fuzzyxai_operator-*.whl")):
        copy_tree(wheel, stage / "wheel" / wheel.name)
    for name in (
        "test_p15_full_report.py", "test_p19_system_semantics.py",
        "test_p19_global_consistency.py", "test_p19_package_hygiene.py",
    ):
        if (repo / "tests" / name).is_file():
            copy_tree(repo / "tests" / name, stage / "tests" / name)
    (target / "REPRODUCIBILITY_CH6.json").write_text(json.dumps(reproducibility(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (stage / "README_EXTRACTED_RU.md").write_text(
        "# Проверка архива\n\n"
        "Raw datasets и checkpoints намеренно исключены. Для review source/reports/case artifacts доступны под `chapter6_medical_validation/`. "
        "Для повторного запуска нужны официальные raw data в переменной `FUZZYXAI_CH6_DATA_ROOT`.\n\n"
        "После распаковки проверяйте контрольные суммы из корня архива: `sha256sum -c SHA256SUMS.txt`.\n",
        encoding="utf-8",
    )
    inventory = sorted(path.relative_to(stage).as_posix() for path in stage.rglob("*") if path.is_file())
    (stage / "BUNDLE_CONTENTS.txt").write_text("\n".join(inventory) + "\n", encoding="utf-8")
    sums = [f"{sha256_file(stage / name)}  {name}" for name in inventory]
    (stage / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n", encoding="utf-8")
    archive = BUNDLES / f"{args.bundle_name}.zip"
    if archive.exists(): archive.unlink()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        for path in sorted(stage.rglob("*")):
            if path.is_file(): zip_file.write(path, path.relative_to(BUNDLES))
    print(json.dumps({"archive": str(archive), "sha256": sha256_file(archive), "files": len(inventory)}, ensure_ascii=False))


if __name__ == "__main__": main()
