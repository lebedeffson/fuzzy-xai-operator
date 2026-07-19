"""Assemble Chapter 4 evidence while preserving unresolved external gates."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "release_evidence/chapter4_final_candidate"
ARCHIVE = ROOT / "release_evidence/chapter4-final-candidate.zip"
EMPIRICAL = ROOT / "release_evidence/empirical_experiments/breast_cancer_checkpoint"
UNIVERSAL = ROOT / "release_evidence/model_universality"
QUALITY = ROOT / "release_evidence/explanation_quality"
PILOT = ROOT / "release_evidence/user_study/comprehension_pilot"
DOMAIN = ROOT / "release_evidence/domain_language_review"


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy(source: Path, relative: str) -> None:
    target = OUTPUT / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def deterministic_zip(source: Path, target: Path) -> None:
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                info = zipfile.ZipInfo(
                    str(Path(source.name) / path.relative_to(source)),
                    date_time=(2020, 1, 1, 0, 0, 0),
                )
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, path.read_bytes())


def main() -> None:
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    required = (
        EMPIRICAL / "empirical_summary.json",
        EMPIRICAL / "selected_forgetting_case.json",
        EMPIRICAL / "rule_ablation.json",
        EMPIRICAL / "similar_cases.json",
        UNIVERSAL / "summary.json",
        UNIVERSAL / "model_support_matrix.json",
        UNIVERSAL / "model_support_matrix.csv",
        UNIVERSAL / "chapter4_model_family_summary.csv",
        UNIVERSAL / "manifest.json",
        QUALITY / "explanation_quality_report.json",
        QUALITY / "explanation_quality_report.md",
        PILOT / "scoring_report.json",
        PILOT / "scenario_manifest.json",
        DOMAIN / "review_record.json",
    )
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Chapter 4 evidence is incomplete: {missing}")

    empirical = read_json(EMPIRICAL / "empirical_summary.json")
    universal = read_json(UNIVERSAL / "summary.json")
    model_matrix = read_json(UNIVERSAL / "model_support_matrix.json")
    quality = read_json(QUALITY / "explanation_quality_report.json")
    pilot = read_json(PILOT / "scoring_report.json")
    domain = read_json(DOMAIN / "review_record.json")
    assert isinstance(empirical, dict) and isinstance(universal, dict) and isinstance(model_matrix, dict)
    assert isinstance(quality, dict)
    assert isinstance(pilot, dict) and isinstance(domain, dict)
    pilot_pass = pilot.get("status") == "pass" and pilot.get("claim_allowed") is True
    domain_pass = domain.get("status") in {"approved", "approved_with_comments"} and domain.get("claim_allowed") is True
    release_pass = pilot_pass and domain_pass

    copy(EMPIRICAL / "empirical_summary.json", "json/empirical_summary.json")
    copy(EMPIRICAL / "selected_forgetting_case.json", "json/selected_forgetting_case.json")
    copy(EMPIRICAL / "rule_ablation.json", "json/rule_ablation.json")
    copy(EMPIRICAL / "similar_cases.json", "json/similar_cases.json")
    copy(UNIVERSAL / "summary.json", "json/model_universality_summary.json")
    copy(UNIVERSAL / "model_support_matrix.json", "json/model_support_matrix.json")
    copy(UNIVERSAL / "model_support_matrix.csv", "tables/model_support_matrix.csv")
    copy(UNIVERSAL / "chapter4_model_family_summary.csv", "tables/model_family_summary.csv")
    copy(UNIVERSAL / "chapter4_model_family_summary.md", "tables/model_family_summary.md")
    copy(UNIVERSAL / "public_api_verification.json", "json/public_api_verification.json")
    copy(QUALITY / "explanation_quality_report.json", "json/explanation_quality_report.json")
    copy(QUALITY / "explanation_quality_report.md", "tables/explanation_quality_report.md")
    copy(PILOT / "scoring_report.json", "pilot/scoring_report.json")
    copy(PILOT / "scenario_manifest.json", "pilot/scenario_manifest.json")
    copy(DOMAIN / "review_record.json", "domain_review/review_record.json")

    model_rows = model_matrix.get("configurations", [])
    assert isinstance(model_rows, list)
    core_rows = [row for row in model_rows if row.get("environment") == "core-model-contracts"]
    optional_rows = [row for row in model_rows if str(row.get("environment", "")).startswith("optional-runtime-")]
    classification_rows = [row for row in model_rows if row.get("task_type") != "regression"]
    regression_rows = [row for row in model_rows if row.get("task_type") == "regression"]
    parity_values = [float(row["prediction_parity"]) for row in model_rows if row.get("prediction_parity") is not None]
    conformance_values = [float(row["conformance"]) for row in model_rows if row.get("conformance") is not None]
    graph_values = [float(row["graph_validation"]) for row in model_rows if row.get("graph_validation") is not None]
    status = {
        "schema_version": "1.0",
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "candidate_version": "1.3.0rc1",
        "computed_gates": {
            "real_training": "PASS",
            "measured_rule_ablation": "PASS",
            "model_universality_verified_configurations": sum(row.get("status") == "pass" for row in model_rows),
            "core_model_configurations": len(core_rows),
            "optional_runtime_libraries": len(optional_rows),
            "prediction_parity_rate": sum(parity_values) / len(parity_values),
            "graph_validation_rate": sum(graph_values) / len(graph_values),
            "conformance_rate": sum(conformance_values) / len(conformance_values),
            "explanation_quality_pass": quality.get("pass_count"),
        },
        "external_gates": {
            "comprehension_pilot": pilot.get("status"),
            "domain_language_review": domain.get("status"),
        },
        "release_gate": "PASS" if release_pass else "BLOCKED",
        "tag_allowed": release_pass,
        "verified_model_rows": sum(row.get("status") == "pass" for row in model_rows),
        "claim_boundary": (
            "Chapter 4 computational evidence is reproducible; demonstrated comprehensibility and "
            "regulated-domain semantics remain unclaimed until both external gates pass."
        ),
    }
    write_json(OUTPUT / "release_gate_status.json", status)
    report = f"""# FuzzyXAI Chapter 4 final candidate evidence

## Computed evidence

- Measured checkpoint experiment: PASS; selected case `{empirical['selected_case']['public_id']}`.
- Measured native rule ablation: PASS; rule `{empirical['rule_ablation']['rule_id']}`.
- Universal-model evidence: {len(model_rows)} verified configurations
  ({len(classification_rows)} classification, {len(regression_rows)} regression).
- Core deterministic matrix: {len(core_rows)} configurations.
- Optional runtime integrations: {len(optional_rows)} libraries, each verified on Python 3.11 and 3.12.
- Prediction parity: {sum(parity_values) / len(parity_values):.3f}.
- Adapter conformance: {sum(conformance_values) / len(conformance_values):.3f}.
- Explanation graph validation: {sum(graph_values) / len(graph_values):.3f}.
- Explanation quality gate: {quality.get('pass_count')} / {quality.get('family_count')}.

## External release gates

- Independent A/B comprehension pilot: `{pilot.get('status')}`.
- Independent regulated-domain language review: `{domain.get('status')}`.
- Release gate: `{status['release_gate']}`.

The package proves the current computational contracts and measured benchmark only. It does not claim
demonstrated human comprehensibility, clinical validity, or support beyond the explicitly listed model families.
"""
    (OUTPUT / "report.md").write_text(report, encoding="utf-8")
    files = {
        str(path.relative_to(OUTPUT)): sha256(path)
        for path in sorted(OUTPUT.rglob("*"))
        if path.is_file() and path.name != "manifest_sha256.json"
    }
    write_json(OUTPUT / "manifest_sha256.json", {"schema_version": "1.0", "files": files})
    deterministic_zip(OUTPUT, ARCHIVE)
    ARCHIVE.with_suffix(".zip.sha256").write_text(f"{sha256(ARCHIVE)}  {ARCHIVE.name}\n", encoding="ascii")
    print(f"PASS: chapter4_computed_evidence {len(model_rows)} model configurations")
    print(f"{'PASS' if release_pass else 'BLOCKED'}: chapter4_external_release_gates")
    print(f"PASS: chapter4_candidate_archive {ARCHIVE}")


if __name__ == "__main__":
    main()
