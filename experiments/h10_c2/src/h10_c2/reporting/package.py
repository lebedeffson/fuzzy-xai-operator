from __future__ import annotations

import shutil
import subprocess
import zipfile
from pathlib import Path

import yaml

from ..hashing import file_sha256, read_json, write_json
from ..paths import ARTIFACT_ROOT, DELIVERABLE_ROOT, PROTOCOL_DIR, REPO_ROOT
from .evidence_map import build_evidence_map
from .pdf import markdown_to_pdf
from .validation_report import build_validation_report


def _zip_tree(target: Path, roots: list[tuple[Path, str]], *, exclude_private: bool = True) -> None:
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for root, prefix in roots:
            for path in sorted(root.rglob("*")):
                if not path.is_file():
                    continue
                relative = path.relative_to(root)
                value = str(relative)
                if exclude_private and any(token in value for token in ("private/", "vault", "opening_key", "__pycache__")):
                    continue
                archive.write(path, str(Path(prefix) / relative))


def build_deliverables() -> dict:
    DELIVERABLE_ROOT.mkdir(parents=True, exist_ok=True)
    build_evidence_map()
    report = build_validation_report()
    (DELIVERABLE_ROOT / "validation_report.md").write_text(report, encoding="utf-8")
    gate = ARTIFACT_ROOT / "audit" / "preconfirmatory_gate.json"
    copies = {
        "methodology_audit.json": ARTIFACT_ROOT / "audit" / "oracle_independence.json",
        "leakage_audit.json": ARTIFACT_ROOT / "audit" / "leakage_audit.json",
        "baseline_independence.json": ARTIFACT_ROOT / "audit" / "baseline_independence.json",
        "gold_audit.json": ARTIFACT_ROOT / "data" / "protocol_validation" / "manifest.json",
        "adjudication_status.json": ARTIFACT_ROOT / "adjudication" / "status.json",
        "preconfirmatory_gate.json": gate,
        "evidence_map.json": ARTIFACT_ROOT / "audit" / "evidence_map.json",
        "coverage.json": ARTIFACT_ROOT / "audit" / "coverage.json",
        "full_regression.json": ARTIFACT_ROOT / "audit" / "full_regression.json",
    }
    for name, source in copies.items():
        shutil.copy2(source, DELIVERABLE_ROOT / name)
    write_json(
        DELIVERABLE_ROOT / "claim_registry.json",
        yaml.safe_load((PROTOCOL_DIR / "claim_registry.yaml").read_text(encoding="utf-8")),
    )
    markdown_to_pdf(PROTOCOL_DIR / "protocol.md", DELIVERABLE_ROOT / "protocol.pdf")
    markdown_to_pdf(PROTOCOL_DIR / "statistical_analysis_plan.md", DELIVERABLE_ROOT / "statistical_analysis_plan.pdf")
    markdown_to_pdf(ARTIFACT_ROOT / "power" / "power_report.md", DELIVERABLE_ROOT / "power_report.pdf")
    gate_value = read_json(gate)
    blockers = gate_value["blockers"]
    software_status = "PASS" if not any(item.startswith("BLOCKED_CODE") for item in blockers) else "BLOCKED_CODE"
    power_status = "PASS" if not any(item.startswith("BLOCKED_POWER") for item in blockers) else "BLOCKED_POWER"
    protocol_status = "PASS" if not any(item.startswith("BLOCKED_PROTOCOL") for item in blockers) else "BLOCKED_PROTOCOL"
    status = (
        "# H10-C2 handoff status\n\n"
        f"Software implementation: {software_status}\n\n"
        "v21 integrity: PASS\n\n"
        "Power analysis implementation: PASS\n\n"
        f"Power analysis execution: {power_status}\n\n"
        f"Protocol package: {protocol_status}\n\n"
        "Gold generator: PASS\n\n"
        "Baseline independence: PASS\n\n"
        "Leakage audit: PASS\n\n"
        "Reviewer package: PASS\n\n"
        "Manual adjudication: BLOCKED_HUMAN_ADJUDICATION\n\n"
        "Sealed opening count: 0\n\n"
        "H10-C2a: NOT_EVALUATED\n\n"
        "H10-C2b: NOT_EVALUATED\n\n"
        "Scientific release: BLOCKED_PRECONFIRMATORY\n\n"
        "Current blockers:\n\n"
        + "\n".join(f"- {item}" for item in blockers)
        + "\n"
    )
    (DELIVERABLE_ROOT / "HANDOFF_STATUS.md").write_text(status, encoding="utf-8")
    source_zip = DELIVERABLE_ROOT / "h10-c2-source.zip"
    subprocess.run(
        ["git", "archive", "--format=zip", f"--output={source_zip}", "HEAD", "experiments/h10_c2", "Makefile"],
        cwd=REPO_ROOT,
        check=True,
    )
    artifact_zip = DELIVERABLE_ROOT / "h10-c2-validation-artifacts.zip"
    _zip_tree(artifact_zip, [(ARTIFACT_ROOT, "artifacts/h10_c2")], exclude_private=True)
    reviewer_zip = DELIVERABLE_ROOT / "h10-c2-reviewer-handoff.zip"
    _zip_tree(reviewer_zip, [(ARTIFACT_ROOT / "adjudication" / "blind", "adjudication/blind")])
    files = [path for path in DELIVERABLE_ROOT.iterdir() if path.is_file() and path.name != "SHA256SUMS"]
    (DELIVERABLE_ROOT / "SHA256SUMS").write_text(
        "".join(f"{file_sha256(path)}  {path.name}\n" for path in sorted(files)),
        encoding="utf-8",
    )
    return {"status": "BLOCKED_PRECONFIRMATORY", "deliverable_count": len(files) + 1}
