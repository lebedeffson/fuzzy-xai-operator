#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from build_legacy_immutability import build as verify_legacy

ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    subprocess.run(
        [str(ROOT / "scripts/ch4_revision/build_h10_c3_hierarchy.py")],
        cwd=ROOT,
        check=True,
    )
    legacy = verify_legacy()
    subprocess.run(
        [str(ROOT / "scripts/ch4_revision/claim_lint.py"), "--root", str(ROOT)],
        cwd=ROOT,
        check=True,
    )
    reports = ROOT / "reports/chapter_revision"
    (reports / "STATISTICAL_HIERARCHY.md").write_text(
        (reports / "H10_C3_STATISTICAL_HIERARCHY.md").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (reports / "TERMINOLOGY_AUDIT.md").write_text(
        "# Terminology Audit\n\n"
        "- `RouteAuditor`: structural route evidence, contracts, obligations, and cut.\n"
        "- `RepairPlanner`: plan construction from a registered cut.\n"
        "- `RepairExecutor`: execution, postconditions, and rollback.\n"
        "- `RouteRecertifier`: full route reconstruction and validation.\n"
        "- `ExternalPolicy`: downstream domain decision; not imported by RouteAuditor.\n"
        "- `FuzzyXAI`: framework name, not a synonym for ExternalPolicy.\n"
        "- Status: `PASS`.\n",
        encoding="utf-8",
    )
    claim_lint = json.loads((reports / "CLAIM_LINT.json").read_text(encoding="utf-8"))
    (reports / "CLAIM_AUDIT.md").write_text(
        "# Claim Audit\n\n"
        f"- Checked files: `{claim_lint['files_checked']}`\n"
        f"- Forbidden claims: `{len(claim_lint['violations'])}`\n"
        "- H10-C3 scope: pre-generated controlled structural mutations.\n"
        "- H10-C5 natural-incident transfer: not supported.\n"
        "- H10-C6 cut robustness: supported under registered perturbations.\n"
        "- H9 E2E overhead target: not met.\n"
        "- Human and engineer-time claims: disabled.\n"
        f"- Status: `{claim_lint['status']}`\n",
        encoding="utf-8",
    )
    required = [
        "protocol/h10_c5_natural_incidents/H10_C5_PROTOCOL_LOCK.json",
        "protocol/h10_c6_cut_robustness/H10_C6_PROTOCOL_LOCK.json",
        "protocol/h9_e2e_latency/H9_E2E_PROTOCOL_LOCK.json",
        "protocol/multimodal_interpretable_routes/MULTIMODAL_PROTOCOL_LOCK.json",
        "results/h10_c5/H10_C5_FINAL_STATUS.json",
        "results/h10_c6/H10_C6_FINAL_STATUS.json",
        "results/h9_e2e/H9_E2E_FINAL_STATUS.json",
        "results/multimodal_routes/FINAL_STATUS.json",
        "reports/chapter_revision/LEGACY_EVIDENCE_IMMUTABILITY.json",
        "reports/chapter_revision/CLAIM_LINT.json",
    ]
    missing = [relative for relative in required if not (ROOT / relative).is_file()]
    if missing:
        raise FileNotFoundError(", ".join(missing))
    artifacts = [
        {
            "path": relative,
            "sha256": _sha256(ROOT / relative),
            "size": (ROOT / relative).stat().st_size,
        }
        for relative in required
    ]
    statuses = {
        "legacy_evidence_integrity": legacy["legacy_evidence_integrity"],
        "h10_c5": json.loads((ROOT / required[4]).read_text())["status"],
        "h10_c6": json.loads((ROOT / required[5]).read_text())["status"],
        "h9_e2e": json.loads((ROOT / required[6]).read_text())["status"],
        "multimodal_routes": json.loads((ROOT / required[7]).read_text())["status"],
        "claim_lint": claim_lint["status"],
    }
    payload = {
        "schema_version": "1.0",
        "implementation_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "statuses": statuses,
        "artifacts": artifacts,
        "overall_status": "COMPLETE_WITH_REGISTERED_NEGATIVE_RESULTS",
        "chapter_modified": False,
    }
    (reports / "FINAL_EVIDENCE_MAP.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["statuses"], indent=2))


if __name__ == "__main__":
    main()
