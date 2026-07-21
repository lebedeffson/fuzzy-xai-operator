#!/usr/bin/env python3
"""Render final Chapter 4 only from complete claim-scoped confirmation."""

from __future__ import annotations

import re
import shutil
import subprocess

from common import CONFIRMATORY, FORMATIVE, ROOT, load_json


def main() -> None:
    registry_path = FORMATIVE.parent / "claim_registry.json"
    blockers = []
    if not registry_path.is_file():
        blockers.append("claim registry is missing")
    else:
        registry = load_json(registry_path)
        if registry.get("confirmatory_run_completed") is not True:
            blockers.append("sealed confirmatory run is incomplete")
        if registry.get("technical_release_allowed") is not True:
            blockers.append("claim-scoped technical release gate is closed")
        for claim in registry.get("new_claims", []):
            if claim.get("enabled") is True and claim.get("status") == "not_run":
                blockers.append(f"enabled claim {claim.get('claim_id')} lacks evidence")
    source = CONFIRMATORY / "chapter4_final.md"
    if not source.is_file():
        blockers.append("machine-generated confirmatory chapter source is missing")
    if blockers:
        print("BLOCKED: chapter4-final")
        for blocker in blockers:
            print(f"- {blocker}")
        raise SystemExit(2)
    text = source.read_text(encoding="utf-8")
    if re.search(r"PLACEHOLDER|PENDING|TBD|TODO", text, flags=re.IGNORECASE):
        raise SystemExit("FAIL: final chapter contains placeholder text")
    if not re.search(r"\[evidence:[^]]+ sha256:[0-9a-f]{64}\]", text):
        raise SystemExit("FAIL: final chapter lacks hash-bound evidence references")
    pandoc, libreoffice = shutil.which("pandoc"), shutil.which("libreoffice")
    if not pandoc or not libreoffice:
        raise SystemExit("FAIL: pandoc and libreoffice are required for DOCX/PDF rendering")
    output = ROOT / "dissertation_artifacts/final_practical_closure/chapter4_final"
    output.mkdir(parents=True, exist_ok=True)
    docx = output / "Глава_4_FuzzyXAI_final.docx"
    subprocess.run([pandoc, str(source), "-o", str(docx)], check=True)
    subprocess.run([libreoffice, "--headless", "--convert-to", "pdf", "--outdir", str(output), str(docx)], check=True)
    if not (output / "Глава_4_FuzzyXAI_final.pdf").is_file():
        raise SystemExit("FAIL: PDF rendering did not complete")
    print("PASS: chapter4-final claim_scope=computational_only placeholders=0")


if __name__ == "__main__":
    main()

