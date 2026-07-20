#!/usr/bin/env python3
"""Fail closed until independent confirmation and external gates are complete."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "study/strong_confirmatory"
CONFIRMATORY = ROOT / "release_evidence/strong_confirmatory/confirmatory"


def main() -> None:
    blockers = []
    lock = STUDY / "confirmatory_protocol_lock.json"
    registry = CONFIRMATORY / "claim_registry.json"
    external = CONFIRMATORY / "external_gates.json"
    for path in (lock, registry, external):
        if not path.is_file():
            blockers.append(f"missing {path.relative_to(ROOT)}")
    if not blockers:
        claims = json.loads(registry.read_text(encoding="utf-8"))
        gates = json.loads(external.read_text(encoding="utf-8"))
        if claims.get("confirmatory_run_completed") is not True:
            blockers.append("confirmatory run is not complete")
        if gates.get("domain_language") != "pass":
            blockers.append("domain-language gate is open")
        if gates.get("comprehension") != "pass":
            blockers.append("comprehension gate is open")
        if gates.get("expert_action") != "pass":
            blockers.append("expert-action gate is open")
    if blockers:
        print("BLOCKED: chapter4-final")
        for blocker in blockers:
            print(f"- {blocker}")
        raise SystemExit(2)
    source = CONFIRMATORY / "chapter4_final.md"
    if not source.is_file():
        raise SystemExit(f"FAIL: missing {source.relative_to(ROOT)}")
    text = source.read_text(encoding="utf-8")
    if "PENDING" in text or "PLACEHOLDER" in text:
        raise SystemExit("FAIL: final chapter contains a placeholder")
    numeric_claims = re.findall(r"\[evidence:[A-Za-z0-9_.:-]+\]", text)
    if not numeric_claims:
        raise SystemExit("FAIL: final chapter has no evidence references")
    pandoc = shutil.which("pandoc")
    libreoffice = shutil.which("libreoffice")
    if not pandoc or not libreoffice:
        raise SystemExit("FAIL: pandoc and libreoffice are required for final DOCX/PDF rendering")
    output = ROOT / "dissertation_artifacts/strong_confirmatory/chapter4_final"
    output.mkdir(parents=True, exist_ok=True)
    docx = output / "Глава_4_FuzzyXAI_final.docx"
    subprocess.run([pandoc, str(source), "-o", str(docx)], check=True)
    subprocess.run([libreoffice, "--headless", "--convert-to", "pdf", "--outdir", str(output), str(docx)], check=True)
    pdf = output / "Глава_4_FuzzyXAI_final.pdf"
    if not docx.is_file() or not pdf.is_file():
        raise SystemExit("FAIL: final chapter rendering did not produce DOCX and PDF")
    print(f"PASS: chapter4-final evidence_refs={len(numeric_claims)}")


if __name__ == "__main__":
    main()
