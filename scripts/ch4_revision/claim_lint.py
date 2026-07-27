#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path

FORBIDDEN = (
    "p = 0.0234",
    "p = 0,0234",
    "доказана практическая полезность",
    "ускоряет работу инженера",
    "снижает трудозатраты",
    "подтверждено на реальных инцидентах",
    "H10-C3a и H10-C3b независимо подтверждены",
    "industrial performance proved",
    "practical utility proved",
    "human time reduced",
    "natural incident repair confirmed universally",
    "noise robustness confirmed",
    "доказана промышленная производительность",
    "доказана универсальная практическая полезность",
    "сокращено время работы специалиста",
    "универсально подтверждено восстановление естественных инцидентов",
    "подтверждена устойчивость к шуму",
)


def _text(path: Path) -> str:
    if path.suffix.lower() != ".docx":
        return path.read_text(encoding="utf-8", errors="ignore")
    with zipfile.ZipFile(path) as archive:
        return archive.read("word/document.xml").decode("utf-8", errors="ignore")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("paths", nargs="*", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    paths = args.paths or [
        root / "reports/chapter_revision",
        root / "reports/h10_c5",
        root / "reports/h10_c6",
        root / "reports/h9_e2e",
        root / "reports/h9_e2e_v2",
        root / "reports/h10_c5_pilot",
        root / "reports/h10_c6_noise",
        root / "reports/integrations",
        root / "reports/final_practical",
        root / "reports/chapter_updates",
        root / "reports/multimodal_routes",
    ]
    files: list[Path] = []
    for path in paths:
        path = path.resolve()
        if path.is_file():
            files.append(path)
        elif path.exists():
            files.extend(
                candidate
                for candidate in path.rglob("*")
                if candidate.suffix.lower() in {".md", ".json", ".txt", ".docx"}
            )
    violations: list[dict[str, str]] = []
    for path in files:
        content = _text(path).lower()
        for phrase in FORBIDDEN:
            if phrase.lower() in content:
                violations.append({"path": str(path.relative_to(root)), "phrase": phrase})
        if re.search(r"H10-C3\s+(?:подтверждено|подтверждена)(?![^.\n]{0,100}контролируем)", content, re.IGNORECASE):
            violations.append({"path": str(path.relative_to(root)), "phrase": "unqualified H10-C3 confirmation"})
    report = {"status": "PASS" if not violations else "FAIL", "files_checked": len(files), "violations": violations}
    output = root / "reports/chapter_revision/CLAIM_LINT.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if violations:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
