#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree

FORBIDDEN_CLAIMS = (
    "доказана практическая полезность",
    "ускоряет работу инженера",
    "снижает трудозатраты",
    "подтверждено на реальных инцидентах",
    "улучшает пользовательские решения",
)
PLACEHOLDERS = ("[TBD", "TODO", "FIXME", "NOT_EVALUATED")
H10_CONFIRMATION = re.compile(
    r"(?:H10-C3.{0,180}подтвержд|подтвержд.{0,180}H10-C3)",
    re.IGNORECASE,
)
FIVE_MILLION = re.compile(r"(?:5\s*млн|5[\s\u00a0]*000[\s\u00a0]*000)", re.IGNORECASE)
PRECOMPUTED = "при заранее рассчитанных объяснениях"


def _docx_blocks(path: Path) -> list[str]:
    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    with zipfile.ZipFile(path) as archive:
        root = ElementTree.fromstring(archive.read("word/document.xml"))
    blocks = []
    for paragraph in root.findall(".//w:p", namespace):
        text = "".join(
            node.text or "" for node in paragraph.findall(".//w:t", namespace)
        ).strip()
        if text:
            blocks.append(text)
    return blocks


def _blocks(path: Path) -> list[str]:
    if path.suffix.lower() == ".docx":
        return _docx_blocks(path)
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def lint(path: Path) -> list[dict[str, object]]:
    errors = []
    for index, block in enumerate(_blocks(path), 1):
        lowered = block.casefold()
        for claim in FORBIDDEN_CLAIMS:
            if claim.casefold() in lowered:
                errors.append(
                    {
                        "block": index,
                        "code": "UNSUPPORTED_CLAIM",
                        "text": claim,
                    }
                )
        for marker in PLACEHOLDERS:
            if marker.casefold() in lowered:
                errors.append(
                    {
                        "block": index,
                        "code": "PLACEHOLDER",
                        "text": marker,
                    }
                )
        if H10_CONFIRMATION.search(block) and not (
            "контролируем" in lowered and "мутац" in lowered
        ):
            errors.append(
                {
                    "block": index,
                    "code": "H10_C3_SCOPE_MISSING",
                    "text": block[:240],
                }
            )
        if FIVE_MILLION.search(block) and PRECOMPUTED not in lowered:
            errors.append(
                {
                    "block": index,
                    "code": "PRECOMPUTED_EXPLANATIONS_QUALIFIER_MISSING",
                    "text": block[:240],
                }
            )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Fail-closed claim lint for manuscripts")
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    failed = False
    for path in args.paths:
        errors = lint(path)
        status = "PASS" if not errors else "FAIL"
        print(f"{path}: {status}")
        for error in errors:
            print(
                f"  block {error['block']}: {error['code']}: {error['text']}",
                file=sys.stderr,
            )
        failed = failed or bool(errors)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
