from __future__ import annotations

import argparse
import hashlib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_PARTS = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "raw", "checkpoints"}
EXCLUDED_SUFFIXES = {".pyc", ".pth", ".pt", ".ckpt"}


def bundle_files() -> list[Path]:
    files = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.name == "CH6_OPHTHALMOLOGY_VALIDATION.zip":
            continue
        relative = path.relative_to(ROOT)
        if any(part in EXCLUDED_PARTS for part in relative.parts) or path.suffix.lower() in EXCLUDED_SUFFIXES:
            continue
        files.append(path)
    return sorted(files, key=lambda value: value.relative_to(ROOT).as_posix())


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the review bundle without raw data/checkpoints")
    parser.add_argument("--output", type=Path, default=ROOT / "CH6_OPHTHALMOLOGY_VALIDATION.zip")
    args = parser.parse_args()
    files = bundle_files()
    checksums = []
    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        checksums.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {relative}")
    checksum_payload = "\n".join(checksums) + "\n"
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(ROOT.parent).as_posix())
        archive.writestr(f"{ROOT.name}/SHA256SUMS.txt", checksum_payload)
    print(f"{args.output}: {len(files) + 1} files")


if __name__ == "__main__":
    main()
