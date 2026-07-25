#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

PARENT_COMMIT = "1b30d4d05d355df266e1ec80f7c29261cf50e5d3"
PARENT_PATHS = (
    "results/h10_c5",
    "reports/h10_c5",
    "results/h9_e2e",
    "reports/h9_e2e",
)


def _git(arguments: list[str], root: Path, *, text: bool = False) -> bytes | str:
    return subprocess.check_output(["git", *arguments], cwd=root, text=text)


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def run(root: Path) -> dict[str, object]:
    listing = str(
        _git(
            ["ls-tree", "-r", "--name-only", PARENT_COMMIT, "--", *PARENT_PATHS],
            root,
            text=True,
        )
    )
    files = tuple(line for line in listing.splitlines() if line)
    rows = []
    for relative in files:
        expected = bytes(_git(["show", f"{PARENT_COMMIT}:{relative}"], root))
        current_path = root / relative
        current = current_path.read_bytes() if current_path.is_file() else b""
        rows.append(
            {
                "path": relative,
                "expected_sha256": _sha256(expected),
                "current_sha256": _sha256(current) if current_path.is_file() else "",
                "status": "PASS" if current == expected else "FAIL",
            }
        )
    result = {
        "parent_commit": PARENT_COMMIT,
        "checked_paths": list(PARENT_PATHS),
        "file_count": len(rows),
        "changed_files": [row["path"] for row in rows if row["status"] != "PASS"],
        "status": "PASS" if rows and all(row["status"] == "PASS" for row in rows) else "FAIL",
        "files": rows,
    }
    output = root / "reports/h10_c5b/PARENT_RESULT_IMMUTABILITY.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = root / "reports/h10_c5b/PARENT_RESULT_IMMUTABILITY.md"
    markdown.write_text(
        "# Parent Result Immutability\n\n"
        f"- Parent commit: `{PARENT_COMMIT}`\n"
        f"- Checked files: `{len(rows)}`\n"
        f"- Changed files: `{len(result['changed_files'])}`\n"
        f"- Status: `{result['status']}`\n\n"
        "The negative H10-C5 and H9-E2E results remain byte-identical. "
        "H10-C5b and H9-E2E-v2 are separate prospective cycles.\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    result = run(args.root.resolve())
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
