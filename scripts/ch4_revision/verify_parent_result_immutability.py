#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path

MANIFESTS = {
    "H10-C3": (
        "protocol/h10_c4/H10_C3_BASELINE_SHA256SUMS",
        "repository",
    ),
    "H10-C4": ("results/h10_c4/SHA256SUMS", "repository"),
    "H10-C5": ("results/h10_c5/SHA256SUMS", "repository"),
    "H10-C5b": ("results/h10_c5b/SHA256SUMS", "repository"),
    "H10-C6": ("results/h10_c6/SHA256SUMS", "repository"),
    "H9-E2E": ("results/h9_e2e/SHA256SUMS", "repository"),
    "H9-E2E-v2": ("results/h9_e2e_v2/SHA256SUMS", "repository"),
    "multimodal routes": (
        "results/multimodal_routes/SHA256SUMS",
        "repository",
    ),
}
DIRECT = {
    "H10-C5c development status": (
        "results/h10_c5c/H10_C5C_DEVELOPMENT_STATUS.json",
        "60810d8ee58b20c6d9b1a5e4800c23fc678ce9e698ad70d339e91d37a7f69ba1",
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    root = Path.cwd()
    checks: list[dict[str, object]] = []
    for result, (manifest_name, mode) in MANIFESTS.items():
        manifest = root / manifest_name
        manifest_parent = manifest.parent
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            expected, name = line.split(maxsplit=1)
            relative = name.lstrip("*")
            path = (
                manifest_parent / relative
                if mode == "manifest_parent"
                else root / relative
            )
            actual = _sha256(path) if path.is_file() else None
            checks.append(
                {
                    "result": result,
                    "path": str(path.relative_to(root)),
                    "expected_sha256": expected,
                    "actual_sha256": actual,
                    "pass": actual == expected,
                }
            )
    for result, (name, expected) in DIRECT.items():
        path = root / name
        actual = _sha256(path) if path.is_file() else None
        checks.append(
            {
                "result": result,
                "path": name,
                "expected_sha256": expected,
                "actual_sha256": actual,
                "pass": actual == expected,
            }
        )
    failed = [check for check in checks if not check["pass"]]
    report = {
        "status": "PASS" if not failed else "FAIL",
        "files_checked": len(checks),
        "results_checked": list(MANIFESTS) + list(DIRECT),
        "checks": checks,
    }
    output = root / "reports/final_practical"
    output.mkdir(parents=True, exist_ok=True)
    (output / "PARENT_RESULT_IMMUTABILITY.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Parent Result Immutability",
        "",
        f"- Status: `{report['status']}`",
        f"- Files verified: `{report['files_checked']}`",
        "- Verification: published SHA256 manifests plus the locked H10-C5c status digest.",
        "",
        "| Result | Checked files | Status |",
        "|---|---:|---|",
    ]
    for name in report["results_checked"]:
        rows = [row for row in checks if row["result"] == name]
        status = "PASS" if all(row["pass"] for row in rows) else "FAIL"
        lines.append(f"| {name} | {len(rows)} | `{status}` |")
    lines.extend(
        [
            "",
            "No parent experiment was recalculated by this audit.",
            "",
        ]
    )
    (output / "PARENT_RESULT_IMMUTABILITY.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    print(json.dumps({key: report[key] for key in ("status", "files_checked")}))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
