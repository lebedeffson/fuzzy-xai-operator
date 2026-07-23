from __future__ import annotations

import ast
import os

from ..config import load_yaml
from ..hashing import file_sha256, write_json
from ..paths import ARTIFACT_ROOT, PACKAGE_ROOT


def audit_baselines() -> dict:
    config = load_yaml("methods.yaml")
    root = PACKAGE_ROOT / "src" / "h10_c2" / "baselines"
    forbidden = tuple(config["forbidden_baseline_symbols"]) + tuple(config["forbidden_gold_paths"])
    violations = []
    hashes: dict[str, str] = {}
    for path in sorted(root.glob("*.py")):
        source = path.read_text(encoding="utf-8")
        ast.parse(source)
        hashes[path.name] = file_sha256(path)
        for token in forbidden:
            if token in source:
                violations.append({"file": path.name, "token": token})
    duplicates = {
        digest: sorted(name for name, value in hashes.items() if value == digest)
        for digest in set(hashes.values())
        if sum(value == digest for value in hashes.values()) > 1
    }
    risky_environment = sorted(key for key in os.environ if any(word in key.upper() for word in ("GOLD", "VAULT", "LABEL")))
    report = {
        "status": "BASELINE_INDEPENDENCE_FAIL" if violations or duplicates else "PASS",
        "violations": violations,
        "duplicate_file_hashes": duplicates,
        "risky_environment_variables_visible": risky_environment,
        "files": hashes,
    }
    write_json(ARTIFACT_ROOT / "audit" / "baseline_independence.json", report)
    if report["status"] != "PASS":
        raise RuntimeError("BASELINE_INDEPENDENCE_FAIL")
    return report

