from __future__ import annotations

import json
import zipfile

from .common import ARTIFACTS, ROOT, git_commit, read_json, sha256_file


def main() -> None:
    evidence = read_json(ARTIFACTS / "closure" / "evidence_map.json")
    paths = [ROOT / item["path"] for item in evidence["records"]]
    paths.extend((ARTIFACTS / "closure" / "evidence_map.json", ARTIFACTS / "closure" / "validation_report.md"))
    paths.extend(sorted((ROOT / "framework" / "fuzzyxai" / "fuzzyxai" / "operational_audit").glob("*.py")))
    paths.extend(sorted((ROOT / "experiments" / "operational_audit_v16").glob("*.py")))
    output = ROOT / "release_artifacts" / f"fuzzyxai-operational-audit-v16-{git_commit()[:12]}.zip"
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(paths)):
            archive.write(path, path.relative_to(ROOT))
        archive.writestr("BUNDLE_MANIFEST.json", json.dumps({"bundle_commit": git_commit(), "evidence_generation_commit": evidence["evidence_generation_commit"], "closure_packaging_commit": evidence["closure_packaging_commit"], "stable_release": False}, indent=2))
    print(f"PASS operational-audit-zip path={output} sha256={sha256_file(output)}")


if __name__ == "__main__":
    main()
