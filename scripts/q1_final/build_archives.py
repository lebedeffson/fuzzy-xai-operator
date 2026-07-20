#!/usr/bin/env python3
"""Build final evidence archives from explicit runtime and committed allowlists."""

from __future__ import annotations

import hashlib
import json
import subprocess
import zipfile
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "release_artifacts/q1_final"
RUNTIME_ARCHIVES: dict[str, tuple[str, ...]] = {
    "fuzzyxai-q1-final-evidence": ("release_evidence/q1_final/", "reports/q1_final/"),
    "fuzzyxai-q1-final-dissertation-artifacts": (
        "dissertation_artifacts/q1_final/",
        "release_evidence/q1_final/claim_registry",
        "release_evidence/q1_final/final_gate_matrix.json",
    ),
    "fuzzyxai-q1-final-external-studies": ("study/q1_final/", "release_evidence/q1_final/external/"),
    "fuzzyxai-q1-final-reviewer-response": (
        "reports/q1_final/",
        "release_evidence/q1_final/claim_registry",
        "release_evidence/q1_final/final_gate_matrix.json",
        "release_evidence/q1_final/dod_185.json",
        "research/q1_final/",
    ),
}
COMMITTED_ARCHIVES: dict[str, tuple[str, ...]] = {
    "fuzzyxai-q1-final-reproducibility": (
        "framework/fuzzyxai/fuzzyxai/q1_final/",
        "scripts/q1_final/",
        "tests/q1_final/",
        "configs/q1_final/",
        "research/q1_final/",
        "study/q1_final/",
        ".github/workflows/q1-final-validation.yml",
        ".github/workflows/q1-external-validation.yml",
        ".github/workflows/q1-stable-release.yml",
        "Dockerfile.q1-final",
        "docker-compose.q1-final.yml",
        "requirements.lock",
        "uv.lock",
        "Makefile",
    ),
}


def git(*args: str, binary: bool = False) -> str | bytes:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=not binary)


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def runtime_paths(prefixes: Sequence[str]) -> list[Path]:
    paths = []
    for prefix in prefixes:
        candidate = ROOT / prefix
        if candidate.is_file():
            paths.append(candidate)
        elif candidate.is_dir():
            paths.extend(path for path in candidate.rglob("*") if path.is_file())
        else:
            paths.extend(path for path in ROOT.glob(f"{prefix}*") if path.is_file())
    return sorted(set(paths))


def committed_paths(commit: str, prefixes: Sequence[str]) -> list[str]:
    names = str(git("ls-tree", "-r", "--name-only", commit)).splitlines()
    return sorted(path for path in names if any(path == prefix or path.startswith(prefix) for prefix in prefixes))


def write_archive(
    name: str,
    commit: str,
    branch: str,
    files: Sequence[tuple[str, bytes]],
    identity: dict[str, object],
) -> dict[str, object]:
    if not files:
        raise RuntimeError(f"archive {name} has no allowlisted inputs")
    hypotheses = _hypothesis_statuses()
    environment = {
        "schema_version": "1.0",
        "python": identity["python"],
        "platform": identity["platform"],
        "threads": identity["threads"],
        "profile": identity["profile"],
    }
    limitations = {
        "schema_version": "1.0",
        "items": [
            "Benchmark evidence does not establish deployment validity.",
            "External human and domain claims remain unavailable while their gates are open.",
            "A null or inconclusive hypothesis is retained and is not rewritten as support.",
            "Raw public datasets are cache-only and are not redistributed.",
        ],
    }
    licenses = (ROOT / "data_manifests/q1_final_datasets.json").read_bytes()
    support = (
        ("_environment_manifest.json", _json_bytes(environment)),
        ("_known_limitations.json", _json_bytes(limitations)),
        ("_hypothesis_statuses.json", _json_bytes(hypotheses)),
        ("_data_license_manifest.json", licenses),
    )
    files = tuple(files) + support
    entries = [{"path": path, "bytes": len(content), "sha256": sha256(content)} for path, content in files]
    manifest = {
        "schema_version": "2.0",
        "archive_type": name,
        "base_commit": identity["base_commit"],
        "final_commit": commit,
        "branch": branch,
        "ci_run_ids": identity["ci_run_ids"],
        "file_count": len(entries),
        "files": entries,
        "hypothesis_statuses": hypotheses,
        "external_gate_status": identity["external_gate_status"],
        "stable_release_allowed": identity["stable_release_allowed"],
    }
    path = OUTPUT / f"{name}-{commit[:12]}.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for relative, content in files:
            _write_zip_entry(archive, f"fuzzy-xai-operator/{relative}", content)
        _write_zip_entry(
            archive,
            "fuzzy-xai-operator/run_identity.json",
            _json_bytes(identity),
        )
        _write_zip_entry(
            archive,
            "fuzzy-xai-operator/_archive_manifest.json",
            _json_bytes(manifest),
        )
    digest = sha256(path.read_bytes())
    path.with_suffix(".zip.sha256").write_text(f"{digest}  {path.name}\n", encoding="ascii")
    return {"archive": path.name, "sha256": digest, "file_count": len(entries), "final_commit": commit}


def _write_zip_entry(archive: zipfile.ZipFile, name: str, content: bytes) -> None:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.external_attr = 0o100644 << 16
    archive.writestr(info, content, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def _json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _hypothesis_statuses() -> dict[str, str]:
    final_path = ROOT / "release_evidence/q1_final/hypotheses/final_results.json"
    h6_path = ROOT / "release_evidence/q1_final/rule_ablation/final_claim_status.json"
    final = json.loads(final_path.read_text(encoding="utf-8")) if final_path.is_file() else {}
    h6 = json.loads(h6_path.read_text(encoding="utf-8")) if h6_path.is_file() else {}
    return {
        "H1": str(final.get("H1_real", {}).get("status", "not_run")),
        "H2": str(final.get("H2_real", {}).get("status", "not_run")),
        "H3_full": str(final.get("H3_real", {}).get("full_population_status", "not_run")),
        "H3_hard": str(final.get("H3_real", {}).get("hard_case_status", "not_run")),
        "H4": str(final.get("H4_real", {}).get("status", "not_run")),
        "H5_structural": str(final.get("H5_real", {}).get("structural", {}).get("status", "not_run")),
        "H5_predictive": str(final.get("H5_real", {}).get("predictive", {}).get("status", "not_run")),
        "H6": str(h6.get("status", "not_run")),
    }


def main() -> None:
    identity_path = ROOT / "release_evidence/q1_final/run_identity.json"
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    commit = str(identity["final_commit"])
    branch = str(identity["branch"])
    if str(git("rev-parse", "HEAD")).strip() != commit:
        raise RuntimeError("run identity final_commit differs from HEAD")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    records = []
    for name, prefixes in RUNTIME_ARCHIVES.items():
        files = [(path.relative_to(ROOT).as_posix(), path.read_bytes()) for path in runtime_paths(prefixes)]
        records.append(write_archive(name, commit, branch, files, identity))
    for name, prefixes in COMMITTED_ARCHIVES.items():
        files = [(path, bytes(git("show", f"{commit}:{path}", binary=True))) for path in committed_paths(commit, prefixes)]
        records.append(write_archive(name, commit, branch, files, identity))
    verification = {
        "schema_version": "2.0",
        "final_commit": commit,
        "branch": branch,
        "archives": records,
        "source_archive_built_by": "python scripts/build_framework_release.py",
        "stable_release_allowed": identity["stable_release_allowed"],
    }
    (OUTPUT / "archive_index.json").write_text(
        json.dumps(verification, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"PASS: q1_final_archives count={len(records)} commit={commit}")


if __name__ == "__main__":
    main()
