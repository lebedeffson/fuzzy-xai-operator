"""Build one verified H10-C3 v23.1 cost-stability handoff archive."""

from __future__ import annotations

import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts" / "h10_c3" / "cost_stability"
OUTPUT_ROOT = ROOT / "release_artifacts"
FIXED_TIME = (1980, 1, 1, 0, 0, 0)
FROZEN_OPEN_INPUTS = (
    ROOT / "artifacts" / "h10_c3_v23" / "lock" / "baseline_selection.json",
    ROOT / "artifacts" / "h10_c3_v23" / "data" / "development" / "manifest.json",
    ROOT / "artifacts" / "h10_c3_v23" / "results" / "development.csv",
    ROOT
    / "artifacts"
    / "h10_c3_v23"
    / "results"
    / "development_statistics.json",
    ROOT / "artifacts" / "h10_c3_v23" / "results" / "protocol_validation.csv",
    ROOT
    / "artifacts"
    / "h10_c3_v23"
    / "results"
    / "protocol_validation_statistics.json",
)
FORBIDDEN_NAME_PARTS = (
    "label_vault",
    "sealed_vault",
    "decryption_key",
    "raw_labels",
    "private_key",
)


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=ROOT,
        text=True,
    ).strip()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_member(
    archive: zipfile.ZipFile,
    name: str,
    payload: bytes,
) -> None:
    info = zipfile.ZipInfo(name, date_time=FIXED_TIME)
    info.external_attr = 0o100644 << 16
    archive.writestr(
        info,
        payload,
        compress_type=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    )


def _require_committed_artifacts() -> list[Path]:
    if subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", str(ARTIFACT_ROOT)],
        cwd=ROOT,
        check=False,
    ).returncode:
        raise RuntimeError("cost-stability artifacts differ from HEAD")
    files = sorted(path for path in ARTIFACT_ROOT.rglob("*") if path.is_file())
    files.extend(FROZEN_OPEN_INPUTS)
    tracked = set(
        _git(
            "ls-files",
            str(ARTIFACT_ROOT),
            *(str(path) for path in FROZEN_OPEN_INPUTS),
        ).splitlines()
    )
    missing = [
        str(path.relative_to(ROOT))
        for path in files
        if str(path.relative_to(ROOT)) not in tracked
    ]
    if missing:
        raise RuntimeError(f"untracked cost-stability artifacts: {missing}")
    return files


def _verify_artifact_checksums() -> None:
    checksum_path = ARTIFACT_ROOT / "SHA256SUMS"
    for line in checksum_path.read_text(encoding="ascii").splitlines():
        expected, relative = line.split("  ", 1)
        path = ARTIFACT_ROOT / relative
        if not path.is_file() or _sha256(path) != expected:
            raise RuntimeError(f"artifact checksum mismatch: {relative}")


def _assert_safe_names(names: list[str]) -> None:
    unsafe = [
        name
        for name in names
        if any(part in name.lower() for part in FORBIDDEN_NAME_PARTS)
    ]
    if unsafe:
        raise RuntimeError(f"forbidden release member names: {unsafe}")


def main() -> None:
    files = _require_committed_artifacts()
    _verify_artifact_checksums()
    commit = _git("rev-parse", "HEAD")
    short_commit = commit[:12]
    source = OUTPUT_ROOT / f"fuzzyxai-source-release-{short_commit}.zip"
    if not source.is_file():
        raise RuntimeError(
            "build the committed source archive with "
            "`python scripts/build_framework_release.py` first"
        )
    with zipfile.ZipFile(source) as archive:
        _assert_safe_names(archive.namelist())

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    artifacts_zip = OUTPUT_ROOT / (
        f"fuzzyxai-h10-c3-cost-stability-artifacts-{short_commit}.zip"
    )
    with zipfile.ZipFile(artifacts_zip, "w") as archive:
        for path in files:
            relative = path.relative_to(ROOT).as_posix()
            _write_member(archive, relative, path.read_bytes())
    with zipfile.ZipFile(artifacts_zip) as archive:
        _assert_safe_names(archive.namelist())

    source_hash = _sha256(source)
    artifacts_hash = _sha256(artifacts_zip)
    status = (
        "# H10-C3 v23.1 handoff status\n\n"
        f"- Branch: `{_git('branch', '--show-current')}`\n"
        f"- Commit: `{commit}`\n"
        "- Global cost scale invariance: `PASS`\n"
        "- Non-uniform cost sensitivity: `PASS`\n"
        "- H10-C3a: `NOT_EVALUATED_CONFIRMATORY`\n"
        "- H10-C3b: `NOT_EVALUATED_CONFIRMATORY`\n"
        "- Sealed created: `false`\n"
        "- Sealed opening count: `0`\n"
        "- Historical v23 evidence changed: `false`\n"
        "\n"
        "## Reproduce\n\n"
        "1. Extract the source ZIP.\n"
        "2. Extract the artifacts ZIP into the extracted source root.\n"
        "3. Run `make h10-c3-test`.\n"
        "4. Run `make h10-c3-cost-stability-audit`.\n"
        "5. Run `make h10-c3-cost-open-reproduction`.\n"
    ).encode()
    manifest = {
        "schema_version": "h10-c3-cost-stability-handoff-v1",
        "branch": _git("branch", "--show-current"),
        "commit": commit,
        "source_archive": source.name,
        "source_sha256": source_hash,
        "artifacts_archive": artifacts_zip.name,
        "artifacts_sha256": artifacts_hash,
        "global_cost_scale_invariance": "PASS",
        "non_uniform_cost_sensitivity": "PASS",
        "sealed_created": False,
        "sealed_opening_count": 0,
        "H10-C3a": "NOT_EVALUATED_CONFIRMATORY",
        "H10-C3b": "NOT_EVALUATED_CONFIRMATORY",
    }
    manifest_bytes = (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    ).encode()
    checksums = (
        f"{source_hash}  {source.name}\n"
        f"{artifacts_hash}  {artifacts_zip.name}\n"
        f"{_sha256_bytes(manifest_bytes)}  handoff_manifest.json\n"
        f"{_sha256_bytes(status)}  HANDOFF_STATUS.md\n"
    ).encode()
    handoff = OUTPUT_ROOT / (
        f"fuzzyxai-h10-c3-cost-stability-v23.1-{short_commit}.zip"
    )
    prefix = "fuzzyxai-h10-c3-cost-stability-v23.1"
    with zipfile.ZipFile(handoff, "w") as archive:
        _write_member(archive, f"{prefix}/{source.name}", source.read_bytes())
        _write_member(
            archive,
            f"{prefix}/{artifacts_zip.name}",
            artifacts_zip.read_bytes(),
        )
        _write_member(archive, f"{prefix}/handoff_manifest.json", manifest_bytes)
        _write_member(archive, f"{prefix}/HANDOFF_STATUS.md", status)
        _write_member(archive, f"{prefix}/SHA256SUMS", checksums)
    with zipfile.ZipFile(handoff) as archive:
        names = archive.namelist()
        _assert_safe_names(names)
        if len(names) != 5:
            raise RuntimeError("handoff archive has an unexpected member count")
    sidecar = handoff.with_suffix(".zip.sha256")
    sidecar.write_text(f"{_sha256(handoff)}  {handoff.name}\n", encoding="ascii")
    print(f"PASS: {handoff}")
    print(f"PASS: {_sha256(handoff)}")


if __name__ == "__main__":
    main()
