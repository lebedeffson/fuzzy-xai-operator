"""Build one verified H10-C3 R4 handoff from committed source and public evidence."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "h10_c3_r4"
OUTPUT = ROOT / "release_artifacts"
FORBIDDEN = (
    "/private/",
    "mutation_log",
    "opening_record",
    "sealed.csv",
    "approval",
    ".git/",
    "__pycache__",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(*args: str) -> str:
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def public_artifacts() -> list[Path]:
    return [
        path
        for path in sorted(ARTIFACTS.rglob("*"))
        if path.is_file()
        and not any(
            token in f"/{path.relative_to(ROOT).as_posix()}"
            for token in FORBIDDEN
        )
    ]


def build() -> Path:
    gate_path = ARTIFACTS / "gate" / "preconfirmatory_gate.json"
    sealed_status_path = ARTIFACTS / "sealed" / "sealed_status.json"
    if not gate_path.is_file() or not sealed_status_path.is_file():
        raise RuntimeError("R4 gate and unopened sealed status are required")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    sealed = json.loads(sealed_status_path.read_text(encoding="utf-8"))
    if gate["status"] != "READY_FOR_SEALED_GENERATION":
        raise RuntimeError("R4 preconfirmatory gate did not pass")
    if (
        sealed["status"] != "READY_FOR_SEALED_SCORING"
        or int(sealed["sealed_opening_count"]) != 0
    ):
        raise RuntimeError("handoff requires an unopened R4 sealed set")

    subprocess.check_call(
        [sys.executable, "scripts/build_framework_release.py"],
        cwd=ROOT,
    )
    commit = run("git", "rev-parse", "HEAD")
    short = commit[:12]
    source = OUTPUT / f"fuzzyxai-source-release-{short}.zip"
    if not source.is_file():
        raise RuntimeError("committed-tree source release was not created")

    output = OUTPUT / f"fuzzyxai-h10-c3-r4-v23.2-{short}.zip"
    files = public_artifacts()
    manifest = {
        "study_id": "FXAI-H10-C3-R4-CONFIRMATORY-READINESS",
        "commit": commit,
        "source_archive": source.name,
        "source_sha256": sha256(source),
        "sealed_created": True,
        "sealed_opening_count": 0,
        "H10-C3a": "NOT_EVALUATED_CONFIRMATORY",
        "H10-C3b": "NOT_EVALUATED_CONFIRMATORY",
        "scientific_status": "READY_FOR_SEALED_SCORING",
        "artifacts": {
            path.relative_to(ROOT).as_posix(): sha256(path)
            for path in files
        },
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        output,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        archive.write(source, f"SOURCE/{source.name}")
        for path in files:
            archive.write(
                path,
                f"EVIDENCE/{path.relative_to(ARTIFACTS).as_posix()}",
            )
        archive.writestr(
            "HANDOFF_MANIFEST.json",
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        )
    checksum = output.with_suffix(".zip.sha256")
    checksum.write_text(
        f"{sha256(output)}  {output.name}\n",
        encoding="utf-8",
    )
    verify(output)
    return output


def verify(path: Path) -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if any(
                token in f"/{name}" for name in names for token in FORBIDDEN
            ):
                raise RuntimeError("handoff contains private or forbidden data")
            archive.extractall(root)
        manifest = json.loads(
            (root / "HANDOFF_MANIFEST.json").read_text(encoding="utf-8")
        )
        for relative, expected in manifest["artifacts"].items():
            extracted = (
                root
                / "EVIDENCE"
                / Path(relative).relative_to("artifacts/h10_c3_r4")
            )
            if sha256(extracted) != expected:
                raise RuntimeError(f"artifact checksum mismatch: {relative}")
        source = root / "SOURCE" / manifest["source_archive"]
        if sha256(source) != manifest["source_sha256"]:
            raise RuntimeError("source archive checksum mismatch")


if __name__ == "__main__":
    print(build())
