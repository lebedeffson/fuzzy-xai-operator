#!/usr/bin/env python3
"""Build and publish auditable runtime-only compatibility images."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PINNED_IMAGE = re.compile(r"^[A-Za-z0-9._/:@-]+@sha256:[0-9a-f]{64}$")
SAFE_INCIDENT = re.compile(r"^[A-Za-z0-9_.-]+$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
TESTBED_PYTHON = "/opt/miniconda3/envs/testbed/bin/python"


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render_dockerfile(base_image: str, wheel_name: str, wheel_sha256: str) -> str:
    if not PINNED_IMAGE.fullmatch(base_image):
        raise ValueError("base image must be pinned by manifest digest")
    if Path(wheel_name).name != wheel_name or not wheel_name.endswith(".whl"):
        raise ValueError("wheel name is unsafe")
    if not SHA256.fullmatch(wheel_sha256):
        raise ValueError("wheel SHA256 is invalid")
    return (
        f"FROM {base_image}\n"
        f"COPY {wheel_name} /opt/fuzzyxai-runtime/{wheel_name}\n"
        f"RUN echo '{wheel_sha256}  /opt/fuzzyxai-runtime/{wheel_name}'"
        " | sha256sum -c - \\\n"
        f" && {TESTBED_PYTHON} -m pip install --no-index --no-deps"
        f" /opt/fuzzyxai-runtime/{wheel_name} \\\n"
        f" && {TESTBED_PYTHON} -c \"import numpy;"
        " assert numpy.__version__ == '1.26.4'\" \\\n"
        f" && rm /opt/fuzzyxai-runtime/{wheel_name}\n"
        'LABEL org.opencontainers.image.title="FuzzyXAI H10-C5b runtime compatibility image"\n'
        'LABEL org.opencontainers.image.description="Runtime-only NumPy compatibility; scientific method unchanged"\n'
        'LABEL org.fuzzyxai.h10-c5b.amendment="runtime-environment-v1"\n'
    )


def _run(command: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )


def _repository_name(prefix: str, incident_id: str) -> str:
    suffix = incident_id.replace("__", "-").replace("_", "-").replace(".", "-")
    return f"{prefix.rstrip('/')}/{suffix}".lower()


def build_images(
    *,
    base_registry_path: Path,
    incident_ids: tuple[str, ...],
    wheel_path: Path,
    wheel_sha256: str,
    registry_prefix: str,
    output_root: Path,
) -> dict[str, Any]:
    if not incident_ids or len(incident_ids) != len(set(incident_ids)):
        raise ValueError("incident IDs must be a non-empty unique sequence")
    if any(not SAFE_INCIDENT.fullmatch(value) for value in incident_ids):
        raise ValueError("unsafe incident ID")
    if sha256_path(wheel_path) != wheel_sha256:
        raise ValueError("wheel SHA256 mismatch")

    base_registry = json.loads(base_registry_path.read_text(encoding="utf-8"))
    output_root.mkdir(parents=True, exist_ok=True)
    published: dict[str, str] = {}
    records: list[dict[str, Any]] = []
    for incident_id in incident_ids:
        base_image = str(base_registry.get(incident_id, ""))
        if not PINNED_IMAGE.fullmatch(base_image):
            raise ValueError(f"missing pinned base image for {incident_id}")
        context = output_root / "contexts" / incident_id
        context.mkdir(parents=True, exist_ok=True)
        context_wheel = context / wheel_path.name
        shutil.copyfile(wheel_path, context_wheel)
        dockerfile = render_dockerfile(base_image, wheel_path.name, wheel_sha256)
        dockerfile_path = context / "Dockerfile"
        dockerfile_path.write_text(dockerfile, encoding="ascii")

        repository = _repository_name(registry_prefix, incident_id)
        tag = f"{repository}:runtime-v1"
        build = _run(
            [
                "docker",
                "build",
                "--pull=false",
                "--provenance=false",
                "--tag",
                tag,
                ".",
            ],
            cwd=context,
        )
        push = _run(["docker", "push", tag])
        inspect = json.loads(_run(["docker", "image", "inspect", tag]).stdout)[0]
        matching = [
            value
            for value in inspect.get("RepoDigests", ())
            if value.startswith(f"{repository}@sha256:")
        ]
        if len(matching) != 1 or not PINNED_IMAGE.fullmatch(matching[0]):
            raise ValueError(f"published digest is unavailable for {incident_id}")
        pinned_image = matching[0]
        smoke = _run(
            [
                "docker",
                "run",
                "--rm",
                "--network",
                "none",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges",
                pinned_image,
                TESTBED_PYTHON,
                "-c",
                (
                    "import numpy; "
                    "assert numpy.__version__ == '1.26.4'; "
                    "print(numpy.__version__)"
                ),
            ]
        )
        published[incident_id] = pinned_image
        records.append(
            {
                "incident_id": incident_id,
                "base_image": base_image,
                "derived_image": pinned_image,
                "dockerfile_sha256": sha256_path(dockerfile_path),
                "wheel_sha256": wheel_sha256,
                "build_stdout_sha256": hashlib.sha256(build.stdout.encode()).hexdigest(),
                "push_stdout_sha256": hashlib.sha256(push.stdout.encode()).hexdigest(),
                "smoke_stdout": smoke.stdout.strip(),
                "scientific_method_changed": False,
            }
        )

    registry_path = output_root / "H10_C5B_DERIVED_IMAGE_REGISTRY.json"
    registry_path.write_text(
        json.dumps(published, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    report = {
        "schema_version": "1.0",
        "status": "PASS",
        "amendment_id": "h10-c5b-runtime-environment-amendment-v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "base_registry_sha256": sha256_path(base_registry_path),
        "wheel": wheel_path.name,
        "wheel_sha256": wheel_sha256,
        "derived_registry": str(registry_path),
        "derived_registry_sha256": sha256_path(registry_path),
        "records": records,
    }
    (output_root / "H10_C5B_DERIVED_IMAGE_PROVENANCE.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-registry", type=Path, required=True)
    parser.add_argument("--incident", action="append", required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--wheel-sha256", required=True)
    parser.add_argument("--registry-prefix", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_images(
        base_registry_path=args.base_registry,
        incident_ids=tuple(args.incident),
        wheel_path=args.wheel,
        wheel_sha256=args.wheel_sha256,
        registry_prefix=args.registry_prefix,
        output_root=args.output,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
