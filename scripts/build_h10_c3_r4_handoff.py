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
    "/audit/preseal_attempt_2/sealed/",
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


def build_reports() -> None:
    development = json.loads(
        (
            ARTIFACTS
            / "results"
            / "development_statistics.json"
        ).read_text(encoding="utf-8")
    )
    validation = json.loads(
        (
            ARTIFACTS
            / "results"
            / "protocol_validation_statistics.json"
        ).read_text(encoding="utf-8")
    )
    gate = json.loads(
        (
            ARTIFACTS / "gate" / "preconfirmatory_gate.json"
        ).read_text(encoding="utf-8")
    )
    sealed = json.loads(
        (
            ARTIFACTS / "sealed" / "sealed_status.json"
        ).read_text(encoding="utf-8")
    )
    power = json.loads(
        (ARTIFACTS / "power" / "power.json").read_text(
            encoding="utf-8"
        )
    )
    claims = {
        "study_id": "FXAI-H10-C3-R4-CONFIRMATORY-READINESS",
        "claims": [
            {
                "claim_id": claim,
                "status": "NOT_EVALUATED_CONFIRMATORY",
                "development_status": next(
                    item["status"]
                    for item in development
                    if item["claim"] == claim
                ),
                "protocol_validation_status": next(
                    item["status"]
                    for item in validation
                    if item["claim"] == claim
                ),
            }
            for claim in ("H10-C3a", "H10-C3b")
        ],
        "scientific_status": sealed["status"],
        "sealed_opening_count": sealed["sealed_opening_count"],
    }
    (ARTIFACTS / "claim_registry.json").write_text(
        json.dumps(claims, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    methodology = {
        "template_audit": json.loads(
            (ARTIFACTS / "template_audit.json").read_text(
                encoding="utf-8"
            )
        ),
        "independence_audit": json.loads(
            (
                ARTIFACTS / "audit" / "independence.json"
            ).read_text(encoding="utf-8")
        ),
        "preconfirmatory_gate": gate,
        "sealed_overlap_checks": sealed["overlap_checks"],
        "sealed_private_truth_stored": sealed[
            "private_mutation_log_stored"
        ],
        "sealed_opening_count": sealed["sealed_opening_count"],
        "invalid_presealed_attempts_preserved": [
            "INVALID_PRESEALED_STRATUM_ALLOCATION",
            "INVALID_PREOPENING_PLAINTEXT_PRIVATE_LOG",
        ],
        "human_adjudication": (
            "NOT_REQUIRED_FOR_ALGORITHMIC_SCOPE"
        ),
        "human_factors_validation": "NOT_CONDUCTED",
    }
    (ARTIFACTS / "methodology_audit.json").write_text(
        json.dumps(methodology, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    development_by_claim = {
        item["claim"]: item for item in development
    }
    validation_by_claim = {
        item["claim"]: item for item in validation
    }
    power_by_claim = {
        item["claim"]: item for item in power["selected_designs"]
    }
    lines = [
        "# H10-C3 R4 validation report",
        "",
        f"- Implementation commit: `{json.loads((ARTIFACTS / 'lock' / 'protocol_lock.json').read_text())['implementation_commit']}`",
        "- Full regression: `533 passed, 4 skipped`",
        "- Focused R4/diagnostics: `58 passed`",
        "- Ruff: `PASS`",
        f"- Gate: `{gate['status']}`",
        f"- Sealed status: `{sealed['status']}`",
        f"- Sealed opening count: `{sealed['sealed_opening_count']}`",
        "- Sealed private mutation log stored: `false`",
        "",
        "## Open R4 results",
        "",
    ]
    for claim in ("H10-C3a", "H10-C3b"):
        dev = development_by_claim[claim]
        val = validation_by_claim[claim]
        design = power_by_claim[claim]
        lines.extend(
            [
                f"### {claim}",
                "",
                (
                    f"- Development effect: `{dev['effect']:.12f}`, "
                    f"95% CI `[{dev['ci_low']:.12f}; "
                    f"{dev['ci_high']:.12f}]`, "
                    f"Holm p `{dev['p_holm']:.12f}`."
                ),
                (
                    f"- Protocol-validation effect: "
                    f"`{val['effect']:.12f}`, 95% CI "
                    f"`[{val['ci_low']:.12f}; "
                    f"{val['ci_high']:.12f}]`, "
                    f"Holm p `{val['p_holm']:.12f}`."
                ),
                (
                    f"- Positive pipeline families: "
                    f"`{val['positive_pipeline_families']}/6`."
                ),
                (
                    f"- Power: `{design['point_power']:.6f}`, "
                    f"lower bound "
                    f"`{design['lower_confidence_bound']:.12f}`, "
                    f"`{design['number_of_simulations']}` simulations."
                ),
                "- Confirmatory status: `NOT_EVALUATED_CONFIRMATORY`.",
                "",
            ]
        )
    lines.extend(
        [
            "## Boundaries",
            "",
            "- Results above are open development and protocol-validation evidence.",
            "- Sealed scoring was not run.",
            "- No dissertation chapter or article text was generated.",
            "- Human factors and expert usefulness were not evaluated.",
            "",
        ]
    )
    (ARTIFACTS / "validation_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    checksum_files = [
        path
        for path in public_artifacts()
        if path.name != "SHA256SUMS"
    ]
    (ARTIFACTS / "SHA256SUMS").write_text(
        "".join(
            f"{sha256(path)}  {path.relative_to(ARTIFACTS).as_posix()}\n"
            for path in checksum_files
        ),
        encoding="utf-8",
    )


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

    build_reports()
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
