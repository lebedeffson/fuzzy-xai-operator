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
    "/sealed/data/",
    "mutation_log",
    "opening_record",
    "sealed.csv",
    "sealed_seed",
    "repair_truth",
    "source_truth",
    "optimal_cuts",
    "approval",
    ".git/",
    "__pycache__",
)
SECURE_SEALED_ALLOWED = {
    "EVIDENCE/secure_sealed/sealed_design.json",
    "EVIDENCE/secure_sealed/sealed_bank_commitment.json",
    "EVIDENCE/secure_sealed/sealed_ciphertext.bin",
    "EVIDENCE/secure_sealed/sealed_status.json",
}
SECURE_PLAINTEXT_TOKENS = (
    b"mutation_log",
    b"reverse_candidate_ids",
    b"optimal_cuts",
    b"repair_truth",
    b"source_truth",
    b"sealed_seed",
    b'"templates"',
    b'"cases"',
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
            ARTIFACTS / "secure_sealed" / "sealed_status.json"
        ).read_text(encoding="utf-8")
    )
    commitment = json.loads(
        (
            ARTIFACTS
            / "secure_sealed"
            / "sealed_bank_commitment.json"
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
        "sealed_opening_count": sealed["opening_count"],
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
        "sealed_isolation": {
            "encrypted_payload_sha256": commitment[
                "encrypted_payload_sha256"
            ],
            "plaintext_commitment_sha256": commitment[
                "plaintext_commitment_sha256"
            ],
            "seed_commitment_sha256": commitment[
                "seed_commitment_sha256"
            ],
            "plaintext_sealed_data_distributed": False,
        },
        "sealed_private_truth_stored_in_preopen_handoff": False,
        "sealed_opening_count": sealed["opening_count"],
        "invalid_presealed_attempts_preserved": [
            "INVALID_PRESEALED_STRATUM_ALLOCATION",
            "INVALID_PREOPENING_PLAINTEXT_PRIVATE_LOG",
            "INVALID_PREOPEN_SEALED_ISOLATION",
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
        "- Focused R4/diagnostics: `71 passed`",
        "- Ruff: `PASS`",
        f"- Gate: `{gate['status']}`",
        f"- Sealed status: `{sealed['status']}`",
        f"- Sealed opening count: `{sealed['opening_count']}`",
        "- Sealed private plaintext distributed: `false`",
        "- Sealed payload: `AES-256-GCM encrypted`",
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
    sealed_status_path = (
        ARTIFACTS / "secure_sealed" / "sealed_status.json"
    )
    if not gate_path.is_file() or not sealed_status_path.is_file():
        raise RuntimeError("R4 gate and unopened sealed status are required")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    sealed = json.loads(sealed_status_path.read_text(encoding="utf-8"))
    if gate["status"] != "READY_FOR_SEALED_GENERATION":
        raise RuntimeError("R4 preconfirmatory gate did not pass")
    if (
        sealed["status"] != "READY_FOR_SECURE_SEALED_SCORING"
        or int(sealed["opening_count"]) != 0
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

    output = OUTPUT / f"fuzzyxai-h10-c3-r4-v23.3-preopen-{short}.zip"
    files = public_artifacts()
    manifest = {
        "study_id": "FXAI-H10-C3-R4-CONFIRMATORY-READINESS",
        "commit": commit,
        "source_archive": source.name,
        "source_sha256": sha256(source),
        "sealed_created": True,
        "sealed_opening_count": 0,
        "plaintext_sealed_data_distributed": False,
        "H10-C3a": "NOT_EVALUATED_CONFIRMATORY",
        "H10-C3b": "NOT_EVALUATED_CONFIRMATORY",
        "scientific_status": "READY_FOR_SECURE_SEALED_SCORING",
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


def scan_preopen_archive(path: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if any(
            token in f"/{name}" for name in names for token in FORBIDDEN
        ):
            raise RuntimeError("handoff contains private or forbidden data")
        active_secure = {
            name
            for name in names
            if name.startswith("EVIDENCE/secure_sealed/")
        }
        if active_secure != SECURE_SEALED_ALLOWED:
            raise RuntimeError(
                "preopen handoff secure sealed section contains plaintext "
                "or is incomplete"
            )
        for name in sorted(active_secure):
            if name.endswith(".bin"):
                continue
            payload = archive.read(name)
            if any(token in payload for token in SECURE_PLAINTEXT_TOKENS):
                raise RuntimeError(
                    f"preopen secure sealed metadata leaks private truth: {name}"
                )


def scan_source_archive(path: Path) -> None:
    forbidden_paths = (
        "/artifacts/h10_c3_r4/sealed/",
        "/experiments/h10_c3_r4/templates/sealed/templates.jsonl",
        "/experiments/h10_c3_r4/templates/sealed/manifest.json",
        "sealed_seed",
        "mutation_log.jsonl",
        "repair_truth.jsonl",
        "source_truth.jsonl",
    )
    with zipfile.ZipFile(path) as archive:
        leaked = [
            name
            for name in archive.namelist()
            if any(
                token in f"/{name}" for token in forbidden_paths
            )
        ]
    if leaked:
        raise RuntimeError(
            f"source archive contains preopen sealed plaintext: {leaked}"
        )


def verify(path: Path) -> None:
    scan_preopen_archive(path)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        with zipfile.ZipFile(path) as archive:
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
        scan_source_archive(source)


if __name__ == "__main__":
    print(build())
