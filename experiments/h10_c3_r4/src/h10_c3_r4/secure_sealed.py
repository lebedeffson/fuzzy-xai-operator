from __future__ import annotations

import datetime as dt
import fcntl
import hashlib
import hmac
import json
import os
import random
import secrets
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from .generator import build_cases
from .models import (
    ContractTemplate,
    EdgeTemplate,
    NodeTemplate,
    RepairCandidateTemplate,
    RouteTemplate,
)
from .templates import PIPELINE_SCHEMAS, _build_template


STUDY_ID = "FXAI-H10-C3-R4-CONFIRMATORY-READINESS"
LOCKED_IMPLEMENTATION = "e729834c077ecb5c0011d9fb85d5f00b10129f18"
LEGACY_METHOD_LOCK_SHA256 = (
    "e9f95742d9de3be89327f293be3864e45ad3b6b1a002e1ae7e16b22cf0bdf881"
)
LEGACY_PROTOCOL_LOCK_SHA256 = (
    "29af0d239b6d9bcc60eb08e4c158bc8a21e2cb2d549b8435896961cc205c322f"
)
AUTHORIZATION = "AUTHORIZE_ONE_TIME_SEALED_SCORING"
MAGIC = b"FXAI-H10-C3-R4-AES256GCM\x00"
NONCE_BYTES = 12
DESIGN = {
    "pipeline_families": 6,
    "case_count": 240,
    "templates_per_family": 40,
    "cases_per_template": 1,
    "stratum_allocation": {
        "S2": 42,
        "S3": 66,
        "S4": 66,
        "S5": 66,
    },
    "per_pipeline_stratum_allocation": {
        "S2": 7,
        "S3": 11,
        "S4": 11,
        "S5": 11,
    },
}

ALLOWED_LEGACY_CHANGES = {
    "experiments/h10_c3_r4/src/h10_c3_r4/cli.py",
    "experiments/h10_c3_r4/src/h10_c3_r4/runner.py",
    "experiments/h10_c3_r4/templates/sealed/manifest.json",
    "experiments/h10_c3_r4/templates/sealed/templates.jsonl",
}
OPERATIONAL_FILES = (
    ".github/workflows/h10-c3-r4.yml",
    "Makefile",
    "PROJECT_MEMORY.md",
    "experiments/h10_c3_r4/src/h10_c3_r4/cli.py",
    "experiments/h10_c3_r4/src/h10_c3_r4/runner.py",
    "experiments/h10_c3_r4/src/h10_c3_r4/scientific_classifier.py",
    "experiments/h10_c3_r4/src/h10_c3_r4/secure_sealed.py",
    "experiments/h10_c3_r4/tests/test_secure_sealed.py",
    "experiments/h10_c3_r4/tests/test_h10_c3_r4.py",
    "scripts/build_h10_c3_r4_handoff.py",
)
OPEN_EVIDENCE_FILES = (
    "artifacts/h10_c3_r4/results/development.csv",
    "artifacts/h10_c3_r4/results/development_statistics.json",
    "artifacts/h10_c3_r4/results/protocol_validation.csv",
    "artifacts/h10_c3_r4/results/protocol_validation_statistics.json",
    "artifacts/h10_c3_r4/results/stability.json",
    "artifacts/h10_c3_r4/power/power.json",
    "artifacts/h10_c3_r4/gate/preconfirmatory_gate.json",
    "artifacts/h10_c3_r4/audit/reconstructible_sealed_v23_2/INVALIDATION.json",
    "artifacts/h10_c3_r4/audit/secure_payload_attempt_v23_3_ci_mismatch/INVALIDATION.json",
)
FORBIDDEN_PREOPEN_PATHS = (
    "artifacts/h10_c3_r4/sealed/data/sealed/cases.jsonl",
    "artifacts/h10_c3_r4/sealed/data/sealed/manifest.json",
    "artifacts/h10_c3_r4/sealed/sealed_status.json",
    "experiments/h10_c3_r4/templates/sealed/templates.jsonl",
    "experiments/h10_c3_r4/templates/sealed/manifest.json",
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_write(path: Path, value: bytes, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_write_json(path: Path, value: object) -> None:
    _atomic_write(path, json.dumps(value, indent=2, sort_keys=True).encode() + b"\n")


def _repo_path(repo_root: Path, relative: str) -> Path:
    return repo_root / Path(relative)


def _current_commit(repo_root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        text=True,
    ).strip()


def _assert_clean(repo_root: Path) -> None:
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        text=True,
    ).strip()
    if dirty:
        raise RuntimeError("secure protocol lock requires a clean committed tree")


def _assert_plaintext_absent(repo_root: Path) -> None:
    present = [
        relative
        for relative in FORBIDDEN_PREOPEN_PATHS
        if _repo_path(repo_root, relative).exists()
    ]
    if present:
        raise PermissionError(
            f"preopen sealed plaintext is still present: {present}"
        )


def create_secure_protocol_lock(
    repo_root: Path,
    artifact_root: Path,
) -> Path:
    _assert_clean(repo_root)
    legacy_method_path = artifact_root / "lock" / "method_lock.json"
    legacy_protocol_path = artifact_root / "lock" / "protocol_lock.json"
    if _sha256_file(legacy_method_path) != LEGACY_METHOD_LOCK_SHA256:
        raise PermissionError("legacy method lock changed")
    if _sha256_file(legacy_protocol_path) != LEGACY_PROTOCOL_LOCK_SHA256:
        raise PermissionError("legacy protocol lock changed")
    _assert_plaintext_absent(repo_root)

    legacy_method = _read_json(legacy_method_path)
    protected = {
        relative: expected
        for relative, expected in legacy_method["files"].items()
        if relative not in ALLOWED_LEGACY_CHANGES
    }
    for relative, expected in protected.items():
        if _sha256_file(_repo_path(repo_root, relative)) != expected:
            raise PermissionError(f"locked scientific file changed: {relative}")

    lock = {
        "study_id": STUDY_ID,
        "lock_version": "v23.3-secure-preopen",
        "source_commit": _current_commit(repo_root),
        "locked_implementation": LOCKED_IMPLEMENTATION,
        "legacy_method_lock_sha256": LEGACY_METHOD_LOCK_SHA256,
        "legacy_protocol_lock_sha256": LEGACY_PROTOCOL_LOCK_SHA256,
        "protected_scientific_files": protected,
        "operational_files": {
            relative: _sha256_file(_repo_path(repo_root, relative))
            for relative in OPERATIONAL_FILES
        },
        "open_evidence_files": {
            relative: _sha256_file(_repo_path(repo_root, relative))
            for relative in OPEN_EVIDENCE_FILES
        },
        "forbidden_preopen_paths": FORBIDDEN_PREOPEN_PATHS,
        "sealed_design": DESIGN,
        "classification_rules": {
            "H10-C3a": {
                "effect_min": 0.04,
                "ci_low_strictly_positive": True,
                "holm_p_max_exclusive": 0.05,
                "positive_pipeline_families_min": 5,
                "cost_regret_effect_strictly_positive": True,
                "cost_regret_ci_low_strictly_positive": True,
                "cost_regret_holm_p_max_exclusive": 0.05,
            },
            "H10-C3b": {
                "effect_min": 0.04,
                "ci_low_strictly_positive": True,
                "holm_p_max_exclusive": 0.05,
                "positive_pipeline_families_min": 5,
            },
            "safety": {
                "false_certification_max": 0.01,
                "new_critical_violations_max": 0.01,
                "runtime_p95_ms_max": 50.0,
            },
        },
        "sealed_opening_count": 0,
        "post_lock_tuning": False,
    }
    output = artifact_root / "lock" / "secure_protocol_lock.json"
    _atomic_write_json(output, lock)
    return output


def verify_secure_protocol_lock(
    repo_root: Path,
    artifact_root: Path,
) -> dict[str, Any]:
    lock_path = artifact_root / "lock" / "secure_protocol_lock.json"
    if not lock_path.is_file():
        raise PermissionError("secure R4 protocol lock is missing")
    lock = _read_json(lock_path)
    if lock.get("study_id") != STUDY_ID:
        raise PermissionError("secure R4 protocol lock study mismatch")
    if lock.get("locked_implementation") != LOCKED_IMPLEMENTATION:
        raise PermissionError("locked implementation mismatch")
    method_path = artifact_root / "lock" / "method_lock.json"
    protocol_path = artifact_root / "lock" / "protocol_lock.json"
    if _sha256_file(method_path) != lock["legacy_method_lock_sha256"]:
        raise PermissionError("legacy method lock changed")
    if _sha256_file(protocol_path) != lock["legacy_protocol_lock_sha256"]:
        raise PermissionError("legacy protocol lock changed")
    for section in (
        "protected_scientific_files",
        "operational_files",
        "open_evidence_files",
    ):
        for relative, expected in lock[section].items():
            path = _repo_path(repo_root, relative)
            if not path.is_file() or _sha256_file(path) != expected:
                raise PermissionError(f"secure lock mismatch: {relative}")
    _assert_plaintext_absent(repo_root)
    return lock


def _seeded_random(seed: bytes, context: str) -> random.Random:
    derived = hmac.new(seed, context.encode(), hashlib.sha256).digest()
    return random.Random(int.from_bytes(derived, "big"))


def _public_template_hashes(repo_root: Path) -> set[str]:
    hashes: set[str] = set()
    for split in ("development", "protocol_validation"):
        manifest = _read_json(
            repo_root
            / "experiments"
            / "h10_c3_r4"
            / "templates"
            / split
            / "manifest.json"
        )
        hashes.update(manifest["canonical_hashes"])
    return hashes


def _private_template_bank(
    seed: bytes,
    repo_root: Path,
) -> tuple[RouteTemplate, ...]:
    public_hashes = _public_template_hashes(repo_root)
    templates: list[RouteTemplate] = []
    seen = set(public_hashes)
    allocation = DESIGN["per_pipeline_stratum_allocation"]
    for pipeline in sorted(PIPELINE_SCHEMAS):
        for stratum in ("S2", "S3", "S4", "S5"):
            rng = _seeded_random(seed, f"templates:{pipeline}:{stratum}")
            count = int(allocation[stratum])
            built = 0
            while built < count:
                index = rng.randrange(100_000_000, 2_000_000_000)
                template = _build_template(
                    "sealed",
                    pipeline,
                    stratum,
                    index,
                    rng,
                )
                if template.canonical_hash in seen:
                    continue
                templates.append(template)
                seen.add(template.canonical_hash)
                built += 1
    order = _seeded_random(seed, "sealed-case-order")
    order.shuffle(templates)
    if len(templates) != DESIGN["case_count"]:
        raise AssertionError("private sealed design size mismatch")
    return tuple(templates)


def _gold_inputs(case: object) -> dict[str, object]:
    candidates = tuple(case.mutated_graph.metadata["public_candidates"])
    obligations = sorted(
        {
            obligation
            for candidate in candidates
            for obligation in candidate["covers"]
        }
    )
    return {
        "obligations": obligations,
        "public_candidates": candidates,
        "repairable": case.repairable,
    }


def _case_record(case: object) -> dict[str, object]:
    return {
        "public_case": case.public_view(),
        "mutation_record": case.private_record(),
        "gold_inputs": _gold_inputs(case),
    }


def _private_payload(seed: bytes, repo_root: Path) -> dict[str, object]:
    templates = _private_template_bank(seed, repo_root)
    cases = build_cases(templates, cases_per_template=1)
    records = tuple(_case_record(case) for case in cases)
    return {
        "payload_version": "v23.3",
        "study_id": STUDY_ID,
        "sealed_design": DESIGN,
        "templates": tuple(template.to_dict() for template in templates),
        "cases": records,
        "generation_manifest": {
            "template_count": len(templates),
            "case_count": len(cases),
            "template_hashes": tuple(
                template.canonical_hash for template in templates
            ),
            "case_record_hashes": tuple(
                _sha256_bytes(_canonical_bytes(record)) for record in records
            ),
            "generator": "secret_derived_private_template_bank_v23.3",
            "public_template_overlap": False,
        },
    }


def _derive_key(seed: bytes, protocol_lock_sha256: str) -> bytes:
    return HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=bytes.fromhex(protocol_lock_sha256),
        info=f"{STUDY_ID}:sealed-payload:v23.3".encode(),
    ).derive(seed)


def _encrypt(
    plaintext: bytes,
    seed: bytes,
    protocol_lock_sha256: str,
) -> bytes:
    nonce = secrets.token_bytes(NONCE_BYTES)
    aad = f"{STUDY_ID}|{protocol_lock_sha256}".encode()
    ciphertext = AESGCM(_derive_key(seed, protocol_lock_sha256)).encrypt(
        nonce,
        plaintext,
        aad,
    )
    return MAGIC + nonce + ciphertext


def _decrypt(
    container: bytes,
    seed: bytes,
    protocol_lock_sha256: str,
) -> bytes:
    if not container.startswith(MAGIC):
        raise ValueError("invalid encrypted sealed container header")
    nonce_start = len(MAGIC)
    nonce_end = nonce_start + NONCE_BYTES
    nonce = container[nonce_start:nonce_end]
    ciphertext = container[nonce_end:]
    aad = f"{STUDY_ID}|{protocol_lock_sha256}".encode()
    return AESGCM(_derive_key(seed, protocol_lock_sha256)).decrypt(
        nonce,
        ciphertext,
        aad,
    )


def _assert_external_secret_path(repo_root: Path, secret_path: Path) -> None:
    repo = repo_root.resolve()
    resolved = secret_path.expanduser().resolve()
    if resolved == repo or repo in resolved.parents:
        raise PermissionError("sealed secret must be stored outside the repository")
    if resolved.exists():
        raise FileExistsError("sealed secret path already exists")


def _write_secret(secret_path: Path, seed: bytes) -> None:
    secret_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        secret_path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(seed)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        secret_path.unlink(missing_ok=True)
        raise
    if (secret_path.stat().st_mode & 0o077) != 0:
        secret_path.unlink(missing_ok=True)
        raise PermissionError("sealed secret permissions are not owner-only")


def create_secure_sealed(
    repo_root: Path,
    artifact_root: Path,
    secret_path: Path,
) -> Path:
    verify_secure_protocol_lock(repo_root, artifact_root)
    gate = _read_json(
        artifact_root / "gate" / "preconfirmatory_gate.json"
    )
    if gate["status"] != "READY_FOR_SEALED_GENERATION":
        raise PermissionError("R4 gate does not permit sealed generation")
    secure_root = artifact_root / "secure_sealed"
    if secure_root.exists():
        raise FileExistsError("secure sealed R4 set already exists")
    _assert_external_secret_path(repo_root, secret_path)

    seed = secrets.token_bytes(32)
    protocol_sha = _sha256_file(
        artifact_root / "lock" / "secure_protocol_lock.json"
    )
    payload = _private_payload(seed, repo_root)
    plaintext = _canonical_bytes(payload)
    container = _encrypt(plaintext, seed, protocol_sha)
    seed_commitment = _sha256_bytes(
        STUDY_ID.encode() + protocol_sha.encode() + seed
    )

    temporary_root = Path(
        tempfile.mkdtemp(prefix=".secure-sealed-", dir=artifact_root)
    )
    try:
        design = {
            "study_id": STUDY_ID,
            "protocol_lock_sha256": protocol_sha,
            **DESIGN,
        }
        commitment = {
            "study_id": STUDY_ID,
            "protocol_lock_sha256": protocol_sha,
            "seed_commitment_sha256": seed_commitment,
            "encrypted_payload_sha256": _sha256_bytes(container),
            "plaintext_commitment_sha256": _sha256_bytes(plaintext),
            "pipeline_families": DESIGN["pipeline_families"],
            "case_count": DESIGN["case_count"],
            "stratum_allocation": DESIGN["stratum_allocation"],
            "opening_count": 0,
        }
        status = {
            "study_id": STUDY_ID,
            "status": "READY_FOR_SECURE_SEALED_SCORING",
            "implementation": "LOCKED",
            "sealed_created": True,
            "plaintext_sealed_data_distributed": False,
            "opening_count": 0,
            "H10-C3a": "NOT_EVALUATED_CONFIRMATORY",
            "H10-C3b": "NOT_EVALUATED_CONFIRMATORY",
            "scientific_status": "READY_FOR_SECURE_SEALED_SCORING",
        }
        _atomic_write_json(temporary_root / "sealed_design.json", design)
        _atomic_write_json(
            temporary_root / "sealed_bank_commitment.json",
            commitment,
        )
        _atomic_write(
            temporary_root / "sealed_ciphertext.bin",
            container,
        )
        _atomic_write_json(temporary_root / "sealed_status.json", status)
        _write_secret(secret_path, seed)
        os.replace(temporary_root, secure_root)
    except Exception:
        shutil.rmtree(temporary_root, ignore_errors=True)
        secret_path.unlink(missing_ok=True)
        raise
    return secure_root / "sealed_status.json"


def _template_from_dict(value: dict[str, object]) -> RouteTemplate:
    return RouteTemplate(
        template_id=str(value["template_id"]),
        split=str(value["split"]),
        pipeline_family=str(value["pipeline_family"]),
        modality=str(value["modality"]),
        stratum=str(value["stratum"]),
        node_schema=tuple(
            NodeTemplate(
                role=str(node["role"]),
                node_type=str(node["node_type"]),
                attributes=tuple(node["attributes"]),
            )
            for node in value["node_schema"]
        ),
        edge_schema=tuple(
            EdgeTemplate(
                source_role=str(edge["source_role"]),
                target_role=str(edge["target_role"]),
                relation=str(edge["relation"]),
                mandatory=bool(edge["mandatory"]),
            )
            for edge in value["edge_schema"]
        ),
        contract_schema=tuple(
            ContractTemplate(
                contract_id=str(contract["contract_id"]),
                subject_role=str(contract["subject_role"]),
                kind=str(contract["kind"]),
                field=str(contract["field"]),
                expected=str(contract["expected"]),
                category=str(contract["category"]),
                source_roles=tuple(contract["source_roles"]),
                repairable=bool(contract["repairable"]),
            )
            for contract in value["contract_schema"]
        ),
        candidates=tuple(
            RepairCandidateTemplate(
                candidate_id=str(candidate["candidate_id"]),
                source_role=str(candidate["source_role"]),
                covers=tuple(candidate["covers"]),
                cost=float(candidate["cost"]),
                dependencies=tuple(candidate["dependencies"]),
                executable=bool(candidate["executable"]),
            )
            for candidate in value["candidates"]
        ),
        mutation_grammar_id=str(value["mutation_grammar_id"]),
        repair_grammar_id=str(value["repair_grammar_id"]),
        graph_hash=str(value["graph_hash"]),
        coverage_hash=str(value["coverage_hash"]),
        mutation_hash=str(value["mutation_hash"]),
        repair_dependency_hash=str(value["repair_dependency_hash"]),
        cost_hash=str(value["cost_hash"]),
        canonical_hash=str(value["canonical_hash"]),
    )


def _cases_from_plaintext(plaintext: bytes) -> tuple[object, ...]:
    payload = json.loads(plaintext)
    if payload.get("study_id") != STUDY_ID:
        raise ValueError("decrypted sealed study mismatch")
    if payload.get("sealed_design") != DESIGN:
        raise ValueError("decrypted sealed design mismatch")
    templates = tuple(
        _template_from_dict(template) for template in payload["templates"]
    )
    cases = build_cases(templates, cases_per_template=1)
    expected_records = payload["cases"]
    actual_records = [_case_record(case) for case in cases]
    if _canonical_bytes(actual_records) != _canonical_bytes(expected_records):
        raise ValueError("decrypted sealed cases fail generation verification")
    manifest = payload["generation_manifest"]
    if manifest["case_record_hashes"] != [
        _sha256_bytes(_canonical_bytes(record)) for record in actual_records
    ]:
        raise ValueError("decrypted sealed case commitments mismatch")
    return cases


def _validate_approval(
    approval: dict[str, object],
    commitment: dict[str, object],
) -> None:
    required = {
        "study_id": STUDY_ID,
        "protocol_lock_sha256": commitment["protocol_lock_sha256"],
        "encrypted_payload_sha256": commitment[
            "encrypted_payload_sha256"
        ],
        "plaintext_commitment_sha256": commitment[
            "plaintext_commitment_sha256"
        ],
        "authorization": AUTHORIZATION,
    }
    if any(approval.get(key) != value for key, value in required.items()):
        raise PermissionError("sealed approval does not match the secure lock")
    if not str(approval.get("authorized_by", "")).strip():
        raise PermissionError("sealed approval lacks protocol owner")
    try:
        dt.datetime.fromisoformat(str(approval["authorized_at_utc"]))
    except (KeyError, ValueError) as exc:
        raise PermissionError("sealed approval timestamp is invalid") from exc


def _record_opening(
    status_path: Path,
    approval_path: Path,
    commitment: dict[str, object],
) -> None:
    status = _read_json(status_path)
    if int(status["opening_count"]) != 0:
        raise PermissionError("secure sealed R4 scoring cannot be repeated")
    opening = {
        "study_id": STUDY_ID,
        "opening_count_before": 0,
        "opening_count_after": 1,
        "opened_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "protocol_lock_sha256": commitment["protocol_lock_sha256"],
        "encrypted_payload_sha256": commitment[
            "encrypted_payload_sha256"
        ],
        "plaintext_commitment_sha256": commitment[
            "plaintext_commitment_sha256"
        ],
        "approval_sha256": _sha256_file(approval_path),
        "purpose": "one_time_confirmatory_scoring",
    }
    status.update(
        {
            "status": "SEALED_SCORING_IN_PROGRESS",
            "opening_count": 1,
            "scientific_status": "SEALED_SCORING_IN_PROGRESS",
        }
    )
    _atomic_write_json(status_path, status)
    _atomic_write_json(status_path.parent / "opening_record.json", opening)


def load_secure_sealed_cases(
    repo_root: Path,
    artifact_root: Path,
    approval_path: Path,
    secret_path: Path,
) -> tuple[object, ...]:
    verify_secure_protocol_lock(repo_root, artifact_root)
    secure_root = artifact_root / "secure_sealed"
    status_path = secure_root / "sealed_status.json"
    commitment = _read_json(
        secure_root / "sealed_bank_commitment.json"
    )
    ciphertext_path = secure_root / "sealed_ciphertext.bin"
    if _sha256_file(ciphertext_path) != commitment[
        "encrypted_payload_sha256"
    ]:
        raise PermissionError("encrypted sealed payload changed")
    if not approval_path.is_file():
        raise PermissionError("sealed approval file is missing")
    approval = _read_json(approval_path)
    _validate_approval(approval, commitment)
    if not secret_path.is_file():
        raise PermissionError("sealed secret is missing")
    seed = secret_path.read_bytes()
    if len(seed) != 32:
        raise PermissionError("sealed secret must contain exactly 256 bits")
    expected_seed_commitment = _sha256_bytes(
        STUDY_ID.encode()
        + str(commitment["protocol_lock_sha256"]).encode()
        + seed
    )
    if not hmac.compare_digest(
        expected_seed_commitment,
        str(commitment["seed_commitment_sha256"]),
    ):
        raise PermissionError("sealed secret does not match commitment")

    lock_path = secure_root / ".opening.lock"
    lock_path.touch(mode=0o600, exist_ok=True)
    with lock_path.open("rb") as lock_stream:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
        status = _read_json(status_path)
        if int(status["opening_count"]) != 0:
            raise PermissionError("secure sealed R4 scoring cannot be repeated")
        _record_opening(status_path, approval_path, commitment)
        try:
            container = ciphertext_path.read_bytes()
            plaintext = _decrypt(
                container,
                seed,
                str(commitment["protocol_lock_sha256"]),
            )
            if _sha256_bytes(plaintext) != commitment[
                "plaintext_commitment_sha256"
            ]:
                raise ValueError("decrypted sealed payload commitment mismatch")
            return _cases_from_plaintext(plaintext)
        except Exception as exc:
            failed = _read_json(status_path)
            failed.update(
                {
                    "status": "SEALED_SCORING_FAILED_NO_REUSE",
                    "scientific_status": "SEALED_SCORING_FAILED_NO_REUSE",
                    "error_type": type(exc).__name__,
                }
            )
            _atomic_write_json(status_path, failed)
            raise


def mark_scoring_failed(
    artifact_root: Path,
    exc: BaseException,
) -> None:
    status_path = (
        artifact_root / "secure_sealed" / "sealed_status.json"
    )
    status = _read_json(status_path)
    if int(status["opening_count"]) != 1:
        raise RuntimeError("cannot mark unopened sealed scoring as failed")
    status.update(
        {
            "status": "SEALED_SCORING_FAILED_NO_REUSE",
            "scientific_status": "SEALED_SCORING_FAILED_NO_REUSE",
            "error_type": type(exc).__name__,
        }
    )
    _atomic_write_json(status_path, status)


def mark_scoring_complete(
    artifact_root: Path,
    *,
    results_sha256: str,
    classification: dict[str, object],
) -> Path:
    status_path = (
        artifact_root / "secure_sealed" / "sealed_status.json"
    )
    status = _read_json(status_path)
    if int(status["opening_count"]) != 1:
        raise RuntimeError("cannot complete unopened sealed scoring")
    status.update(
        {
            "status": "SEALED_SCORED",
            "results_sha256": results_sha256,
            **classification,
        }
    )
    _atomic_write_json(status_path, status)
    return status_path
