from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts" / "independent_confirmatory"
DATA = ROOT / "data" / "independent_confirmatory"
PROTOCOL = ROOT / "config" / "independent_confirmatory_protocol.json"
AMENDMENT = ROOT / "config" / "independent_confirmatory_protocol_amendment_001.json"
SUMS = ARTIFACTS / "protocol" / "SHA256SUMS"
LOCK = ARTIFACTS / "lock" / "confirmatory_lock.json"
OPENING = ARTIFACTS / "opening" / "opening_record.json"
PRIVATE = DATA / "private_pipeline"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_protocol() -> dict[str, str]:
    registered = {}
    for line in SUMS.read_text(encoding="utf-8").splitlines():
        digest, path = line.split(maxsplit=1)
        registered[path] = digest
    result = {}
    for path in (PROTOCOL, AMENDMENT):
        relative = str(path.relative_to(ROOT))
        actual = sha256_file(path)
        if registered.get(relative) != actual:
            raise RuntimeError(f"protocol file changed after registration: {relative}")
        result[relative] = actual
    return result


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def git_tree_clean() -> bool:
    return not subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()


def stable_unit_interval(value: str, *, salt: str) -> float:
    digest = hashlib.sha256(f"{salt}:{value}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def decrypt_label_vault(dataset_id: str) -> dict[str, str]:
    dataset = DATA / dataset_id / "private"
    encrypted = dataset / "confirmatory_label_vault.enc"
    key = dataset / ".vault_passphrase"
    if not OPENING.is_file():
        raise RuntimeError("label vault cannot be opened before an immutable opening record exists")
    completed = subprocess.run(
        [
            "openssl",
            "enc",
            "-d",
            "-aes-256-cbc",
            "-pbkdf2",
            "-in",
            str(encrypted),
            "-pass",
            f"file:{key}",
        ],
        check=True,
        capture_output=True,
    )
    return json.loads(completed.stdout)


def private_mode(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        os.chmod(path, 0o600)
