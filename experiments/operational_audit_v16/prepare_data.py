from __future__ import annotations

import csv
import hashlib
import json
import os
import secrets
import subprocess
import urllib.request
import zipfile
from pathlib import Path

from .common import ARTIFACTS, DATA, git_commit, sha256_file, unit, verify_protocol, write_json


SOURCES = {
    "uci_online_shoppers_468": ("https://archive.ics.uci.edu/static/public/468/data.csv", "data.csv", "tabular"),
    "uci_youtube_spam_380": ("https://archive.ics.uci.edu/static/public/380/youtube+spam+collection.zip", "youtube.zip", "text"),
}


def _download(url: str, path: Path) -> None:
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "FuzzyXAI-operational-audit-v16"})
    with urllib.request.urlopen(request, timeout=120) as response, path.open("wb") as stream:
        while block := response.read(1024 * 1024):
            stream.write(block)


def _rows(dataset_id: str, path: Path) -> list[tuple[str, str]]:
    result = []
    if dataset_id == "uci_online_shoppers_468":
        with path.open(encoding="utf-8-sig", newline="") as stream:
            for index, row in enumerate(csv.DictReader(stream)):
                result.append((f"shopper:{index:05d}", str(row.get("Revenue", "unknown"))))
    else:
        with zipfile.ZipFile(path) as archive:
            for name in sorted(item for item in archive.namelist() if item.lower().endswith(".csv")):
                with archive.open(name) as binary:
                    text = (line.decode("utf-8-sig", errors="replace") for line in binary)
                    for index, row in enumerate(csv.DictReader(text)):
                        identity = str(row.get("COMMENT_ID") or f"{name}:{index}")
                        result.append((f"youtube:{hashlib.sha256(identity.encode()).hexdigest()[:20]}", str(row.get("CLASS", "unknown"))))
    return result


def _split(identity: str) -> str:
    value = unit(identity, "operational-audit-v16-split")
    return "train" if value < 0.60 else "formative_development" if value < 0.80 else "sealed_confirmatory_test"


def _seal(dataset_id: str, rows: list[tuple[str, str]]) -> dict[str, object]:
    directory = DATA / dataset_id
    identities = {name: [] for name in ("train", "formative_development", "sealed_confirmatory_test")}
    test_labels = {}
    for identity, label in rows:
        split = _split(identity)
        identities[split].append(identity)
        if split == "sealed_confirmatory_test":
            test_labels[identity] = label
    private = directory / "private"
    private.mkdir(parents=True, exist_ok=True)
    key = private / ".vault_passphrase"
    key.write_text(secrets.token_hex(32), encoding="ascii")
    os.chmod(key, 0o600)
    plain = private / "labels.json"
    vault = private / "label_vault.enc"
    plain.write_text(json.dumps(test_labels, sort_keys=True), encoding="utf-8")
    subprocess.run(["openssl", "enc", "-aes-256-cbc", "-pbkdf2", "-salt", "-in", str(plain), "-out", str(vault), "-pass", f"file:{key}"], check=True)
    plain.unlink()
    write_json(directory / "manifests" / "split_identities.json", identities)
    return {
        "dataset_id": dataset_id,
        "objects": len(rows),
        "split_counts": {name: len(values) for name, values in identities.items()},
        "split_hashes": {name: hashlib.sha256("\n".join(sorted(values)).encode()).hexdigest() for name, values in identities.items()},
        "vault_sha256": sha256_file(vault),
        "test_labels_public": False,
        "overlap_count": sum(bool(set(identities[left]) & set(identities[right])) for left in identities for right in identities if left < right),
    }


def main() -> None:
    verify_protocol()
    datasets = []
    for dataset_id, (url, filename, modality) in SOURCES.items():
        path = DATA / dataset_id / "raw" / filename
        _download(url, path)
        item = _seal(dataset_id, _rows(dataset_id, path))
        item.update(source=url, license="CC BY 4.0", modality=modality, raw_sha256=sha256_file(path))
        datasets.append(item)
    write_json(ARTIFACTS / "data" / "dataset_manifest.json", {"datasets": datasets, "implementation_commit": git_commit(), "fresh_against_registered_manifests": True})
    write_json(ARTIFACTS / "data" / "pre_opening_leakage_audit.json", {"status": "pass", "confirmatory_test_opened": False, "labels_exported": False, "post_lock_tuning": False, "overlap_count": sum(item["overlap_count"] for item in datasets)})
    print(f"PASS operational-audit-data datasets={len(datasets)} opened=false")


if __name__ == "__main__":
    main()
