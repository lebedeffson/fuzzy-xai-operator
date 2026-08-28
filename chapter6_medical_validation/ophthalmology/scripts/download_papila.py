"""Download and verify PAPILA exclusively through the official Figshare API."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import urllib.request
from pathlib import Path


ARTICLE_ID = 14798004
API_URL = f"https://api.figshare.com/v2/articles/{ARTICLE_ID}"


def digest(path: Path, algorithm: str) -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def data_root(explicit: Path | None) -> Path:
    root = explicit or Path(os.environ["FUZZYXAI_CH6_DATA_ROOT"]) / "eyes" / "papila"
    root.mkdir(parents=True, exist_ok=True)
    return root


def retrieve(url: str, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".partial")
    with urllib.request.urlopen(url) as source, temporary.open("wb") as target:
        shutil.copyfileobj(source, target, length=1024 * 1024)
    temporary.replace(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--extract", action="store_true", help="extract PAPILA.zip after checksum verification")
    args = parser.parse_args()
    root = data_root(args.data_root)
    with urllib.request.urlopen(API_URL) as response:
        metadata = json.load(response)
    if int(metadata["id"]) != ARTICLE_ID or int(metadata["version"]) != 2:
        raise RuntimeError(f"unexpected PAPILA Figshare article/version: {metadata['id']}/{metadata.get('version')}")
    (root / "figshare_article_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    inventory: list[dict[str, object]] = []
    for remote in metadata["files"]:
        target = root / str(remote["name"])
        expected_size = int(remote["size"])
        expected_md5 = str(remote.get("supplied_md5") or "")
        if not target.is_file() or target.stat().st_size != expected_size:
            retrieve(str(remote["download_url"]), target)
        actual_md5 = digest(target, "md5")
        if expected_md5 and actual_md5 != expected_md5:
            raise RuntimeError(f"Figshare MD5 mismatch for {target.name}: {actual_md5} != {expected_md5}")
        inventory.append({"name": target.name, "size": target.stat().st_size, "figshare_md5": expected_md5, "actual_md5": actual_md5, "sha256": digest(target, "sha256")})
    (root / "papila_download_inventory.json").write_text(json.dumps({"article_id": ARTICLE_ID, "version": metadata["version"], "license": metadata.get("license"), "files": inventory}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.extract:
        archive = root / "PAPILA.zip"
        destination = root / "raw"
        if not destination.is_dir():
            shutil.unpack_archive(archive, destination)
    print(json.dumps({"article_id": ARTICLE_ID, "version": metadata["version"], "license": metadata.get("license", {}).get("name"), "files": inventory}, ensure_ascii=False))


if __name__ == "__main__":
    main()
