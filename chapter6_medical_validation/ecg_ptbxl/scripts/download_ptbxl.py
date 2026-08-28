from __future__ import annotations

import argparse
import os
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = "https://physionet.org/files/ptb-xl/1.0.3/"
METADATA = ("LICENSE.txt", "RECORDS", "SHA256SUMS.txt", "ptbxl_database.csv", "scp_statements.csv", "ptbxl_v103_changelog.txt")


def links(url: str) -> list[str]:
    html = urllib.request.urlopen(url, timeout=60).read().decode("utf-8")
    return [value for value in re.findall(r'href="([^"]+)"', html) if value not in {"../", "./"}]


def download(url: str, target: Path, retries: int = 6) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_size > 0:
        return
    temporary = target.with_suffix(target.suffix + ".part")
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as stream:
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
            temporary.replace(target)
            return
        except (OSError, TimeoutError, urllib.error.URLError):
            if attempt + 1 == retries:
                raise
            time.sleep(2**attempt)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download only official PTB-XL 1.0.3 records100")
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    data_root = os.environ.get("FUZZYXAI_CH6_DATA_ROOT")
    if not data_root:
        raise FileNotFoundError("FUZZYXAI_CH6_DATA_ROOT is not set")
    root = Path(data_root) / "ecg" / "ptb-xl-1.0.3"
    for name in METADATA:
        download(BASE + name, root / name)
    jobs: list[tuple[str, Path]] = []
    records_url = BASE + "records100/"
    for directory in (value for value in links(records_url) if value.endswith("/")):
        for filename in links(records_url + directory):
            if filename.endswith((".dat", ".hea")):
                jobs.append((records_url + directory + filename, root / "records100" / directory / filename))
    print(f"registered {len(jobs)} waveform files", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for index, _ in enumerate(pool.map(lambda item: download(*item), jobs), 1):
            if index % 1000 == 0:
                print(f"downloaded/verified {index}/{len(jobs)}", flush=True)
    print(root)


if __name__ == "__main__":
    main()
