from __future__ import annotations

import csv
import hashlib
import json
import urllib.request
from pathlib import Path

from fuzzyxai.audit.common import ROOT, current_commit


OUT = ROOT / "data" / "real_public"
REPORTS = ROOT / "reports" / "real_validation"

IRIS_URL = "https://commons.wikimedia.org/wiki/Special:Redirect/file/Close-up_Image_of_a_Human_Iris.jpg"
ECG_HEA_URL = "https://physionet.org/files/mitdb/1.0.0/100.hea"
ECG_DAT_URL = "https://physionet.org/files/mitdb/1.0.0/100.dat"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "FuzzyXAI real validation"})
    with urllib.request.urlopen(req, timeout=30) as response:
        path.write_bytes(response.read())


def jpeg_size(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    i = 2
    while i < len(data):
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        size = int.from_bytes(data[i + 2 : i + 4], "big")
        if marker in {0xC0, 0xC2}:
            h = int.from_bytes(data[i + 5 : i + 7], "big")
            w = int.from_bytes(data[i + 7 : i + 9], "big")
            return w, h
        i += 2 + size
    raise ValueError(f"Cannot read JPEG size: {path}")


def parse_wfdb_212(dat_path: Path, n_samples: int = 3600) -> list[tuple[int, int]]:
    raw = dat_path.read_bytes()
    out: list[tuple[int, int]] = []
    for i in range(0, min(len(raw) - 2, n_samples // 2 * 3), 3):
        b0, b1, b2 = raw[i], raw[i + 1], raw[i + 2]
        s1 = b0 | ((b1 & 0x0F) << 8)
        s2 = b2 | ((b1 & 0xF0) << 4)
        if s1 & 0x800:
            s1 -= 0x1000
        if s2 & 0x800:
            s2 -= 0x1000
        out.append((s1, s2))
    return out[: n_samples // 2]


def build_ecg_csv(dat_path: Path, csv_path: Path) -> None:
    pairs = parse_wfdb_212(dat_path, n_samples=3600)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["sample", "time_sec", "mlII", "v5"])
        sample = 0
        for s1, s2 in pairs:
            writer.writerow([sample, round(sample / 360, 6), s1, s2])
            sample += 1
            writer.writerow([sample, round(sample / 360, 6), s2, s1])
            sample += 1


def main() -> None:
    iris = OUT / "iris" / "close_up_human_iris.jpg"
    hea = OUT / "ecg" / "mitdb_100.hea"
    dat = OUT / "ecg" / "mitdb_100.dat"
    ecg_csv = OUT / "ecg" / "mitdb_100_first_10s.csv"
    download(IRIS_URL, iris)
    download(ECG_HEA_URL, hea)
    download(ECG_DAT_URL, dat)
    build_ecg_csv(dat, ecg_csv)
    w, h = jpeg_size(iris)
    manifest = {
        "source_commit": current_commit(),
        "artifacts": {
            "iris": {
                "status": "real_public_artifact",
                "source": "Wikimedia Commons",
                "url": IRIS_URL,
                "path": str(iris.relative_to(ROOT)),
                "sha256": sha256(iris),
                "width": w,
                "height": h,
            },
            "ecg": {
                "status": "real_public_artifact",
                "source": "PhysioNet MIT-BIH Arrhythmia Database record 100",
                "url_header": ECG_HEA_URL,
                "url_signal": ECG_DAT_URL,
                "header_path": str(hea.relative_to(ROOT)),
                "signal_path": str(dat.relative_to(ROOT)),
                "csv_path": str(ecg_csv.relative_to(ROOT)),
                "header_sha256": sha256(hea),
                "signal_sha256": sha256(dat),
                "csv_sha256": sha256(ecg_csv),
                "sampling_rate": 360,
            },
        },
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "real_artifacts_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(REPORTS / "real_artifacts_manifest.json")


if __name__ == "__main__":
    main()

