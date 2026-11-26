"""Verify and prepare the original public dataset in an isolated data root."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path, PurePosixPath

import pandas as pd
from chapter6_medical_validation.shared.hashing import sha256_file, sha256_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.data_root / "ecg" / "ptb-xl-1.0.3"
    root.mkdir(parents=True, exist_ok=False)
    metadata = {"LICENSE.txt", "SHA256SUMS.txt", "ptbxl_database.csv", "scp_statements.csv"}
    with zipfile.ZipFile(args.zip) as archive:
        written = set()
        for info in archive.infolist():
            if info.is_dir():
                continue
            parts = PurePosixPath(info.filename).parts
            if "records100" in parts:
                rel = PurePosixPath(*parts[parts.index("records100"):])
                if len(rel.parts) != 3 or rel.suffix not in {".dat", ".hea"}:
                    continue
            elif parts[-1] in metadata:
                rel = PurePosixPath(parts[-1])
            else:
                continue
            if any(p in {"..", ".", "/"} for p in rel.parts) or str(rel) in written:
                raise ValueError(f"Unsafe or duplicate archive member: {rel}")
            target = root / str(rel)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            written.add(str(rel))
    expected = {}
    for line in (root / "SHA256SUMS.txt").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        expected[name.lstrip("* ")] = digest
    expected_records = {p for p in expected if p.startswith("records100/") and p.endswith((".dat", ".hea"))}
    actual_records = {p for p in written if p.startswith("records100/")}
    assert expected_records == actual_records and len(actual_records) == 2 * 21799
    for name in ("ptbxl_database.csv", "scp_statements.csv", "LICENSE.txt"):
        assert expected[name] == sha256_file(root / name), name
    env = dict(os.environ, FUZZYXAI_CH6_DATA_ROOT=str(args.data_root))
    subprocess.run([sys.executable, "-m", "chapter6_medical_validation.ecg_ptbxl.scripts.prepare_ptbxl", "--verify-sha"], env=env, check=True)
    prepared = root / "prepared"
    manifest = json.loads((prepared / "dataset_manifest.json").read_text())
    assert manifest["counts"]["included_primary"] == 20943
    assert manifest["counts"]["test"] == 2107
    assert manifest["counts"]["validation"] == 2105
    # The previous manifest hard-coded the wrong license name; retain the
    # official LICENSE and attribution rather than carrying that error forward.
    manifest["license"] = "CC BY 4.0, see official LICENSE.txt"
    manifest["source_archive_sha256"] = sha256_file(args.zip)
    manifest["all_43598_records100_files_verified"] = True
    manifest["source_commit"] = os.environ.get("GITHUB_SHA", "local")
    manifest.pop("manifest_sha256", None)
    manifest["manifest_sha256"] = sha256_json(manifest)
    (prepared / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    shutil.copyfile(root / "LICENSE.txt", prepared / "PTBXL_LICENSE.txt")
    rows = pd.read_csv(prepared / "label_construction.csv")
    refs = {}
    for row in rows.loc[rows["included"]].itertuples():
        checksums = {ext: expected[str(row.filename_lr) + ext] for ext in (".dat", ".hea")}
        refs[str(int(row.ecg_id))] = {"filename_lr": str(row.filename_lr), "files": checksums, "source_sha256": sha256_json(checksums)}
    assert len(refs) == 20943
    (prepared / "source_pairs.json").write_text(json.dumps(refs, separators=(",", ":")) + "\n")
    print(json.dumps(manifest["counts"], indent=2), flush=True)


if __name__ == "__main__":
    main()
