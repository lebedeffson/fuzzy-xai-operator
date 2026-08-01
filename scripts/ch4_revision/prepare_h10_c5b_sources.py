#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
from pathlib import Path

import pandas as pd

PROTOCOL_ID = "h10-c5b-repository-grounded-v1"
SOURCES = {
    "development": {
        "url": (
            "https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite/"
            "resolve/main/data/dev-00000-of-00001.parquet"
        ),
        "sha256": "8312f321838051849d2fa7c6ca071244733a3e90bb517ff6ca8186f722199b5c",
    },
    "held_out": {
        "url": (
            "https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite/"
            "resolve/main/data/test-00000-of-00001.parquet"
        ),
        "sha256": "7a21f37b8bc179c7db5beeb14e88ac538ba283455c776e6b2535bbfb6e3551b4",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, output: Path, expected: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.is_file() or _sha256(output) != expected:
        temporary = output.with_suffix(output.suffix + ".part")
        with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as stream:
            shutil.copyfileobj(response, stream)
        temporary.replace(output)
    actual = _sha256(output)
    if actual != expected:
        raise ValueError(f"source checksum mismatch for {output}: {actual}")


def _parse_sequence(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return (value,) if value else ()
    if isinstance(value, (tuple, list)):
        return tuple(str(item) for item in value)
    return (str(value),)


def _rank(instance_id: str) -> str:
    return hashlib.sha256(f"{PROTOCOL_ID}\0{instance_id}".encode()).hexdigest()


def select_rows(frame: pd.DataFrame, per_repository: int) -> tuple[dict[str, object], ...]:
    rows = frame.to_dict(orient="records")
    repositories = sorted({str(row["repo"]) for row in rows})
    selected = []
    for repository in repositories:
        candidates = sorted(
            (row for row in rows if str(row["repo"]) == repository),
            key=lambda row: (_rank(str(row["instance_id"])), str(row["instance_id"])),
        )
        selected.extend(candidates[:per_repository])
    return tuple(selected)


def _run(arguments: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        arguments,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )


def _repository_cache(repo: str, cache_root: Path) -> Path:
    output = cache_root / repo.replace("/", "__")
    if not output.is_dir():
        output.parent.mkdir(parents=True, exist_ok=True)
        _run(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                "--quiet",
                f"https://github.com/{repo}.git",
                str(output),
            ]
        )
    return output


def _ensure_commit(repository: Path, commit: str) -> None:
    result = subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
        cwd=repository,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        _run(["git", "fetch", "--quiet", "--depth=1", "origin", commit], cwd=repository)


def _safe_extract_git_archive(repository: Path, commit: str, output: Path) -> None:
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    process = subprocess.Popen(
        ["git", "archive", "--format=tar", commit],
        cwd=repository,
        stdout=subprocess.PIPE,
    )
    assert process.stdout is not None
    with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
        archive.extractall(output, filter="data")
    if process.wait() != 0:
        raise subprocess.CalledProcessError(process.returncode, process.args)


def _changed_files(patch: str) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            right
            for _left, right in re.findall(
                r"^diff --git a/(.+?) b/(.+?)$",
                patch,
                flags=re.MULTILINE,
            )
            if right != "/dev/null"
        )
    )


def _show(repository: Path, commit: str, file_path: str) -> str:
    result = subprocess.run(
        ["git", "show", f"{commit}:{file_path}"],
        cwd=repository,
        capture_output=True,
        check=False,
    )
    return result.stdout.decode("utf-8", errors="replace") if result.returncode == 0 else ""


def _gold_snapshots(
    repository: Path,
    commit: str,
    patch: str,
) -> tuple[dict[str, str], dict[str, str]]:
    changed = _changed_files(patch)
    before = {path: _show(repository, commit, path) for path in changed}
    with tempfile.TemporaryDirectory(prefix="fuzzyxai-h10-c5b-gold-") as temporary:
        root = Path(temporary)
        for path, content in before.items():
            target = root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        patch_path = root / "gold.patch"
        patch_path.write_text(patch, encoding="utf-8")
        result = subprocess.run(
            ["git", "apply", "--unsafe-paths", str(patch_path)],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode:
            raise ValueError(f"patch application failed: {result.stderr.strip()}")
        after = {
            path: (root / path).read_text(encoding="utf-8", errors="replace")
            if (root / path).is_file()
            else ""
            for path in changed
        }
    return before, after


def materialize(
    row: dict[str, object],
    *,
    split: str,
    output_root: Path,
    cache_root: Path,
) -> dict[str, object]:
    repo = str(row["repo"])
    commit = str(row["base_commit"])
    incident_id = str(row["instance_id"])
    repository = _repository_cache(repo, cache_root)
    _ensure_commit(repository, commit)
    incident_root = output_root / "incidents" / split / incident_id
    buggy = incident_root / "buggy"
    _safe_extract_git_archive(repository, commit, buggy)
    patch = str(row["patch"])
    before, after = _gold_snapshots(repository, commit, patch)
    incident_root.mkdir(parents=True, exist_ok=True)
    (incident_root / "fix.patch").write_text(patch, encoding="utf-8")
    (incident_root / "before_sources.json").write_text(
        json.dumps(before, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (incident_root / "after_sources.json").write_text(
        json.dumps(after, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "incident_id": incident_id,
        "repository": repo,
        "buggy_revision": commit,
        "repository_root": str(buggy.resolve()),
        "failing_tests": _parse_sequence(row.get("FAIL_TO_PASS")),
        "split": split,
        "patch_path": str((incident_root / "fix.patch").resolve()),
        "before_sources_path": str((incident_root / "before_sources.json").resolve()),
        "after_sources_path": str((incident_root / "after_sources.json").resolve()),
        "runtime_evidence_status": "FAILING_TEST_IDS_ONLY",
        "selection_rank_sha256": _rank(incident_id),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path("/tmp/h10c5b-git-cache"))
    parser.add_argument("--development-per-repository", type=int, default=1)
    parser.add_argument("--held-out-per-repository", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    output = args.output.resolve()
    source_root = output / "sources"
    selected: list[tuple[str, dict[str, object]]] = []
    source_registry = []
    for split, details in SOURCES.items():
        path = source_root / f"{split}.parquet"
        _download(str(details["url"]), path, str(details["sha256"]))
        frame = pd.read_parquet(path)
        count = (
            args.development_per_repository
            if split == "development"
            else args.held_out_per_repository
        )
        selected.extend((split, row) for row in select_rows(frame, count))
        source_registry.append(
            {
                "split": split,
                "url": details["url"],
                "sha256": _sha256(path),
                "rows": len(frame),
                "repositories": int(frame["repo"].nunique()),
            }
        )
    development = {str(row["repo"]) for split, row in selected if split == "development"}
    held_out = {str(row["repo"]) for split, row in selected if split == "held_out"}
    if development & held_out:
        raise ValueError("development and held-out repositories overlap")
    if args.limit:
        selected = selected[: args.limit]
    manifest = [
        materialize(
            row,
            split=split,
            output_root=output,
            cache_root=args.cache.resolve(),
        )
        for split, row in selected
    ]
    manifest_paths = {}
    for name, selected_rows in (
        ("H10_C5B_INCIDENT_MANIFEST", manifest),
        (
            "H10_C5B_DEVELOPMENT_MANIFEST",
            [row for row in manifest if row["split"] == "development"],
        ),
        (
            "H10_C5B_HELD_OUT_UNSCORED_MANIFEST",
            [row for row in manifest if row["split"] == "held_out"],
        ),
    ):
        manifest_path = output / f"{name}.jsonl"
        manifest_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in selected_rows),
            encoding="utf-8",
        )
        manifest_paths[name] = {
            "path": str(manifest_path),
            "rows": len(selected_rows),
            "sha256": _sha256(manifest_path),
        }
    (output / "SOURCE_AND_SELECTION_REGISTRY.json").write_text(
        json.dumps(
            {
                "protocol_id": PROTOCOL_ID,
                "selection": (
                    "lowest SHA256(protocol_id + NUL + instance_id) within "
                    "each repository; no patch-derived selection"
                ),
                "sources": source_registry,
                "development_repositories": sorted(development),
                "held_out_repositories": sorted(held_out),
                "repository_overlap": [],
                "materialized_incidents": len(manifest),
                "manifests": manifest_paths,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "manifests": manifest_paths,
                "incident_count": len(manifest),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
