from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import time
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config"
ARTIFACTS = ROOT / "artifacts" / "chapter4_v13"
PROTOCOL_PATH = CONFIG / "chapter4_v13_protocol.yaml"
PROTOCOL_HASH_PATH = CONFIG / "chapter4_v13_protocol.sha256"
RUNTIME_PATH = CONFIG / "chapter4_v13_runtime.yaml"


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected mapping in {path}")
    return value


def protocol() -> dict[str, Any]:
    verify_protocol_hash()
    return load_yaml(PROTOCOL_PATH)


def runtime_config() -> dict[str, Any]:
    return load_yaml(RUNTIME_PATH)["runtime"]


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(canonical_bytes(dict(row)).decode("utf-8") + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"non-object JSONL row in {path}")
                yield value


def verify_protocol_hash() -> str:
    expected, relative = PROTOCOL_HASH_PATH.read_text(encoding="utf-8").strip().split(maxsplit=1)
    if relative != "config/chapter4_v13_protocol.yaml":
        raise RuntimeError("protocol hash file points to an unexpected path")
    actual = sha256_file(PROTOCOL_PATH)
    if actual != expected:
        raise RuntimeError(f"protocol changed after lock: expected {expected}, got {actual}")
    return actual


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def ensure_dirs() -> None:
    for relative in (
        "raw",
        "processed",
        "private",
        "predictions",
        "explanations",
        "policies",
        "route_faults",
        "runtime",
        "end_to_end_case",
        "tables",
        "figures",
        "manifests",
    ):
        (ARTIFACTS / relative).mkdir(parents=True, exist_ok=True)


@contextmanager
def measured_stage(name: str) -> Iterator[dict[str, float | int | str]]:
    import psutil

    process = psutil.Process(os.getpid())
    rss_before = process.memory_info().rss
    start = time.perf_counter_ns()
    record: dict[str, float | int | str] = {"stage": name}
    try:
        yield record
    finally:
        elapsed = (time.perf_counter_ns() - start) / 1e9
        record.update(
            elapsed_seconds=elapsed,
            rss_before_bytes=rss_before,
            rss_after_bytes=process.memory_info().rss,
            rss_peak_proxy_bytes=max(rss_before, process.memory_info().rss),
        )
        try:
            import torch

            if torch.cuda.is_available():
                record["peak_vram_bytes"] = int(torch.cuda.max_memory_allocated())
        except ImportError:
            record["peak_vram_bytes"] = 0


def environment_manifest() -> dict[str, object]:
    processor = platform.processor()
    if not processor:
        try:
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
                if line.lower().startswith("model name"):
                    processor = line.split(":", maxsplit=1)[1].strip()
                    break
        except OSError:
            processor = "unknown"
    manifest: dict[str, object] = {
        "protocol_sha256": verify_protocol_hash(),
        "commit": git_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": processor or "unknown",
        "cpu_count": os.cpu_count(),
    }
    try:
        import importlib.metadata as metadata

        packages = {}
        for name in ("numpy", "pandas", "pyarrow", "scikit-learn", "scipy", "torch", "transformers", "psutil"):
            try:
                packages[name] = metadata.version(name)
            except metadata.PackageNotFoundError:
                packages[name] = None
        manifest["packages"] = packages
        try:
            import psutil

            manifest["ram_total_bytes"] = int(psutil.virtual_memory().total)
        except ImportError:
            manifest["ram_total_bytes"] = None
    except ImportError:
        pass
    try:
        import torch

        manifest["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            manifest["cuda_version"] = torch.version.cuda
            manifest["gpu"] = torch.cuda.get_device_name(0)
            manifest["gpu_total_memory_bytes"] = torch.cuda.get_device_properties(0).total_memory
    except ImportError:
        manifest["cuda_available"] = False
    return manifest
