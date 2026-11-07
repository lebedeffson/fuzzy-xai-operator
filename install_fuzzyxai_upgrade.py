"""Transactional preflight and installation of FuzzyXAI chapters 1-5 patches.

Extract this overlay into an existing clean fuzzy-xai-operator checkout. The
entire eight-file plan is validated before any original source is modified.
The operation is idempotent; on a mid-write failure, earlier writes are rolled
back. This script never creates git commits, touches DOCX manuscripts or
silently downgrades scientific validation gates.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import stat
import tempfile
import tomllib
from pathlib import Path
from typing import Callable

import apply_ch5_practice as ecg
import apply_doctoral_upgrade as doctoral

ROOT = Path(__file__).resolve().parent
REGISTRY = "framework/fuzzyxai/fuzzyxai/core/scenario_registry.py"
ADAPTER = "framework/fuzzyxai/fuzzyxai/adapters/signal_quality.py"


def transform_ecg_registry(source: str) -> str:
    pattern = re.compile(r"def build_medical_ecg_signal_route\(adapted: AdaptedInput\) -> OperatorRoute:\n.*?(?=def build_gd_anfis_shap_route\()", re.S)
    found = list(pattern.finditer(source))
    if len(found) != 1:
        raise RuntimeError("ECG scenario route changed; exactly one original function required")
    return source[:found[0].start()] + ecg.FUNCTION + source[found[0].end():]


def transform_ecg_adapter(source: str) -> str:
    marker = '    required_fields = ("sampling_rate", "quality_score", "noise_segments", "missing_fragments", "baseline_instability")\n'
    validation = '''\n    def validate(self, payload: dict) -> None:
        super().validate(payload)
        from fuzzyxai.ecg_quality import assess_ecg_quality

        assess_ecg_quality(payload)
'''
    if validation in source:
        return source
    if source.count(marker) != 1:
        raise RuntimeError("ECG adapter anchor changed; refusing to overwrite")
    return source.replace(marker, marker + validation, 1)


def _fingerprint(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _target_set() -> dict[str, Callable[[str], str]]:
    return {**doctoral.TARGETS, REGISTRY: transform_ecg_registry, ADAPTER: transform_ecg_adapter}


def stage(root: Path) -> tuple[dict[Path, bytes], dict[Path, bytes], dict[str, object]]:
    before: dict[Path, bytes] = {}
    changed: dict[Path, bytes] = {}
    rows: list[dict[str, object]] = []
    for name, transform in _target_set().items():
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"original repository file missing: {path}")
        existing = path.read_bytes()
        try:
            text = existing.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise RuntimeError(f"upstream source must be UTF-8: {name}") from exc
        patched = transform(text)
        if name.endswith(".toml"):
            tomllib.loads(patched)
        else:
            ast.parse(patched, filename=name)
        after = patched.encode("utf-8")
        before[path] = existing
        if existing != after:
            changed[path] = after
        rows.append({"path": name, "needs_update": existing != after,
                     "sha256_before": _fingerprint(existing), "sha256_after": _fingerprint(after)})
    receipt: dict[str, object] = {"schema": "fuzzyxai-upgrade-preflight-v5", "original_files_checked": len(before),
                                  "files_to_update": len(changed), "files": rows,
                                  "note": "source compatibility preflight only; not a regression test"}
    return before, changed, receipt


def _replace_bytes(path: Path, data: bytes, mode: int) -> None:
    # Named temp file in the same directory gives atomic replacement on a
    # normal filesystem. Preserve file mode; do not follow symlinks.
    if path.is_symlink():
        raise RuntimeError(f"refuse to overwrite symlink: {path}")
    tmp_name = None
    try:
        with tempfile.NamedTemporaryFile(prefix=f".{path.name}.upgrade-", dir=path.parent, delete=False) as fh:
            tmp_name = fh.name
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(tmp_name, mode)
        os.replace(tmp_name, path)
        tmp_name = None
    finally:
        if tmp_name is not None:
            try:
                os.unlink(tmp_name)
            except FileNotFoundError:
                pass


def apply(root: Path) -> dict[str, object]:
    before, changed, report = stage(root)
    updated: list[Path] = []
    modes = {path: stat.S_IMODE(path.stat().st_mode) for path in changed}
    try:
        for path, data in changed.items():
            _replace_bytes(path, data, modes[path])
            updated.append(path)
    except BaseException as exc:
        rollback_errors: list[str] = []
        for path in reversed(updated):
            try:
                _replace_bytes(path, before[path], modes[path])
            except OSError as rb_exc:
                rollback_errors.append(f"{path}: {rb_exc}")
        if rollback_errors:
            raise RuntimeError(f"upgrade failed and rollback incomplete: {rollback_errors}") from exc
        raise
    report["applied"] = [str(path.relative_to(root)) for path in updated]
    report["status"] = "applied" if updated else "already_applied"
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="preflight all eight existing-file edits; no writes")
    parser.add_argument("--root", type=Path, default=ROOT, help="full Git checkout root")
    parser.add_argument("--receipt", type=Path, help="optional report destination (written only on successful apply)")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    receipt = stage(root)[2] if args.check else apply(root)
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2))
    if args.receipt and not args.check:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
