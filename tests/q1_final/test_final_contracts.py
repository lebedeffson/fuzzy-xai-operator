from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from fuzzyxai.q1_final.contracts import ExternalGateRecord, FinalRunIdentity
from fuzzyxai.q1_final.multiclass import _stratified_cap
from scripts.q1_final import build_archives


COMMIT = "1" * 40
ROOT = build_archives.ROOT


def test_stable_identity_rejects_open_external_gate() -> None:
    identity = FinalRunIdentity(
        schema_version="2.0",
        branch="feat/q1-final-closure",
        base_commit=COMMIT,
        final_commit=COMMIT,
        ci_run_ids=(),
        profile="full_q1_final",
        real_benchmark_status="pass",
        external_gate_status={"comprehension": "open"},
        stable_release_allowed=False,
        created_at="2026-07-20T00:00:00Z",
        python="3.12",
        platform="linux",
        threads=1,
    )
    with pytest.raises(ValueError, match="external gate"):
        replace(identity, stable_release_allowed=True)


def test_external_gate_cannot_close_without_real_records() -> None:
    with pytest.raises(ValueError, match="raw, signed and scored"):
        ExternalGateRecord(
            gate_id="comprehension",
            status="supported",
            required_count=24,
            observed_count=24,
            ethics_status="approved",
            raw_records=(),
            signed_records=(),
            scorer_output=None,
            claim_removed_if_not_supported=False,
        )


def test_stratified_cap_accepts_composite_string_strata() -> None:
    indices = np.arange(60)
    labels = np.asarray([f"{index % 3}:{index % 2}:{index % 4}" for index in indices])
    selected = _stratified_cap(indices, labels, 24, 4201)
    assert len(selected) == 24
    assert len(set(selected.tolist())) == 24


def test_runtime_archive_inputs_fail_closed(tmp_path: object, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(build_archives, "ROOT", tmp_path)
    with pytest.raises(RuntimeError, match="required runtime archive input is missing"):
        build_archives.runtime_paths(("release_evidence/q1_final/dod_185.json",))


def test_final_docker_context_keeps_git_identity() -> None:
    ignored = {
        line.strip()
        for line in (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    assert ".git" not in ignored
