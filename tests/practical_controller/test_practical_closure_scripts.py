from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _run(script: str):
    return subprocess.run(
        [sys.executable, script],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_formative_evidence_verifies_and_preserves_boundaries() -> None:
    result = _run("scripts/final_practical_closure/verify_formative.py")
    assert result.returncode == 0, result.stderr + result.stdout
    manifest = json.loads(
        (ROOT / "release_evidence/final_practical_closure/formative/manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["confirmatory_test_opened"] is False
    assert manifest["confirmatory_claim_allowed"] is False
    assert manifest["experiment_count"] == 6


def test_lock_and_confirmatory_runner_fail_closed_without_external_inputs() -> None:
    lock = ROOT / "study/final_practical_closure/confirmatory_protocol_lock.json"
    assert not lock.exists()
    freeze = _run("scripts/final_practical_closure/lock_protocol.py")
    assert freeze.returncode == 2
    assert "BLOCKED: practical_confirmatory_protocol_lock" in freeze.stdout
    run = _run("scripts/final_practical_closure/run_confirmatory.py")
    assert run.returncode != 0
    assert "BLOCKED: confirmatory protocol is not locked" in run.stderr


def test_claim_registry_scopes_external_gates_without_claiming_completion() -> None:
    result = _run("scripts/final_practical_closure/build_claim_registry.py")
    assert result.returncode == 0, result.stderr + result.stdout
    registry = json.loads(
        (ROOT / "release_evidence/final_practical_closure/claim_registry.json").read_text(encoding="utf-8")
    )
    assert registry["confirmatory_run_completed"] is False
    assert registry["technical_release_allowed"] is False
    assert registry["external_claims"]["technical_release_blocked"] is False
    assert all(claim["status"] == "not_run" for claim in registry["new_claims"])
    assert registry["immutable_original_results"]["H3-original"] == "not_supported"
