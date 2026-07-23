from __future__ import annotations

import csv
import importlib
from pathlib import Path

import pytest

from h10_c2 import cli
from h10_c2.adjudication.export_blind_cases import FIELDS
from h10_c2.hashing import file_sha256, write_json
from h10_c2.paths import ARTIFACT_ROOT, DELIVERABLE_ROOT
from h10_c2.reporting import build_deliverables
from h10_c2.sealing.seal import seal_current_inputs
from h10_c2.sealing.scoring_gate import freeze_protocol


@pytest.mark.parametrize(
    ("arguments", "symbol"),
    [
        (["bootstrap"], "bootstrap"),
        (["power"], "run_power"),
        (["generate", "--split", "development"], "generate"),
        (["run", "--split", "development"], "run_split"),
        (["audit-baselines"], "audit_baselines"),
        (["audit-oracle"], "audit_oracle"),
        (["leakage-audit"], "run_leakage_audit"),
        (["export-adjudication"], "export_blind_cases"),
        (["preconfirmatory-gate"], "preconfirmatory_gate"),
        (["package"], "build_deliverables"),
    ],
)
def test_cli_dispatches_nonsealed_commands(monkeypatch: pytest.MonkeyPatch, arguments: list[str], symbol: str) -> None:
    if symbol == "preconfirmatory_gate":
        monkeypatch.setattr(cli, symbol, lambda: {"status": "READY_FOR_SEALED_SCORING"})
    elif symbol == "run_split":
        monkeypatch.setattr(cli, symbol, lambda split: Path(f"{split}.csv"))
        monkeypatch.setattr(cli, "build_nonconfirmatory_statistics", lambda split: [])
    else:
        monkeypatch.setattr(cli, symbol, lambda *args, **kwargs: {"status": "PASS"})
    assert cli.main(arguments) == 0


def test_cli_dispatches_import_freeze_and_score(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    reviewer = tmp_path / "reviewer.csv"
    approval = tmp_path / "approval.json"
    lock = tmp_path / "lock.json"
    for path in (reviewer, approval, lock):
        path.write_text("x", encoding="utf-8")
    monkeypatch.setattr(cli, "import_results", lambda *args: {"status": "PASS"})
    assert cli.main(["import-adjudication", "--reviewer-1", str(reviewer), "--reviewer-2", str(reviewer)]) == 0
    monkeypatch.setattr(cli, "freeze_protocol", lambda *args: {"status": "PASS"})
    assert cli.main(["freeze-protocol", "--approval", str(approval)]) == 0
    monkeypatch.setattr(cli, "score_sealed", lambda *args: None)
    assert cli.main(["score-sealed", "--lock", str(lock), "--approval", str(approval)]) == 0
    monkeypatch.setattr(cli, "bootstrap", lambda: (_ for _ in ()).throw(ValueError("expected failure")))
    assert cli.main(["bootstrap"]) == 2


def test_generate_wrapper_seals_only_sealed_split(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr(cli, "generate_from_design", lambda split, seed_offset: {"split": split, "seed": seed_offset})
    monkeypatch.setattr(cli, "seal_current_inputs", lambda: calls.append("sealed"))
    assert cli.generate("development")["seed"] == 1
    assert calls == []
    assert cli.generate("sealed")["seed"] == 3
    assert calls == ["sealed"]


def test_score_sealed_complete_path_is_isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    lock = tmp_path / "lock.json"
    approval = tmp_path / "approval.json"
    write_json(lock, {"protocol_sha256": "a" * 64, "method_tree_sha256": "ignored"})
    approval.write_text("approval", encoding="utf-8")
    monkeypatch.setattr(cli, "ARTIFACT_ROOT", tmp_path)
    monkeypatch.setattr(cli, "preconfirmatory_gate", lambda: {"status": "READY_FOR_SEALED_SCORING"})
    monkeypatch.setattr(
        cli,
        "validate_approval",
        lambda *args: {"approved_by": "external-board"},
    )
    monkeypatch.setattr(cli, "record_opening", lambda *args, **kwargs: {"opening_count": 1})
    monkeypatch.setattr(cli, "run_split", lambda split: Path(f"{split}.csv"))
    monkeypatch.setattr(
        cli,
        "build_nonconfirmatory_statistics",
        lambda split: [
            {"claim": "H10-C2a", "effect": 0.1, "ci_low": 0.01, "ci_high": 0.2, "p_holm": 0.04},
            {"claim": "H10-C2b", "effect": 0.0, "ci_low": -0.1, "ci_high": 0.1, "p_holm": 1.0},
        ],
    )
    hashing = importlib.import_module("h10_c2.hashing")
    monkeypatch.setattr(hashing, "tree_sha256", lambda *args: "ignored")
    cli.score_sealed(lock, approval)
    claims = (tmp_path / "sealed" / "claim_registry.json").read_text(encoding="utf-8")
    assert '"supported"' in claims
    assert '"not_supported"' in claims


def test_score_sealed_refuses_failed_gate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(cli, "preconfirmatory_gate", lambda: {"status": "BLOCKED_POWER", "blockers": ["power"]})
    with pytest.raises(PermissionError, match="SCORING_BLOCKED"):
        cli.score_sealed(tmp_path / "lock", tmp_path / "approval")


def _write_completed_form(path: Path, signature: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerow(
            {
                "case_id": "case-1",
                "mutation_log_consistent": "yes",
                "optimal_cuts_valid": "yes",
                "repair_actions_valid": "yes",
                "additional_valid_variants_json": "[]",
                "ambiguous": "no",
                "sufficient_evidence": "yes",
                "comments": "checked",
                "reviewer_signature": signature,
            }
        )


def test_reviewer_import_success_is_isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = importlib.import_module("h10_c2.adjudication.import_reviewer_results")
    monkeypatch.setattr(module, "ARTIFACT_ROOT", tmp_path)
    first, second = tmp_path / "first.csv", tmp_path / "second.csv"
    _write_completed_form(first, "reviewer-one")
    _write_completed_form(second, "reviewer-two")
    report = module.import_results(first, second)
    assert report["status"] == "PASS"
    assert report["agreement"] == 1.0


def test_protocol_freeze_and_seal_are_isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    gate_module = importlib.import_module("h10_c2.sealing.scoring_gate")
    seal_module = importlib.import_module("h10_c2.sealing.seal")
    artifact_root = tmp_path / "artifacts"
    (artifact_root / "power").mkdir(parents=True)
    design = artifact_root / "power" / "recommended_design.json"
    write_json(design, {"status": "power_target_reached"})
    approval = tmp_path / "approval.json"
    write_json(
        approval,
        {
            "approved": True,
            "approved_by": "protocol-owner",
            "signature": "external-signature",
            "recommended_design_sha256": file_sha256(design),
        },
    )
    monkeypatch.setattr(gate_module, "ARTIFACT_ROOT", artifact_root)
    assert freeze_protocol(approval)["opening_count"] == 0
    public = artifact_root / "data" / "sealed"
    private = artifact_root / "private" / "sealed"
    public.mkdir(parents=True)
    private.mkdir(parents=True)
    (public / "cases.jsonl").write_text("{}\n", encoding="utf-8")
    (private / "gold.jsonl").write_text("{}\n", encoding="utf-8")
    (private / "transactions.jsonl").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(seal_module, "ARTIFACT_ROOT", artifact_root)
    assert seal_current_inputs()["vault_release_policy"].startswith("private")


def test_delivery_packaging_runs_and_excludes_private_gold() -> None:
    result = build_deliverables()
    assert result["status"] == "BLOCKED_PRECONFIRMATORY"
    assert (DELIVERABLE_ROOT / "h10-c2-source.zip").is_file()
    assert (DELIVERABLE_ROOT / "h10-c2-validation-artifacts.zip").is_file()
    assert not (ARTIFACT_ROOT / "data" / "sealed").exists()
