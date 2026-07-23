from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .adjudication import export_blind_cases, import_results
from .audit import audit_baselines, audit_oracle, run_leakage_audit
from .hashing import file_sha256, read_json, write_json
from .paths import ARTIFACT_ROOT, PACKAGE_ROOT, REPO_ROOT
from .reporting import build_deliverables
from .runner import build_nonconfirmatory_statistics, generate_from_design, run_split
from .sealing import freeze_protocol, preconfirmatory_gate, seal_current_inputs
from .sealing.open_guard import validate_approval
from .sealing.opening_counter import initialize, record_opening
from .statistics import run_power


def bootstrap() -> dict:
    for name in ("power", "data", "results", "lock", "sealed", "audit", "adjudication", "private"):
        (ARTIFACT_ROOT / name).mkdir(parents=True, exist_ok=True)
    initialize(ARTIFACT_ROOT / "sealed" / "opening_record.json")
    approval = ARTIFACT_ROOT / "lock" / "design_approval_template.json"
    if not approval.exists():
        write_json(
            approval,
            {
                "approved": False,
                "approved_by": "",
                "signature": "",
                "recommended_design_sha256": "",
                "note": "External protocol owner must approve the computed design.",
            },
        )
    return {"status": "PASS", "artifact_root": str(ARTIFACT_ROOT)}


def generate(split: str) -> dict:
    offsets = {"development": 1, "protocol_validation": 2, "sealed": 3}
    value = generate_from_design(split, seed_offset=offsets[split])
    if split == "sealed":
        seal_current_inputs()
    return value


def score_sealed(lock_path: Path, approval_path: Path) -> None:
    gate = preconfirmatory_gate()
    if gate["status"] != "READY_FOR_SEALED_SCORING":
        raise PermissionError(f"SEALED_SCORING_BLOCKED: {gate['blockers']}")
    lock = read_json(lock_path)
    approval = validate_approval(approval_path, lock["protocol_sha256"])
    methods = list((PACKAGE_ROOT / "src" / "h10_c2" / "methods").glob("*.py"))
    methods += list((PACKAGE_ROOT / "src" / "h10_c2" / "baselines").glob("*.py"))
    current_tree = __import__("h10_c2.hashing", fromlist=["tree_sha256"]).tree_sha256(methods, PACKAGE_ROOT)
    if current_tree != lock["method_tree_sha256"]:
        raise PermissionError("POST_LOCK_METHOD_CHANGE")
    record_opening(
        ARTIFACT_ROOT / "sealed" / "opening_record.json",
        {
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
            "protocol_sha256": lock["protocol_sha256"],
            "approval_sha256": file_sha256(approval_path),
            "approved_by": approval["approved_by"],
            "purpose": "scoring_only",
        },
    )
    output = run_split("sealed")
    statistics = build_nonconfirmatory_statistics("sealed")
    claims = {
        item["claim"]: {
            "status": "supported" if item["ci_low"] > 0 and item["p_holm"] < 0.05 else "not_supported",
            "effect": item["effect"],
            "ci_low": item["ci_low"],
            "ci_high": item["ci_high"],
            "p_holm": item["p_holm"],
            "scope": "new_sealed_h10_c2_cases_only",
        }
        for item in statistics
    }
    write_json(ARTIFACT_ROOT / "sealed" / "claim_registry.json", claims)
    write_json(
        ARTIFACT_ROOT / "sealed" / "completion.json",
        {"status": "SEALED_SCORED", "results": str(output), "claim_registry": claims},
    )
    print(json.dumps({"status": "SEALED_SCORED", "results": str(output)}, ensure_ascii=False, indent=2))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="h10_c2")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("bootstrap")
    power = sub.add_parser("power")
    power.add_argument("--config", default="power_scenarios.yaml")
    power.add_argument("--output", type=Path)
    generate_parser = sub.add_parser("generate")
    generate_parser.add_argument("--split", required=True, choices=("development", "protocol_validation", "sealed"))
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--split", required=True, choices=("development", "protocol_validation"))
    sub.add_parser("audit-baselines")
    sub.add_parser("audit-oracle")
    sub.add_parser("leakage-audit")
    adjudication = sub.add_parser("export-adjudication")
    adjudication.add_argument("--sample-size", type=int, default=200)
    imported = sub.add_parser("import-adjudication")
    imported.add_argument("--reviewer-1", type=Path, required=True)
    imported.add_argument("--reviewer-2", type=Path, required=True)
    freeze = sub.add_parser("freeze-protocol")
    freeze.add_argument("--approval", type=Path, required=True)
    sub.add_parser("preconfirmatory-gate")
    sub.add_parser("package")
    score = sub.add_parser("score-sealed")
    score.add_argument("--lock", type=Path, required=True)
    score.add_argument("--approval", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "bootstrap":
            result = bootstrap()
        elif args.command == "power":
            result = run_power(args.config, args.output)
        elif args.command == "generate":
            result = generate(args.split)
        elif args.command == "run":
            result = {"results": str(run_split(args.split)), "statistics": build_nonconfirmatory_statistics(args.split)}
        elif args.command == "audit-baselines":
            result = audit_baselines()
        elif args.command == "audit-oracle":
            result = audit_oracle()
        elif args.command == "leakage-audit":
            result = run_leakage_audit()
        elif args.command == "export-adjudication":
            result = export_blind_cases(args.sample_size)
        elif args.command == "import-adjudication":
            result = import_results(args.reviewer_1, args.reviewer_2)
        elif args.command == "freeze-protocol":
            result = freeze_protocol(args.approval)
        elif args.command == "preconfirmatory-gate":
            result = preconfirmatory_gate()
        elif args.command == "package":
            result = build_deliverables()
        elif args.command == "score-sealed":
            score_sealed(args.lock, args.approval)
            return 0
        else:
            raise AssertionError(args.command)
    except (FileNotFoundError, PermissionError, RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str))
    if args.command == "preconfirmatory-gate" and result["status"] != "READY_FOR_SEALED_SCORING":
        return 3
    return 0
