#!/usr/bin/env python3
"""Certify an already-produced FuzzyXAI view-model, or replay a receipt.

Examples:
 python scripts/certify_doctoral_route.py certify --input view_model.json --taxonomy ... --representation ... --out-dir ./cert
 python scripts/certify_doctoral_route.py verify --receipt ./cert/certificate.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from fuzzyxai.doctoral_batch import certify_batch
from fuzzyxai.doctoral_report import render_doctoral_report
from fuzzyxai.doctoral_certification import (  # noqa: E402
    certify_payload, compare_recertifications, project_for_audience, verify_certificate,
)


def load(path: Path) -> dict:
    content = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(content, dict):
        raise ValueError(f"{path}: expected JSON object")
    return content


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    cmd = p.add_subparsers(dest="command", required=True)
    run = cmd.add_parser("certify", help="run all chapter 1-4 structural checks")
    for name in ("input", "taxonomy", "representation"):
        run.add_argument("--" + name, type=Path, required=True)
    run.add_argument("--policy", type=Path)
    run.add_argument("--out-dir", type=Path, required=True)
    run.add_argument("--html", action="store_true", help="write offline human-readable HTML report")
    batch = cmd.add_parser("batch", help="certify independent JSON view models in a directory")
    batch.add_argument("--input-dir", type=Path, required=True)
    batch.add_argument("--taxonomy", type=Path, required=True)
    batch.add_argument("--representation", type=Path, required=True)
    batch.add_argument("--policy", type=Path)
    batch.add_argument("--out-dir", type=Path, required=True)
    check = cmd.add_parser("verify", help="replay a deterministic receipt")
    check.add_argument("--receipt", type=Path, required=True)
    compare = cmd.add_parser("recertify", help="compare two independently verified receipts")
    compare.add_argument("--before", type=Path, required=True)
    compare.add_argument("--after", type=Path, required=True)
    compare.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    try:
        if args.command == "certify":
            payload = load(args.input)
            receipt = certify_payload(payload, taxonomy=load(args.taxonomy),
                                      representation=load(args.representation),
                                      policy=load(args.policy) if args.policy else None)
            write(args.out_dir / "certificate.json", receipt)
            write(args.out_dir / "verification.json", verify_certificate(receipt))
            if args.html:
                render_doctoral_report(receipt, args.out_dir / "report.html")
            for audience in ("end_user", "developer", "auditor"):
                write(args.out_dir / f"{audience}.json", project_for_audience(receipt, audience))
            issues = receipt["decision"]["issues"]
            with (args.out_dir / "issues.csv").open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["code", "path", "detail", "kind", "causes"])
                writer.writeheader()
                for issue in issues:
                    writer.writerow({**issue, "causes": "|".join(issue["causes"])})
            print(json.dumps({"status": receipt["decision"]["status"], "sha256": receipt["sha256"],
                              "issues": len(issues), "out_dir": str(args.out_dir)}, ensure_ascii=False))
            return 0 if receipt["decision"]["status"] == "valid" else 2
        if args.command == "batch":
            inputs = sorted(args.input_dir.glob("*.json"))
            if not inputs:
                raise ValueError(f"no JSON view models in {args.input_dir}")
            report = certify_batch(((p.stem, load(p)) for p in inputs),
                                   taxonomy=load(args.taxonomy),
                                   representation=load(args.representation),
                                   policy=load(args.policy) if args.policy else None)
            write(args.out_dir / "batch_summary.json", report)
            args.out_dir.mkdir(parents=True, exist_ok=True)
            with (args.out_dir / "batch_cases.csv").open("w", encoding="utf-8", newline="") as f:
                fields = ("case_id", "status", "action", "auto_accept_permitted", "n_issues", "issue_codes", "receipt_sha256")
                writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                writer.writeheader()
                for row in report["cases"]:
                    writer.writerow({**row, "issue_codes": "|".join(row["issue_codes"])})
            print(json.dumps({"n_cases": report["n_cases"], "status_counts": report["status_counts"]}, ensure_ascii=False))
            return 0 if report["status_counts"].get("invalid", 0) == 0 and report["status_counts"].get("insufficient_evidence", 0) == 0 else 2
        if args.command == "verify":
            verified = verify_certificate(load(args.receipt))
            print(json.dumps(verified, ensure_ascii=False))
            return 0 if verified["valid"] else 3
        report = compare_recertifications(load(args.before), load(args.after))
        write(args.out, report)
        print(json.dumps(report, ensure_ascii=False))
        return 0 if report["recertified"] else 2
    except (ValueError, TypeError, KeyError, OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
