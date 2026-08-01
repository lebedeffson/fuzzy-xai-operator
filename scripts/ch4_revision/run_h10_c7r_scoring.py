#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from fuzzyxai.experiments.h10_c7a import FrozenBudgetRankingEngine
from fuzzyxai.experiments.h10_c7r import (
    final_status,
    load_held_out_inputs,
    repository_cluster_bootstrap,
    repository_rows,
    score_incidents,
    sha256,
)
from fuzzyxai.repository_diagnostics.guided_diagnosis import (
    GuidedNaturalDiagnosisEngine,
)

METHOD_COMMIT = "358ed40a0fb7f5adc1291695ff15affa39cae485"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, values: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(value, sort_keys=True) + "\n" for value in values),
        encoding="utf-8",
    )


def _write_csv(path: Path, values: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(
            target,
            fieldnames=list(values[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(values)


def _verify_locks(protocol_dir: Path, root: Path) -> tuple[bool, bool]:
    manifest = json.loads(
        (protocol_dir / "H10_C7R_LOCK_MANIFEST.json").read_text(
            encoding="utf-8"
        )
    )
    for name, expected in manifest["lock_sha256"].items():
        if sha256(protocol_dir / name) != expected:
            raise ValueError(f"H10-C7R lock changed: {name}")
    method = json.loads(
        (protocol_dir / "H10_C7R_METHOD_LOCK.json").read_text(
            encoding="utf-8"
        )
    )
    if method["method_commit"] != METHOD_COMMIT:
        raise ValueError("H10-C7R method commit mismatch")
    for relative, expected in method["frozen_file_sha256"].items():
        if sha256(root / relative) != expected:
            raise ValueError(f"H10-C7R method signature mismatch: {relative}")
    budget = json.loads(
        (protocol_dir / "H10_C7R_BUDGET_LOCK.json").read_text(
            encoding="utf-8"
        )
    )
    budget_pass = (
        budget["primary"] == {"budget": 20, "method": "R5"}
        and budget["baseline"] == {"budget": 160, "method": "B_BM25"}
        and float(budget["minimum_recall"]) == 0.8
    )
    if not budget_pass:
        raise ValueError("H10-C7R budget signature mismatch")
    return True, True


def _report(status: dict[str, object]) -> str:
    bootstrap = status["bootstrap"]
    return f"""# H10-C7R final report

## Status

```text
{status["status"]}
Scientific result: {status["scientific_result"]}
```

Held-out: {status["incident_count"]} incidents from
{status["repository_count"]} repositories.

- R5 Recall@20: {status["r5_recall_at_20"]:.6f}
- B_BM25 Recall@160: {status["baseline_recall_at_160"]:.6f}
- R5 coverage: {status["r5_coverage"]:.6f}
- Mean R5 search-space reduction:
  {status["mean_r5_search_space_reduction"]:.6f}
- Mean B_BM25 search-space reduction:
  {status["mean_baseline_search_space_reduction"]:.6f}
- Repository-cluster 95% CI:
  [{bootstrap["ci_lower"]:.6f}, {bootstrap["ci_upper"]:.6f}]

Contract-family inference is descriptive and did not affect this status.
The result concerns candidate-space reduction, not automatic root-cause
confirmation or repair.

Although the repository-cluster confidence interval for the reduction
difference is strictly positive, the primary recall condition failed.
Therefore the larger reduction cannot be interpreted as supported practical
search-space reduction at the registered recall level.
"""


def _chapter_fragment(status: dict[str, object]) -> str:
    bootstrap = status["bootstrap"]
    return f"""# H10-C7R chapter fragment

On a new repository-disjoint held-out set of {status["incident_count"]}
natural incidents from {status["repository_count"]} repositories, frozen R5
retained the target program element within 20 candidates in
{status["r5_recall_at_20"]:.1%} of incidents. The frozen B_BM25 baseline
reached {status["baseline_recall_at_160"]:.1%} recall with 160 candidates.
R5 produced greater mean search-space reduction
({status["mean_r5_search_space_reduction"]:.4f} versus
{status["mean_baseline_search_space_reduction"]:.4f}); the
repository-cluster 95% confidence interval for the reduction difference was
[{bootstrap["ci_lower"]:.4f}, {bootstrap["ci_upper"]:.4f}].

The registered minimum R5 Recall@20 was 0.80, but the observed value was
{status["r5_recall_at_20"]:.2f}. H10-C7R therefore has official status
`{status["status"]}`. The result does not support practical candidate-space
reduction at the registered recall level and does not evaluate automatic
root-cause confirmation, repair, or developer-time reduction.
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = args.manifest.resolve()
    gold_path = args.gold.resolve()
    protocol_dir = args.protocol_dir.resolve()
    output = args.output.resolve()
    reports = args.reports.resolve()
    authorization_path = args.authorization.resolve()
    ledger_path = output / "SCORING_OPENING_LEDGER.json"
    if ledger_path.exists():
        raise ValueError("H10-C7R official scoring was already opened")
    authorization = json.loads(
        authorization_path.read_text(encoding="utf-8")
    )
    lock_manifest_path = protocol_dir / "H10_C7R_LOCK_MANIFEST.json"
    authorization_checks = {
        "authorization_status": authorization["authorization_status"]
        == "H10_C7R_SCORING_AUTHORIZED_ONCE",
        "manifest_sha256": authorization["held_out_manifest_sha256"]
        == sha256(manifest),
        "gold_sha256": authorization["gold_sha256"] == sha256(gold_path),
        "lock_manifest_sha256": authorization["lock_manifest_sha256"]
        == sha256(lock_manifest_path),
        "opening_count_zero": authorization["opening_count_before_scoring"] == 0,
    }
    if not all(authorization_checks.values()):
        raise ValueError("H10-C7R scoring authorization mismatch")
    method_pass, budget_pass = _verify_locks(protocol_dir, root)
    output.mkdir(parents=True, exist_ok=True)
    _write_json(
        ledger_path,
        {
            "opening_count": 1,
            "protocol_id": "H10-C7R-v1",
            "scoring_id": hashlib.sha256(
                (
                    sha256(manifest)
                    + sha256(gold_path)
                    + sha256(lock_manifest_path)
                ).encode()
            ).hexdigest(),
            "status": "OFFICIAL_SCORING_OPENED",
        },
    )
    inputs = load_held_out_inputs(
        manifest,
        gold_path,
        protocol_dir / "H10_C7R_EXCLUSION_LOCK.json",
    )
    rows = score_incidents(
        inputs,
        FrozenBudgetRankingEngine(
            GuidedNaturalDiagnosisEngine(structural_only=True)
        ),
    )
    per_repository = repository_rows(rows)
    bootstrap = repository_cluster_bootstrap(rows)
    status = final_status(
        rows,
        bootstrap,
        gold_leakage=0,
        method_signature_passed=method_pass,
        budget_signature_passed=budget_pass,
        single_official_scoring=True,
    )
    _write_jsonl(output / "PER_INCIDENT_RESULTS.jsonl", rows)
    _write_csv(output / "PER_REPOSITORY_RESULTS.csv", per_repository)
    _write_csv(output / "BOOTSTRAP_RESULTS.csv", [bootstrap])
    _write_json(output / "H10_C7R_FINAL_STATUS.json", status)
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "H10_C7R_FINAL_REPORT.md").write_text(
        _report(status),
        encoding="utf-8",
    )
    (reports / "H10_C7R_CHAPTER_FRAGMENT.md").write_text(
        _chapter_fragment(status),
        encoding="utf-8",
    )
    _write_json(
        reports / "H10_C7R_SCORING_AUDIT.json",
        {
            "authorization_checks": authorization_checks,
            "budget_signature_passed": budget_pass,
            "gold_leakage": 0,
            "method_signature_passed": method_pass,
            "opening_count": 1,
        },
    )
    print(json.dumps(status, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
