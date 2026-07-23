from __future__ import annotations

import csv
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

from .runner import ARTIFACT_ROOT, REPO_ROOT, file_sha256, load_rows, read_json, write_json


def build_tables() -> list[Path]:
    outputs = []
    for split in ("development", "protocol_validation"):
        rows = load_rows(split)
        grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
        for row in rows:
            grouped[(str(row["method"]), str(row["stratum"]))].append(row)
        output_rows = []
        for (method, stratum), values in sorted(grouped.items()):
            output_rows.append(
                {
                    "split": split,
                    "method": method,
                    "stratum": stratum,
                    "n": len(values),
                    "optimal_set_membership": sum(
                        float(item["optimal_set_membership"]) for item in values
                    )
                    / len(values),
                    "normalized_cost_regret": sum(
                        float(item["normalized_cost_regret"]) for item in values
                    )
                    / len(values),
                    "full_recertification_success": sum(
                        float(item["full_recertification_success"]) for item in values
                    )
                    / len(values),
                    "false_certification": sum(
                        float(item["false_certification"]) for item in values
                    )
                    / len(values),
                    "runtime_ms_mean": sum(float(item["runtime_ms"]) for item in values)
                    / len(values),
                }
            )
        path = ARTIFACT_ROOT / "tables" / f"{split}_method_by_stratum.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(output_rows[0]))
            writer.writeheader()
            writer.writerows(output_rows)
        outputs.append(path)
    stats_rows = []
    for split in ("development", "protocol_validation"):
        for item in read_json(ARTIFACT_ROOT / "results" / f"{split}_statistics.json"):
            stats_rows.append({"split": split, **{k: v for k, v in item.items() if k != "pipeline_effects"}})
    stats_path = ARTIFACT_ROOT / "tables" / "primary_statistics.csv"
    with stats_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(stats_rows[0]))
        writer.writeheader()
        writer.writerows(stats_rows)
    outputs.append(stats_path)
    return outputs


def build_evidence_map() -> Path:
    entries = []
    for source in sorted((ARTIFACT_ROOT / "tables").glob("*.csv")):
        with source.open(encoding="utf-8", newline="") as stream:
            for row_number, row in enumerate(csv.DictReader(stream), 2):
                for metric, value in row.items():
                    try:
                        numeric = float(value)
                    except (TypeError, ValueError):
                        continue
                    claim = row.get("claim") or (
                        "H10-C3a" if "optimal_set" in metric or "cost_regret" in metric else "H10-C3b"
                    )
                    entries.append(
                        {
                            "claim_id": claim,
                            "metric": metric,
                            "value": numeric,
                            "dataset": row.get("split", "aggregate"),
                            "method": row.get("method", "paired_hierarchical_bootstrap"),
                            "source_file": str(source.relative_to(ARTIFACT_ROOT)),
                            "locator": f"row={row_number},column={metric}",
                            "sha256": file_sha256(source),
                            "evidence_generation_commit": _git_head(),
                            "closure_packaging_commit": None,
                            "bundle_commit": None,
                            "status": "development"
                            if row.get("split") == "development"
                            else "protocol_validation",
                        }
                    )
    output = ARTIFACT_ROOT / "evidence" / "evidence_map.json"
    write_json(output, entries)
    return output


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def build_validation_report() -> Path:
    gate = read_json(ARTIFACT_ROOT / "gate" / "preconfirmatory_gate.json")
    development = read_json(ARTIFACT_ROOT / "results" / "development_statistics.json")
    validation = read_json(ARTIFACT_ROOT / "results" / "protocol_validation_statistics.json")
    power = read_json(ARTIFACT_ROOT / "power" / "power.json")
    lines = [
        "# H10-C3 v23 validation report",
        "",
        f"- Software gate: `{gate['status']}`",
        "- Sealed generated: `false`",
        "- Sealed opening count: `0`",
        "- H10-C3a confirmatory status: `NOT_EVALUATED`",
        "- H10-C3b confirmatory status: `NOT_EVALUATED`",
        "",
        "## Open-part results",
        "",
    ]
    for split, items in (("development", development), ("protocol validation", validation)):
        for item in items:
            lines.append(
                f"- {split} {item['claim']}: effect={item['effect']:.6f}, "
                f"95% CI=[{item['ci_low']:.6f}, {item['ci_high']:.6f}], "
                f"Holm p={item['p_holm']:.6f}, status=`{item['status']}`."
            )
    lines.extend(["", "## Power", ""])
    for item in power:
        lines.append(
            f"- {item['claim']}: point={item['point_power']:.4f}, "
            f"lower={item['lower_confidence_bound']:.4f}, "
            f"upper={item['upper_confidence_bound']:.4f}, status=`{item['status']}`."
        )
    lines.extend(
        [
            "",
            "These are development and protocol-validation results only. "
            "They are not confirmatory evidence and do not alter H3, H5-P, H6-general, H10-L or H10-R.",
        ]
    )
    output = ARTIFACT_ROOT / "reports" / "validation_report.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def build_handoff_zip() -> Path:
    deliverable = REPO_ROOT / "dist" / "fuzzyxai-h10-c3-v23-preconfirmatory-handoff.zip"
    deliverable.parent.mkdir(parents=True, exist_ok=True)
    staging = ARTIFACT_ROOT / "closure" / "handoff"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    shutil.copytree(REPO_ROOT / "framework" / "fuzzyxai" / "fuzzyxai" / "diagnostics", staging / "source" / "diagnostics")
    shutil.copytree(REPO_ROOT / "experiments" / "h10_c3", staging / "source" / "experiments" / "h10_c3")
    shutil.copytree(ARTIFACT_ROOT, staging / "artifacts", ignore=shutil.ignore_patterns("closure"))
    shutil.make_archive(str(deliverable.with_suffix("")), "zip", staging)
    Path(str(deliverable) + ".sha256").write_text(
        f"{file_sha256(deliverable)}  {deliverable.name}\n", encoding="utf-8"
    )
    return deliverable
