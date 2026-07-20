#!/usr/bin/env python3
"""Build concise formative tables with explicit evidence provenance."""

from __future__ import annotations

import csv

from common import EXPERIMENTS, TABLES, evidence_ref, load_experiment, prepare


def main() -> None:
    prepare()
    rows = []
    for experiment in EXPERIMENTS:
        payload = load_experiment(experiment)
        evidence = evidence_ref(experiment)
        rows.append(
            {
                "experiment": experiment,
                "phase": str(payload.get("phase", "formative")),
                "formative_target_met": str(bool(payload.get("formative_target_met", False))).lower(),
                "confirmatory_status": "not_run",
                "claim_allowed": "false",
                "source_file": evidence["source_file"],
                "sha256": evidence["sha256"],
            }
        )
    csv_path = TABLES / "formative_experiment_status.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    markdown = [
        "| Experiment | Phase | Formative target | Confirmatory | Claim allowed |",
        "| --- | --- | --- | --- | --- |",
    ]
    markdown.extend(
        f"| {row['experiment']} | {row['phase']} | {row['formative_target_met']} | not_run | false |" for row in rows
    )
    (TABLES / "formative_experiment_status.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(f"PASS: chapter4_formative_tables rows={len(rows)}")


if __name__ == "__main__":
    main()
