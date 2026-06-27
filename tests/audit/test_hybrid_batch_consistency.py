import csv
import json
from pathlib import Path


def test_batch_summary_is_computed_from_cases() -> None:
    rows = list(csv.DictReader(Path("reports/studio_batch/hybrid_xiris_batch_cases.csv").open(encoding="utf-8")))
    summary = json.loads(Path("reports/studio_batch/hybrid_xiris_batch_summary.json").read_text(encoding="utf-8"))

    assert sum(row["fuzzyxai_action"] == "accept" for row in rows) == summary["accept"] == 612
    assert sum(row["fuzzyxai_action"] == "lower_confidence" for row in rows) == summary["lower_confidence"] == 201
    assert sum(row["fuzzyxai_action"] == "block" for row in rows) == summary["block"] == 187
    assert sum(int(row["baseline_miss"]) for row in rows) == summary["baseline_critical_misses"] == 168
    assert sum(int(row["fuzzyxai_miss"]) for row in rows) == summary["fuzzyxai_critical_misses"] == 0


def test_critical_cases_are_baseline_accept_and_fuzzyxai_block() -> None:
    rows = list(csv.DictReader(Path("reports/studio_batch/hybrid_xiris_batch_cases.csv").open(encoding="utf-8")))
    critical = [row for row in rows if row["is_critical_case"] == "1"]

    assert len(critical) == 168
    assert sum(row["baseline_action"] == "accept" for row in critical) == 168
    assert sum(row["fuzzyxai_action"] == "block" for row in critical) == 168
    assert sum(int(row["fuzzyxai_miss"]) for row in critical) == 0
