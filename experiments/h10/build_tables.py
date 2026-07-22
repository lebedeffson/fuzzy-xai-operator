from __future__ import annotations


import pandas as pd

from .common import ARTIFACT_ROOT, read_json, write_csv, write_json


def build() -> None:
    out = ARTIFACT_ROOT / "tables"
    datasets = read_json(ARTIFACT_ROOT / "data" / "dataset_manifest.json")
    data_rows = []
    raw = pd.read_csv(ARTIFACT_ROOT / "confirmatory" / "raw_results.csv")
    for item in datasets:
        test = raw[(raw["dataset"] == item["dataset_id"]) & (raw["method"] == "full_h10")]
        data_rows.append(
            {
                "dataset": item["dataset_id"],
                "modality": item["modality"],
                "train": item["split_counts"]["train"],
                "development": item["split_counts"]["development"],
                "sealed_test": item["split_counts"]["sealed_test"],
                "known_faults": int((~test["truth_unknown"] & (test["truth_status"] != "valid")).sum()),
                "unknown_faults": int(test["truth_unknown"].sum()),
                "composite_faults": int(test["composite"].sum()),
            }
        )
    method = pd.read_csv(ARTIFACT_ROOT / "confirmatory" / "method_summary.csv")
    table2 = method[["method", "source_localization_f1", "repair_set_f1", "false_certification", "false_block", "diagnostic_latency"]]
    table3 = method[["method", "parent_f1", "leaf_f1", "unknown_auroc", "unknown_recall", "abstention_accuracy"]].copy()
    table4 = method[["method", "cut_exact", "cut_jaccard", "cut_cost_ratio", "extra_nodes", "runtime_ms"]]
    replay_path = ARTIFACT_ROOT / "replay" / "method_summary.csv"
    table5 = pd.read_csv(replay_path) if replay_path.exists() else pd.DataFrame()
    tests = read_json(ARTIFACT_ROOT / "confirmatory" / "statistical_tests.json")
    table6 = [
        {
            "claim": "H10-L" if row["metric"] == "source_f1" else "H10-R",
            "effect": row["effect"],
            "ci_low": row["ci_low"],
            "ci_high": row["ci_high"],
            "p_raw": row["p_raw"],
            "p_holm": row["p_holm"],
            "status": row["status"],
        }
        for row in tests
    ]
    write_csv(out / "table1_datasets.csv", data_rows)
    table2.to_csv(out / "table2_primary_results.csv", index=False, lineterminator="\n")
    table3.to_csv(out / "table3_hierarchical_diagnosis.csv", index=False, lineterminator="\n")
    table4.to_csv(out / "table4_minimal_cut.csv", index=False, lineterminator="\n")
    if not table5.empty:
        table5.to_csv(out / "table5_replay.csv", index=False, lineterminator="\n")
    write_csv(out / "table6_statistics.csv", table6)
    for csv_path in sorted(out.glob("*.csv")):
        current = pd.read_csv(csv_path)
        records = current.where(pd.notna(current), None).to_dict(orient="records")
        write_json(csv_path.with_suffix(".json"), records)


if __name__ == "__main__":
    build()
