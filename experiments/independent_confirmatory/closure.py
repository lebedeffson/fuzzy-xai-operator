from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

from .common import ARTIFACTS, LOCK, OPENING, ROOT, git_commit, read_json, sha256_file, verify_protocol, write_json


CLAIMS = ARTIFACTS / "closure" / "claim_registry.json"
EVIDENCE = ARTIFACTS / "closure" / "evidence_map.json"
REPORT = ARTIFACTS / "closure" / "validation_report.md"


def build_claims() -> None:
    result = read_json(ARTIFACTS / "confirmatory" / "summary.json")
    replay = read_json(ARTIFACTS / "replay" / "chronological_summary.json")
    h3 = result["H3"]
    h5 = result["H5"]["typed_open_set"]
    h3_r1 = (
        h3["relative_reduction"] >= 0.15
        and h3["hierarchical_bootstrap_ci_95"][1] < 0.0
        and h3["holm_adjusted_p"] < 0.05
        and h3["hard_block_rate"] <= 0.05
        and h3["false_block_rate"] <= 0.01
    )
    h3_r2 = h3["coverage_gain"] >= 0.05
    p0 = h3["P0_vs_full"]
    h3_r3 = p0["absolute_rate_difference_full_minus_predictive"] < 0.0 and p0["hierarchical_bootstrap_ci_95"][1] < 0.0 and p0["holm_adjusted_p"] < 0.05
    h5_a3 = (
        h5["unknown_fault_recall"] >= 0.85
        and h5["false_certification"] <= 0.01
        and h5["known_type_macro_f1_degradation"] <= 0.03
        and h5["unknown_rejection_auroc"] >= 0.90
        and h5["source_region_localization"] >= 0.80
    )
    h5_a4 = h5["repair_candidate_recall"] >= 0.80
    registry = {
        "schema_version": "1.0",
        "generated_from_frozen_evidence": True,
        "manual_positive_override": False,
        "frozen_predecessor_claims": {
            "H3-original": "not_supported",
            "H5-P-original": "not_supported",
            "H6-general": "not_supported",
        },
        "claims": {
            "H3-R1": "supported_independent_with_controlled_route_faults" if h3_r1 else "not_supported",
            "H3-R2": "supported_independent_with_controlled_route_faults" if h3_r2 else "not_supported",
            "H3-R3": "supported_independent_with_controlled_route_faults" if h3_r3 else "not_supported",
            "H3-chronological-replay": "descriptive_controlled_replay_only" if replay["incident_level_recall"] > 0 else "not_supported",
            "H5-A3": "supported_held_out_registered_families" if h5_a3 else "not_supported",
            "H5-A4": "supported_held_out_registered_families" if h5_a4 else "not_supported",
            "H5-P2": "not_evaluated_secondary_endpoint",
            "H5-P3": "not_evaluated_secondary_endpoint",
            "H6-R6": "not_evaluated_confirmatory_gate_closed",
            "H6-R7": "not_evaluated_confirmatory_gate_closed",
            "H6-R8": "formative_only_confirmatory_gate_closed",
        },
        "scientific_release_status": "experimental_not_stable",
        "merge_to_main_allowed": False,
    }
    write_json(CLAIMS, registry)
    report = f"""# Independent Confirmatory Closure Validation Report

- Scientific release status: `experimental_not_stable`
- Merge to `main`: `false`
- Sealed scoring objects: `{h3['objects']}`
- Frozen primary baseline: `{h3['primary_baseline']}`
- H3-R1 relative reduction: `{h3['relative_reduction']:.8f}` (required: `>= 0.15`)
- H3-R1 hierarchical 95% CI for full minus baseline: `{h3['hierarchical_bootstrap_ci_95']}`
- H3-R1 Holm-adjusted p: `{h3['holm_adjusted_p']:.8g}`
- H3 hard-block rate: `{h3['hard_block_rate']:.8f}`
- H3 false-block rate: `{h3['false_block_rate']:.8f}`
- H3-R2 coverage gain: `{h3['coverage_gain']:.8f}`
- H5 unknown recall / AUROC: `{h5['unknown_fault_recall']:.8f}` / `{h5['unknown_rejection_auroc']:.8f}`
- H5 known-type macro-F1: `{h5['known_type_macro_f1']:.8f}`
- H5 source localization: `{h5['source_region_localization']:.8f}`
- H6 confirmatory opening: `false`
- Replay incident recall: `{replay['incident_level_recall']:.8f}`
- Replay hard-block rate: `{replay['controllers']['full_hierarchical_fuzzyxai']['hard_block_rate']:.8f}`

The replay incident-level table was recomputed after a declared aggregation-only defect. Controller actions,
policy thresholds and sealed H3/H5 results were not changed. The full details are in
`artifacts/independent_confirmatory/replay/scoring_recovery_deviation.json`.
"""
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(report, encoding="utf-8")


def build_evidence() -> None:
    paths = [
        Path("config/independent_confirmatory_protocol.json"),
        Path("config/independent_confirmatory_protocol_amendment_001.json"),
        Path("artifacts/independent_confirmatory/data/dataset_manifest.json"),
        Path("artifacts/independent_confirmatory/data/leakage_audit.json"),
        Path("artifacts/independent_confirmatory/models/model_manifest.json"),
        Path("artifacts/independent_confirmatory/formative/summary.json"),
        LOCK.relative_to(ROOT),
        OPENING.relative_to(ROOT),
        Path("artifacts/independent_confirmatory/confirmatory/summary.json"),
        Path("artifacts/independent_confirmatory/replay/chronological_summary.json"),
        Path("artifacts/independent_confirmatory/replay/scoring_recovery_deviation.json"),
        CLAIMS.relative_to(ROOT),
        REPORT.relative_to(ROOT),
    ]
    rows = []
    for index, relative in enumerate(paths, 1):
        path = ROOT / relative
        rows.append({"evidence_id": f"IC{index:02d}", "path": str(relative), "sha256": sha256_file(path), "commit": git_commit()})
    write_json(EVIDENCE, {"schema_version": "1.0", "records": rows})


def check() -> None:
    verify_protocol()
    for path in (LOCK, OPENING, ARTIFACTS / "confirmatory" / "summary.json", ARTIFACTS / "replay" / "chronological_summary.json", CLAIMS, REPORT, EVIDENCE):
        if not path.is_file():
            raise RuntimeError(f"missing closure artifact: {path.relative_to(ROOT)}")
    claims = read_json(CLAIMS)
    expected = {"H3-original": "not_supported", "H5-P-original": "not_supported", "H6-general": "not_supported"}
    if claims["frozen_predecessor_claims"] != expected:
        raise RuntimeError("frozen negative claims changed")
    if claims["merge_to_main_allowed"] or claims["scientific_release_status"] != "experimental_not_stable":
        raise RuntimeError("independent branch cannot be promoted automatically")
    for record in read_json(EVIDENCE)["records"]:
        if sha256_file(ROOT / record["path"]) != record["sha256"]:
            raise RuntimeError(f"evidence hash mismatch: {record['path']}")
    print("PASS independent-release-check experimental_not_stable=true merge_to_main=false")


def build_zip() -> None:
    check()
    output = ROOT / "release_artifacts" / f"fuzzyxai-independent-confirmatory-{git_commit()[:12]}.zip"
    output.parent.mkdir(parents=True, exist_ok=True)
    include = [EVIDENCE, *(ROOT / record["path"] for record in read_json(EVIDENCE)["records"])]
    include.extend(
        [
            ROOT / "experiments" / "independent_confirmatory" / name
            for name in ("common.py", "modeling.py", "formative.py", "freeze.py", "confirmatory.py", "replay.py", "closure.py")
        ]
    )
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(include)):
            archive.write(path, path.relative_to(ROOT))
        archive.writestr("BUNDLE_MANIFEST.json", json.dumps({"commit": git_commit(), "scope": "experimental independent confirmatory evidence", "stable_release": False}, indent=2))
    print(f"PASS independent-one-zip path={output} sha256={sha256_file(output)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("claims", "evidence", "check", "zip"))
    action = parser.parse_args().action
    {"claims": build_claims, "evidence": build_evidence, "check": check, "zip": build_zip}[action]()


if __name__ == "__main__":
    main()
