from __future__ import annotations

from .common import ARTIFACTS, ROOT, git_commit, read_json, sha256_file, verify_protocol, write_json


def main() -> None:
    verify_protocol()
    result = read_json(ARTIFACTS / "confirmatory" / "summary.json")
    replay = read_json(ARTIFACTS / "replay" / "summary.json")
    a1 = result["A1_localization_gain"] >= 0.10 and result["hierarchical_inference"]["A1"]["ci_95"][0] > 0 and result["hierarchical_inference"]["A1"]["hierarchical_p"] < 0.05
    a2 = result["A2_repair_gain"] >= 0.10 and result["hierarchical_inference"]["A2"]["ci_95"][0] > 0 and result["hierarchical_inference"]["A2"]["hierarchical_p"] < 0.05
    a3 = False  # Controlled service times have no registered incident-level inferential interval.
    a4 = False  # The point estimate passes, but no aligned hierarchical boundary interval was registered.
    a5 = result["A5_byte_identical_rate"] == 1.0
    claims = {"A1": "supported" if a1 else "not_supported", "A2": "supported" if a2 else "not_supported", "A3": "supported" if a3 else "descriptive_controlled_replay_only", "A4": "supported" if a4 else "descriptive_point_estimate_only", "A5": "supported_byte_identical" if a5 else "not_supported"}
    registry = {"claims": claims, "frozen_negative_claims": read_json(ROOT / "config" / "operational_audit_v16_claim_registry.json")["frozen_negative_claims"], "scientific_status": "experimental_operational_audit", "production_validated": False, "merge_to_main_allowed": False}
    write_json(ARTIFACTS / "closure" / "claim_registry.json", registry)
    paths = [
        "config/operational_audit_v16_protocol.json",
        "config/operational_audit_v16_protocol_amendment_001.json",
        "artifacts/operational_audit_v16/data/dataset_manifest.json",
        "artifacts/operational_audit_v16/data/pre_opening_leakage_audit.json",
        "artifacts/operational_audit_v16/data/post_scoring_leakage_audit.json",
        "artifacts/operational_audit_v16/formative/summary.json",
        "artifacts/operational_audit_v16/lock/protocol_lock.json",
        "artifacts/operational_audit_v16/opening/opening_record.json",
        "artifacts/operational_audit_v16/confirmatory/summary.json",
        "artifacts/operational_audit_v16/replay/summary.json",
        "artifacts/operational_audit_v16/closure/claim_registry.json",
    ]
    generation = read_json(ARTIFACTS / "opening" / "opening_record.json")["scoring_commit"]
    packaging = git_commit()
    write_json(ARTIFACTS / "closure" / "evidence_map.json", {"evidence_generation_commit": generation, "closure_packaging_commit": packaging, "bundle_commit": packaging, "records": [{"path": path, "sha256": sha256_file(ROOT / path)} for path in paths]})
    report = f"""# Operational Audit v16 Validation Report

- Fresh sealed objects: {sum(item['objects'] for item in result['datasets'])}
- A1 localization gain: {result['A1_localization_gain']:.8f}; hierarchical CI {result['hierarchical_inference']['A1']['ci_95']}; p={result['hierarchical_inference']['A1']['hierarchical_p']}
- A2 repair gain: {result['A2_repair_gain']:.8f}; hierarchical CI {result['hierarchical_inference']['A2']['ci_95']}; p={result['hierarchical_inference']['A2']['hierarchical_p']}
- A3 controlled replay restoration reduction: {replay['A3_relative_restoration_reduction_vs_route_only']:.8f}; descriptive only because service times are protocol-defined simulation values
- A4 false-certification point estimate: {result['A4_false_certification']:.8f}; descriptive until an aligned boundary interval is available
- A5 byte-identical replay: {result['A5_byte_identical_rate']:.8f}
- Production validation: false
- Merge to main: false
"""
    (ARTIFACTS / "closure" / "validation_report.md").write_text(report, encoding="utf-8")
    print(f"PASS operational-audit-claims A1={claims['A1']} A2={claims['A2']} A3={claims['A3']} A4={claims['A4']} A5={claims['A5']}")


if __name__ == "__main__":
    main()
