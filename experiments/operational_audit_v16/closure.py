from __future__ import annotations

from .common import ARTIFACTS, ROOT, git_commit, read_json, sha256_file, verify_protocol, write_json


def main() -> None:
    verify_protocol()
    result = read_json(ARTIFACTS / "confirmatory" / "summary.json")
    replay = read_json(ARTIFACTS / "replay" / "summary.json")
    a1 = result["A1_localization_gain"] >= 0.10 and result["hierarchical_inference"]["A1"]["ci_95"][0] > 0 and result["hierarchical_inference"]["A1"]["hierarchical_p"] < 0.05
    a2 = result["A2_repair_gain"] >= 0.10 and result["hierarchical_inference"]["A2"]["ci_95"][0] > 0 and result["hierarchical_inference"]["A2"]["hierarchical_p"] < 0.05
    a3 = replay["A3_relative_restoration_reduction_vs_route_only"] >= 0.15
    a4 = result["A4_false_certification"] <= 0.01
    a5 = result["A5_byte_identical_rate"] == 1.0
    claims = {"A1": "supported" if a1 else "not_supported", "A2": "supported" if a2 else "not_supported", "A3": "supported_controlled_replay_only" if a3 else "not_supported", "A4": "supported_registered_mutations_only" if a4 else "not_supported", "A5": "supported_byte_identical" if a5 else "not_supported"}
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
    generation = git_commit()
    write_json(ARTIFACTS / "closure" / "evidence_map.json", {"evidence_generation_commit": generation, "closure_packaging_commit": None, "bundle_commit": None, "records": [{"path": path, "sha256": sha256_file(ROOT / path)} for path in paths]})
    print(f"PASS operational-audit-claims A1={claims['A1']} A2={claims['A2']} A3={claims['A3']} A4={claims['A4']} A5={claims['A5']}")


if __name__ == "__main__":
    main()
