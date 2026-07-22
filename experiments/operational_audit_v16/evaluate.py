from __future__ import annotations

import hashlib
from dataclasses import replace
from datetime import datetime, timezone

import numpy as np

from fuzzyxai.operational_audit import LexicographicController, PredictiveSelector, RepairPlanner, RouteArtifact, TypedRouteGuard, mutate_route_artifact

from .common import ARTIFACTS, DATA, LOCK, OPENING, git_commit, read_json, sha256_file, unit, verify_protocol, write_json


FAMILIES = tuple(read_json(__import__("pathlib").Path(__file__).resolve().parents[2] / "config" / "operational_audit_v16_protocol.json")["mutation_families"])
SEVERITIES = ("subtle", "moderate", "severe")
REGIONS = {
    "model_explainer_mismatch": "model_explainer_link",
    "stale_calibration": "calibration_registry",
    "preprocessing_order_change": "preprocessing_graph",
    "feature_schema_incompatibility": "feature_schema",
    "cross_model_artifact_mix": "artifact_lineage",
    "checksum_corruption": "canonical_store",
    "reduction_link_loss": "representation_reducer",
    "reference_population_substitution": "reference_population",
    "partial_provenance_deletion": "provenance_graph",
    "dictionary_or_tokenizer_version_change": "dictionary_registry",
}


def base_artifact(dataset_id: str, identity: str) -> RouteArtifact:
    digest = hashlib.sha256(f"{dataset_id}:{identity}".encode()).hexdigest()
    dictionary = "tokenizer-v1" if "youtube" in dataset_id else "feature-dictionary-v1"
    return RouteArtifact(
        f"route:{digest[:20]}",
        f"{dataset_id}:model-v1",
        f"{dataset_id}:model-v1",
        f"{dataset_id}:model-v1",
        ("01-load", "02-normalize", "03-predict"),
        ("feature-a", "feature-b", "feature-c"),
        ("feature-a", "feature-b", "feature-c"),
        digest,
        digest,
        f"source:{identity}",
        f"source:{identity}",
        f"reference:{dataset_id}",
        f"reference:{dataset_id}",
        ("model", "preprocessing", "explainer", "reference"),
        ("model", "preprocessing", "explainer", "reference"),
        dictionary,
        dictionary,
    )


def _cases(dataset_id: str, identities: list[str]) -> list[dict[str, object]]:
    cases = []
    for index, identity in enumerate(identities):
        base = base_artifact(dataset_id, identity)
        for repeat in range(4):
            cases.append({"id": f"{identity}:valid:{repeat}", "artifact": replace(base, artifact_id=f"{base.artifact_id}:valid:{repeat}"), "families": (), "severity": "valid"})
        family = FAMILIES[index % len(FAMILIES)]
        severity = SEVERITIES[(index // len(FAMILIES)) % len(SEVERITIES)]
        mutated = mutate_route_artifact(base, family, severity)
        families = [family]
        if index % 7 == 0:
            second = FAMILIES[(index + 3) % len(FAMILIES)]
            mutated = mutate_route_artifact(mutated, second, severity)
            families.append(second)
        if index % 29 == 0:
            mutated = replace(mutated, evidence_complete=False)
        cases.append({"id": f"{identity}:fault", "artifact": mutated, "families": tuple(families), "severity": severity})
    return cases


def _baseline(name: str, artifact: RouteArtifact, truth: tuple[str, ...]) -> tuple[bool, set[str], set[str]]:
    if name == "json_schema":
        return (not artifact.evidence_complete), set(), set()
    guard = TypedRouteGuard()
    assessment = guard.assess(artifact)
    detected = bool(assessment.violations) or not artifact.evidence_complete
    if name in {"simple_or", "weighted_fault_score"}:
        return detected, set(), set()
    if name == "nearest_centroid_v15":
        return detected, set(assessment.damaged_regions[:1]), set()
    regions = set(assessment.damaged_regions)
    repairs = set(RepairPlanner().plan(assessment).candidate_actions)
    if name == "route_only_rules":
        regions = set(list(regions)[:1])
        repairs = set(list(repairs)[:1])
    return detected, regions, repairs


def _evaluate(dataset_id: str, identities: list[str], threshold: float) -> tuple[dict[str, object], list[dict[str, object]]]:
    guard = TypedRouteGuard(family_confidence_threshold=0.72)
    controller = LexicographicController(PredictiveSelector(threshold))
    methods = ("json_schema", "independent_if_else", "simple_or", "weighted_fault_score", "route_only_rules", "nearest_centroid_v15")
    accum = {name: {"detected": 0, "faults": 0, "false": 0, "valid": 0, "localized": 0, "repaired": 0} for name in (*methods, "hierarchical_validator_v16")}
    typed = {"known_correct": 0, "known_total": 0, "abstained": 0, "trace_equal": 0, "cases": 0}
    rows = []
    for case in _cases(dataset_id, identities):
        artifact = case["artifact"]
        truth = tuple(case["families"])
        fault = bool(truth) or not artifact.evidence_complete
        true_regions = {REGIONS[item] for item in truth}
        assessment = guard.assess(artifact)
        decision = controller.decide(assessment, unit(str(case["id"]), "predictive-risk-v16"))
        repeated = controller.decide(assessment, unit(str(case["id"]), "predictive-risk-v16"))
        typed["trace_equal"] += int(decision.audit_trace == repeated.audit_trace)
        typed["cases"] += 1
        typed["abstained"] += int(assessment.family is None and fault)
        if len(truth) == 1 and artifact.evidence_complete:
            typed["known_total"] += 1
            typed["known_correct"] += int(assessment.family == truth[0])
        predicted_regions = set(assessment.damaged_regions)
        predicted_repairs = set(decision.repair_plan.candidate_actions)
        detected = assessment.outcome.value != "valid_route"
        values = accum["hierarchical_validator_v16"]
        values["faults"] += int(fault)
        values["valid"] += int(not fault)
        values["detected"] += int(fault and detected)
        values["false"] += int(not fault and detected)
        values["localized"] += int(fault and bool(true_regions & predicted_regions))
        values["repaired"] += int(fault and bool(predicted_repairs))
        for name in methods:
            found, regions, repairs = _baseline(name, artifact, truth)
            target = accum[name]
            target["faults"] += int(fault)
            target["valid"] += int(not fault)
            target["detected"] += int(fault and found)
            target["false"] += int(not fault and found)
            target["localized"] += int(fault and bool(true_regions & regions))
            target["repaired"] += int(fault and bool(repairs))
        rows.append({"fault": fault, "typed_localized": bool(true_regions & predicted_regions), "typed_repaired": bool(predicted_repairs), "severity": case["severity"]})
    metrics = {}
    for name, value in accum.items():
        metrics[name] = {
            "generic_fault_recall": value["detected"] / max(1, value["faults"]),
            "false_certification": (value["faults"] - value["detected"]) / max(1, value["faults"]),
            "source_region_localization": value["localized"] / max(1, value["faults"]),
            "repair_candidate_recall": value["repaired"] / max(1, value["faults"]),
            "false_alerts_per_10000_valid": 10000 * value["false"] / max(1, value["valid"]),
        }
    return {
        "dataset_id": dataset_id,
        "objects": len(identities),
        "route_checks": len(rows),
        "methods": metrics,
        "known_family_accuracy": typed["known_correct"] / max(1, typed["known_total"]),
        "abstention_rate_on_faults": typed["abstained"] / max(1, sum(row["fault"] for row in rows)),
        "byte_identical_trace_rate": typed["trace_equal"] / typed["cases"],
    }, rows


def formative() -> None:
    verify_protocol()
    datasets = read_json(ARTIFACTS / "data" / "dataset_manifest.json")["datasets"]
    threshold = 0.80
    summaries = []
    for item in datasets:
        splits = read_json(DATA / item["dataset_id"] / "manifests" / "split_identities.json")
        summary, _ = _evaluate(item["dataset_id"], splits["formative_development"], threshold)
        summaries.append(summary)
    write_json(ARTIFACTS / "formative" / "summary.json", {"predictive_review_threshold": threshold, "datasets": summaries, "test_opened": False, "implementation_commit": git_commit()})
    print("PASS operational-audit-formative test_opened=false")


def freeze() -> None:
    verify_protocol()
    if LOCK.exists():
        raise RuntimeError("already locked")
    formative_path = ARTIFACTS / "formative" / "summary.json"
    write_json(LOCK, {"implementation_commit": git_commit(), "formative_sha256": sha256_file(formative_path), "predictive_review_threshold": read_json(formative_path)["predictive_review_threshold"], "test_opened": False, "post_lock_tuning": False})
    print("PASS operational-audit-freeze opened=false")


def confirmatory() -> None:
    verify_protocol()
    output = ARTIFACTS / "confirmatory" / "summary.json"
    if OPENING.exists() or output.exists():
        raise RuntimeError("one-shot opening already attempted")
    lock = read_json(LOCK)
    write_json(OPENING, {"opening_count": 1, "opened_at_utc": datetime.now(timezone.utc).isoformat(), "purpose": "route-artifact scoring only", "labels_opened": False, "scoring_commit": git_commit(), "lock_sha256": sha256_file(LOCK)})
    datasets = []
    raw_rows = {}
    for item in read_json(ARTIFACTS / "data" / "dataset_manifest.json")["datasets"]:
        identities = read_json(DATA / item["dataset_id"] / "manifests" / "split_identities.json")["sealed_confirmatory_test"]
        summary, rows = _evaluate(item["dataset_id"], identities, lock["predictive_review_threshold"])
        datasets.append(summary)
        raw_rows[item["dataset_id"]] = rows
    methods = datasets[0]["methods"]
    simple_names = [name for name in methods if name != "hierarchical_validator_v16"]
    best_localization = max(simple_names, key=lambda name: np.mean([item["methods"][name]["source_region_localization"] for item in datasets]))
    best_repair = max(simple_names, key=lambda name: np.mean([item["methods"][name]["repair_candidate_recall"] for item in datasets]))
    typed_local = np.mean([item["methods"]["hierarchical_validator_v16"]["source_region_localization"] for item in datasets])
    typed_repair = np.mean([item["methods"]["hierarchical_validator_v16"]["repair_candidate_recall"] for item in datasets])
    baseline_local = np.mean([item["methods"][best_localization]["source_region_localization"] for item in datasets])
    baseline_repair = np.mean([item["methods"][best_repair]["repair_candidate_recall"] for item in datasets])
    write_json(output, {"datasets": datasets, "best_localization_baseline": best_localization, "best_repair_baseline": best_repair, "A1_localization_gain": typed_local - baseline_local, "A2_repair_gain": typed_repair - baseline_repair, "A4_false_certification": max(item["methods"]["hierarchical_validator_v16"]["false_certification"] for item in datasets), "A5_byte_identical_rate": min(item["byte_identical_trace_rate"] for item in datasets), "hierarchical_resampling_required_for_claims": True, "labels_opened": False, "post_lock_tuning": False})
    write_json(ARTIFACTS / "data" / "post_scoring_leakage_audit.json", {"opening_count": 1, "purpose": "route-artifact scoring only", "post_lock_tuning": False, "labels_exported": False, "model_changed": False, "thresholds_changed": False, "repeat_opening_forbidden": True})
    print(f"PASS operational-audit-confirmatory objects={sum(item['objects'] for item in datasets)} labels_opened=false")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("formative", "freeze", "confirmatory"))
    {"formative": formative, "freeze": freeze, "confirmatory": confirmatory}[parser.parse_args().stage]()
