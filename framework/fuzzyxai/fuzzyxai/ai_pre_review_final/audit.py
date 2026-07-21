"""Automated leakage, interpretability, traceability and distinguishability audit."""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from .contracts import DENYLIST, FinalStudyError, canonical_json, digest


SPECIAL_EVIDENCE_IDS = {
    "PREDICTION",
    "O-CONFIDENCE",
    "O-AGREEMENT",
    "O-STABILITY",
    "O-CHANNELS",
    "O-SHIFT",
    "O-APPLICABILITY",
}


def audit_blind_records(
    rows: list[dict[str, Any]],
    *,
    root: Path | None = None,
    expected_cases: int = 360,
    expected_records: int = 1080,
) -> dict[str, Any]:
    failures: list[str] = []
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    modality_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        case_id = str(row.get("case_id", "unknown"))
        variant_id = str(row.get("variant_id", "unknown"))
        label = f"{case_id}/{variant_id}"
        by_case[case_id].append(row)
        modality_counts[str(row.get("modality"))] += 1
        _audit_leakage(row, label, failures)
        _audit_evidence(row, label, failures)
        _audit_claims(row, label, failures)
        _audit_hash(row, label, failures)
        if root is not None:
            _audit_assets(row, label, root, failures)
    for case_id, variants in by_case.items():
        _audit_variants(case_id, variants, failures)
    if len(rows) != expected_records or len(by_case) != expected_cases:
        failures.append(f"study size mismatch: rows={len(rows)} cases={len(by_case)}")
    result = {
        "schema_version": "2.0",
        "status": "PASS" if not failures else "FAIL",
        "records": len(rows),
        "cases": len(by_case),
        "modality_variant_counts": dict(sorted(modality_counts.items())),
        "denylist_terms": list(DENYLIST),
        "outcome_leakage": any("denylist leakage" in item for item in failures),
        "claim_evidence_coverage_min": min((float(row.get("claim_evidence_coverage", 0.0)) for row in rows), default=0.0),
        "failures": failures,
    }
    if failures:
        raise FinalStudyError("blindness audit failed:\n" + "\n".join(failures[:30]))
    return result


def _audit_leakage(row: dict[str, Any], label: str, failures: list[str]) -> None:
    text = canonical_json(row).lower()
    method_terms = {"fuzzyxai", "shap", "lime", "anchors", "rulefit", "grad-cam", "integrated gradients"}
    for term in DENYLIST:
        if term in method_terms:
            leaked = re.search(rf"(?<![a-z]){re.escape(term)}(?![a-z])", text) is not None
        else:
            leaked = term in text
        if leaked:
            failures.append(f"{label}: denylist leakage: {term}")
    if "stratum" in row:
        failures.append(f"{label}: stratum key is public")
    prediction = row.get("prediction")
    if not isinstance(prediction, dict) or set(prediction) != {"display_label", "confidence"}:
        failures.append(f"{label}: prediction contains forbidden or missing fields")
    if "class_" in str(prediction).lower():
        failures.append(f"{label}: technical class label shown to domain user")


def _audit_evidence(row: dict[str, Any], label: str, failures: list[str]) -> None:
    items = row.get("interpretable_evidence")
    if not isinstance(items, list) or len(items) < 2:
        failures.append(f"{label}: fewer than two interpretable evidence items")
        return
    required = {"evidence_id", "display_name", "direction", "magnitude_normalized", "rank", "stability", "source_agreement", "evidence_refs", "limitations"}
    for item in items:
        if not isinstance(item, dict) or not required.issubset(item):
            failures.append(f"{label}: incomplete evidence item")
            continue
        if item["direction"] not in {"supports", "opposes", "neutral"}:
            failures.append(f"{label}: invalid evidence direction")
        for field in ("magnitude_normalized", "stability", "source_agreement"):
            value = item[field]
            if not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
                failures.append(f"{label}: invalid {field}")
        if not str(item["display_name"]).strip() or not item["evidence_refs"]:
            failures.append(f"{label}: evidence lacks readable name or references")
    reasons = row.get("candidate_explanation", {}).get("main_reasons", [])
    if not 2 <= len(reasons) <= 5:
        failures.append(f"{label}: explanation must show 2-5 reasons")
    fidelity = row.get("fidelity_metadata")
    required_fidelity = {"value", "scale_min", "scale_max", "higher_is_better", "interpretation_band", "metric_name"}
    if fidelity is not None and (not isinstance(fidelity, dict) or not required_fidelity.issubset(fidelity)):
        failures.append(f"{label}: incomplete fidelity metadata")


def _audit_claims(row: dict[str, Any], label: str, failures: list[str]) -> None:
    evidence_ids = {str(item["evidence_id"]) for item in row.get("interpretable_evidence", [])} | SPECIAL_EVIDENCE_IDS
    links = row.get("claim_evidence_links")
    if not isinstance(links, list) or not links:
        failures.append(f"{label}: claim-evidence graph is empty")
        return
    claim_ids = set()
    for link in links:
        claim_id = str(link.get("claim_id", ""))
        refs = set(map(str, link.get("evidence_ids", [])))
        if not claim_id or claim_id in claim_ids or not refs:
            failures.append(f"{label}: invalid or duplicate claim link")
        claim_ids.add(claim_id)
        if not refs.issubset(evidence_ids | {ref for item in row.get("interpretable_evidence", []) for ref in item.get("evidence_refs", [])}):
            failures.append(f"{label}: claim references unknown evidence")
    explanation = row.get("candidate_explanation", {})
    displayed_claims = {str(explanation.get("decision", {}).get("claim_id", "")), str(explanation.get("recommended_action", {}).get("claim_id", ""))}
    for field in ("main_reasons", "concerns", "limitations"):
        displayed_claims.update(str(item.get("claim_id", "")) for item in explanation.get(field, []))
    if not displayed_claims.issubset(claim_ids):
        failures.append(f"{label}: displayed claim lacks evidence link")
    if float(row.get("claim_evidence_coverage", 0.0)) != 1.0:
        failures.append(f"{label}: claim-evidence coverage is not complete")


def _audit_variants(case_id: str, variants: list[dict[str, Any]], failures: list[str]) -> None:
    if len(variants) != 3 or {row.get("variant_id") for row in variants} != {"X1", "X2", "X3"}:
        failures.append(f"{case_id}: three blind variants are required")
        return
    baseline = [row for row in variants if _variant_signature(row) == "baseline"]
    selective = [row for row in variants if _variant_signature(row) == "selective"]
    full = [row for row in variants if _variant_signature(row) == "full"]
    if not (len(baseline) == len(selective) == len(full) == 1):
        failures.append(f"{case_id}: A/B/C roles are not structurally distinguishable")
        return
    a_blocks = _observed_blocks(baseline[0])
    b_blocks = _observed_blocks(selective[0])
    c_blocks = _observed_blocks(full[0])
    if len(a_blocks ^ b_blocks) < 2 or len(b_blocks ^ c_blocks) < 2:
        failures.append(f"{case_id}: variants differ by fewer than two observed content blocks")
    if len(full[0]["interpretable_evidence"]) <= len(selective[0]["interpretable_evidence"]):
        failures.append(f"{case_id}: full variant has no additional evidence")
    if any(block in a_blocks for block in ("source_agreement", "prospective_action", "full_provenance")):
        failures.append(f"{case_id}: baseline contains system diagnostics")


def _variant_signature(row: dict[str, Any]) -> str:
    evidence_count = len(row.get("interpretable_evidence", []))
    provenance_count = len(row.get("candidate_explanation", {}).get("provenance_summary", []))
    detail = row.get("presentation", {}).get("detail")
    if evidence_count == 3 and provenance_count == 0 and detail == "short":
        return "baseline"
    if evidence_count == 4 and provenance_count == 2 and detail == "short":
        return "selective"
    if evidence_count == 5 and provenance_count >= 3 and detail == "full":
        return "full"
    return "unknown"


def _observed_blocks(row: dict[str, Any]) -> set[str]:
    explanation = row.get("candidate_explanation", {})
    conditions = row.get("observable_conditions", {})
    blocks = {"prediction", "reasons"}
    if "не предоставлено" not in str(conditions.get("source_agreement_summary", "")).lower():
        blocks.add("source_agreement")
    if explanation.get("concerns"):
        blocks.add("concerns")
    provenance_count = len(explanation.get("provenance_summary", []))
    if provenance_count:
        blocks.add("short_provenance")
    if provenance_count >= 3:
        blocks.add("full_provenance")
    if len(row.get("interpretable_evidence", [])) >= 5:
        blocks.add("extended_evidence")
    if any(item.get("claim_id") == "C-L3" for item in explanation.get("limitations", [])):
        blocks.add("applicability_limitation")
    if explanation.get("counterfactuals"):
        blocks.add("sensitivity")
    if _variant_signature(row) != "baseline":
        blocks.add("prospective_action")
    return blocks


def _audit_hash(row: dict[str, Any], label: str, failures: list[str]) -> None:
    expected = str(row.get("record_sha256", ""))
    copy = dict(row)
    copy["record_sha256"] = ""
    if expected != digest(canonical_json(copy)):
        failures.append(f"{label}: record SHA256 mismatch")


def _audit_assets(row: dict[str, Any], label: str, root: Path, failures: list[str]) -> None:
    if row.get("modality") != "image":
        return
    asset = row.get("observable_asset", {})
    reference = asset.get("thumbnail_ref")
    if not reference or not (root / str(reference)).is_file():
        failures.append(f"{label}: image thumbnail is missing")
