#!/usr/bin/env python3
"""Build a claim-safe technical handoff from measured formative evidence."""

from __future__ import annotations

from common import FORMATIVE, ROOT, load_json, sha256


def main() -> None:
    summary = load_json(FORMATIVE / "summary.json")
    h3 = load_json(FORMATIVE / "H3_practical/summary.json")
    h5 = load_json(FORMATIVE / "H5_A_route_validity/summary.json")
    h7 = load_json(FORMATIVE / "H7_canonical_projection/summary.json")
    h9 = load_json(FORMATIVE / "H9_scaling/summary.json")
    report = ROOT / "reports/final_practical_closure/FORMATIVE_HANDOFF.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# FuzzyXAI final practical closure: formative handoff",
        "",
        "> Status: FORMATIVE DEVELOPMENT ONLY. Confirmatory test data have not been opened and no new positive confirmatory claim is allowed.",
        "",
        "## Boundaries",
        "",
        "- H3-original, H5-P-original and H6-general remain `not_supported`.",
        "- External domain-language, comprehension and expert-action claims are removed from technical release scope.",
        "- AI formative review is not external expert validation.",
        "- H9 measures the operator layer separately from local-explainer cost.",
        "",
        "## Measured formative status",
        "",
        f"- H3 practical formative target met: `{str(h3['formative_target_met']).lower()}`.",
        f"- H5-A controlled route-validity target met: `{str(h5['formative_target_met']).lower()}`; natural failures: `{h5['natural_failures']['status']}`.",
        f"- H7-A exact canonical hash rate: `{h7['H7_A']['exact_source_hash_rate']}`; H7-B confirmatory status: `{h7['H7_B']['confirmatory_status']}`.",
        f"- H9 smoke maximum: `{h9['maximum_objects']}` objects; local explainer included: `{str(h9['cached_operator_layer']['local_explainer_included']).lower()}`.",
        f"- H6-B: `{summary['H6_B_status']}`.",
        "",
        "## Evidence",
        "",
        f"- Formative summary: `{(FORMATIVE / 'summary.json').relative_to(ROOT)}`",
        f"- SHA256: `{sha256(FORMATIVE / 'summary.json')}`",
        "- Run `make practical-controller-formative-check` to verify every package checksum and Parquet file.",
        "- Run `make practical-controller-freeze`; it must remain BLOCKED until real sealed inputs exist.",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"PASS: practical_formative_handoff path={report.relative_to(ROOT)} confirmatory_claim=false")


if __name__ == "__main__":
    main()

