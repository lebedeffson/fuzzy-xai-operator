#!/usr/bin/env python3
"""Build a concise handoff from the measured formative evidence."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "release_evidence/strong_confirmatory/formative"
REPORT = ROOT / "reports/strong_confirmatory/FORMATIVE_HANDOFF.md"


def main() -> None:
    h3 = _load("H3_v2_selective")
    h5 = _load("H5_A_route_validity")
    h6 = _load("H6_A_planted_rules")
    h7 = _load("H7_stability")
    h8 = _load("H8_grid_sensitivity")
    h9 = _load("H9_scalability")
    typed = next(row for row in h5["methods"] if row["method"] == "typed_route_validity")
    h7_rows = "\n".join(
        f"| {row['modality']} | {str(row['formative_target_met']).lower()} | "
        f"{row['fidelity_noninferiority']['effect']:.4f} | {str(row['fidelity_noninferiority']['noninferior']).lower()} |"
        for row in h7["modalities"]
    )
    lines = [
        "# Strong confirmatory closure: formative handoff",
        "",
        "Status: **FORMATIVE ONLY**. Confirmatory data have not been opened and no new positive dissertation claim is allowed.",
        "",
        "## Immutable results",
        "",
        "- `H3-original = not_supported`;",
        "- `H5-P-original = not_supported`;",
        "- `H6-general = not_supported`.",
        "",
        "## Measured formative status",
        "",
        "| Experiment | Formative target | Scope |",
        "| --- | --- | --- |",
        f"| H3-v2 | {str(h3['formative_target_met']).lower()} | OOF development policy; confirmatory criterion intentionally unavailable |",
        f"| H5-A | {str(h5['formative_target_met']).lower()} | controlled route faults only |",
        f"| H6-A | {str(h6['formative_target_met']).lower()} | semisynthetic labels on real tabular features |",
        f"| H7 | {str(h7['formative_target_met']).lower()} | development stability and fidelity |",
        f"| H8 | {str(h8['formative_target_met']).lower()} | controlled component-grid perturbations |",
        f"| H9 | {str(h9['formative_target_met']).lower()} | cached streaming operator layer; explainer excluded |",
        "",
        "## H5-A controlled measurement",
        "",
        f"Typed route validity: F1 `{typed['f1']:.6f}`, false certification `{typed['false_certification']:.6f}`, "
        f"source localization `{typed['fault_source_localization']:.6f}`, invalid-action recall `{typed['invalid_action_recall']:.6f}`.",
        "",
        "## H6-A planted-rule measurement",
        "",
        f"Configurations: `{h6['n_configurations']}`; planted-rule detection `{h6['planted_rule_detection_rate']:.6f}`; "
        f"mean specific effect `{h6['mean_specific_effect']:.6f}`. H6-B remains not run.",
        "",
        "## H7 fidelity boundary",
        "",
        "| Modality | Formative target | Fidelity effect | Non-inferior |",
        "| --- | --- | ---: | --- |",
        h7_rows,
        "",
        "The tabular and text profiles fail fidelity non-inferiority. This negative result is preserved.",
        "",
        "## H9 operator-layer scaling",
        "",
        f"Maximum measured size: `{max(row['n_objects'] for row in h9['measurements']):,}` objects; empirical exponent "
        f"`{h9['empirical_scaling_exponent']:.6f}`; deterministic repeat `{str(h9['deterministic_repeat']).lower()}`. "
        "The measurement excludes local-explainer cost and is not an end-to-end latency claim.",
        "",
        "## Blocking gates",
        "",
        "- sealed independent confirmatory dataset manifests are absent;",
        "- formative AI-review acceptance is absent;",
        "- domain-language, comprehension and expert-action gates are open;",
        "- `chapter4-final` and stable release remain blocked.",
        "",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"PASS: strong_formative_report path={REPORT.relative_to(ROOT)}")


def _load(name: str) -> dict[str, object]:
    return json.loads((EVIDENCE / f"{name}.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
