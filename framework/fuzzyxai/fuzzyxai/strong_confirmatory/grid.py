"""Component-grid sensitivity summaries shared by all modalities."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def compare_grid_configurations(
    configurations: Mapping[str, Mapping[str, object]],
    *,
    reference: str = "default",
    risk_tolerance: float = 0.05,
) -> dict[str, object]:
    if reference not in configurations or len(configurations) < 2:
        raise ValueError("grid comparison requires a reference and at least one alternative")
    expected = configurations[reference]
    expected_action = np.asarray(expected["actions"])
    expected_representation = np.asarray(expected["representations"])
    expected_risk = np.asarray(expected["risk"], dtype=float)
    expected_top = [set(values) for values in expected["top_k"]]
    rows = []
    for name, candidate in configurations.items():
        actions = np.asarray(candidate["actions"])
        representations = np.asarray(candidate["representations"])
        risk = np.asarray(candidate["risk"], dtype=float)
        top = [set(values) for values in candidate["top_k"]]
        if not (len(actions) == len(expected_action) == len(representations) == len(risk) == len(top)):
            raise ValueError("grid outputs must be object-aligned")
        top_agreement = [len(left & right) / max(1, len(left | right)) for left, right in zip(expected_top, top, strict=True)]
        rows.append(
            {
                "configuration": name,
                "action_agreement": float(np.mean(actions == expected_action)),
                "representation_agreement": float(np.mean(representations == expected_representation)),
                "mean_absolute_risk_change": float(np.mean(np.abs(risk - expected_risk))),
                "top_k_jaccard": float(np.mean(top_agreement)),
            }
        )
    alternatives = [row for row in rows if row["configuration"] != reference]
    return {
        "phase": "formative_only",
        "reference": reference,
        "risk_tolerance": risk_tolerance,
        "configurations": rows,
        "formative_target_met": all(
            row["action_agreement"] >= 0.95
            and row["representation_agreement"] >= 0.90
            and row["mean_absolute_risk_change"] <= risk_tolerance
            for row in alternatives
        ),
        "confirmatory_claim_allowed": False,
    }
