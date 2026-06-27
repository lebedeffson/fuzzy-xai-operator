import csv
from pathlib import Path

import pytest


TABLE_DIR = Path("reports/chapter5/studio_tables")


def _rows(name: str) -> list[dict[str, str]]:
    return list(csv.DictReader((TABLE_DIR / name).open(encoding="utf-8")))


def test_table_5_2_explainplan_has_required_parameters() -> None:
    rows = _rows("table_5_2_explainplan.csv")
    params = {row["parameter"] for row in rows}
    required = {
        "explain_plan_version",
        "explain_plan_hash",
        "gamma_max",
        "delta_max",
        "w_d_mu",
        "w_d_R",
        "w_d_u",
        "w_d_tau",
        "w_model_signal",
        "w_block_rule",
        "w_source_conflict",
        "w_reduction_component",
        "kappa_delta",
        "r_delta",
        "theta_1",
        "theta_2",
        "theta_3",
        "theta_4",
        "selected_class",
        "action_rule",
    }
    assert required <= params


def test_table_5_3_membership_has_functions() -> None:
    rows = _rows("table_5_3_membership.csv")
    assert {"variable", "term", "input_value", "membership_function", "computed_membership"} <= set(rows[0])
    assert all(row["membership_function"].strip() for row in rows)


def test_table_5_4_de_sums_to_gamma() -> None:
    rows = _rows("table_5_4_dE.csv")
    assert {"component", "value", "weight", "contribution", "definition"} <= set(rows[0])
    assert sum(float(row["contribution"]) for row in rows) == pytest.approx(0.351)


def test_table_5_6_risk_decomposition_sums_to_rho() -> None:
    rows = _rows("table_5_6_risk_decomposition.csv")
    subtotal = sum(float(row["contribution"]) for row in rows if row["component"] != "total")
    total = next(row for row in rows if row["component"] == "total")
    assert subtotal == pytest.approx(0.800)
    assert float(total["contribution"]) == pytest.approx(0.800)
