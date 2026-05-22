from apps import defense_demo


def test_defense_demo_computes_summary():
    defense_demo.build_plan_from_state()
    defense_demo.recompute()
    report = defense_demo.summary_report()
    assert report["selected_class"] == "FML-audit"
    assert report["composition"]["gamma"] is not None
    assert 0 <= report["composition"]["gamma"] <= 1


def test_defense_demo_conflict_produces_diagnostic():
    defense_demo.build_plan_from_state()
    defense_demo.STATE["conflict"] = True
    defense_demo.recompute()
    report = defense_demo.summary_report()
    assert report["composition"]["diagnostic"] == "D_ij"
    defense_demo.STATE["conflict"] = False


def test_defense_demo_visual_explainers_are_not_empty():
    defense_demo.build_plan_from_state()
    defense_demo.recompute()
    membership = defense_demo.plan_membership_demo_figure()
    layers = defense_demo.representation_layers_figure(defense_demo.STATE["explanation"])
    selection = defense_demo.selection_figure()
    model = defense_demo.model_contribution_figure()
    breakdown = defense_demo.composition_breakdown_figure(defense_demo.STATE["composition"])
    assert len(membership.data) >= 3
    assert membership.layout.shapes
    assert len(layers.data) >= 3
    assert len(selection.data) >= 3
    assert len(model.data) >= 1
    assert len(breakdown.data) >= 1


def test_composition_breakdown_explains_gamma_parts():
    defense_demo.build_plan_from_state()
    defense_demo.recompute()
    rows = defense_demo.composition_breakdown_rows(defense_demo.STATE["composition"])
    assert {row["part"] for row in rows} >= {"активные правила", "след tau", "неопределённость"}
    assert all("contribution" in row for row in rows)


def test_defense_demo_uses_sklearn_model_risk():
    defense_demo.STATE["case_index"] = 2
    defense_demo.build_plan_from_state()
    report = defense_demo.STATE["model_report"]
    case = defense_demo.current_case_row()
    assert report["features"] == ["age", "pressure", "marker"]
    assert 0 <= report["risk"] <= 1
    assert abs(case["risk_score"] - report["risk"]) < 1e-9


def test_operator_benchmark_compares_with_and_without_operator():
    defense_demo.STATE["operator_benchmark"] = None
    report = defense_demo.operator_benchmark_report()
    assert report["dataset"] == "sklearn breast_cancer"
    assert report["model"] == "RandomForestClassifier"
    assert report["accuracy"] > 0.8
    assert report["roc_auc"] > 0.9
    assert report["without_operator"]["detects_term_conflict"] is False
    assert report["with_operator"]["detects_term_conflict"] is True
    assert len(defense_demo.operator_added_value_figure().data) == 2


def test_defense_demo_has_full_pipeline_report_for_home_page():
    defense_demo.STATE["full_pipeline_report"] = None
    report = defense_demo.full_pipeline_report()
    assert report["status"] == "PASS"
    assert report["composition"]["index"] > 0
    assert report["risk_observer"]["risk_reduction"] > 0
    assert report["stages"][0]["id"] == "D"
