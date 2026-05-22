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
    assert len(membership.data) >= 3
    assert membership.layout.shapes
    assert len(layers.data) >= 3
    assert len(selection.data) >= 3
