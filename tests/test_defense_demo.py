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
