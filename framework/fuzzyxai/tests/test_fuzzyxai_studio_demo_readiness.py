from pathlib import Path

from fuzzyxai.studio.operator_scenarios import build_ecosystem_entities, build_report, load_scenarios


def _scenario(scenario_id: str) -> dict:
    return next(s for s in load_scenarios() if s["scenario_id"] == scenario_id)


def _entities() -> dict[str, dict]:
    ecosystem = build_ecosystem_entities(load_scenarios())
    out: dict[str, dict] = {}
    for collection in ["articles", "models", "scenarios", "operators"]:
        for entity in ecosystem[collection]:
            out[entity.get("id") or entity.get("scenario_id")] = entity
    return out


def test_hybrid_xiris_demo_integrity() -> None:
    scenario = _scenario("hybrid_xiris")

    assert scenario["scenario_id"] == "hybrid_xiris"
    assert scenario["summary"]["objects_total"] == 1000
    assert scenario["summary"]["accept"] == 612
    assert scenario["summary"]["lower_confidence"] == 201
    assert scenario["summary"]["block"] == 187

    pipeline = scenario["pipeline"]
    assert [node["node_id"] for node in pipeline] == [
        "input_artifact",
        "adapter",
        "explanation_object",
        "alignment",
        "f_selector",
        "reduction",
        "risk_observer",
        "action",
        "report",
    ]

    assert any(node["node_id"] == "risk_observer" and node["status"] == "blocked" for node in pipeline)
    assert any(node["node_id"] == "action" and node["status"] == "blocked" for node in pipeline)


def test_hybrid_xiris_global_diagnostics_are_deduplicated() -> None:
    report = build_report(_scenario("hybrid_xiris"))

    diagnostics = report["diagnostics"]
    assert len(diagnostics) == 1

    diagnostic = diagnostics[0]
    assert diagnostic["diagnostic_id"] == "D_quality_source_conflict"
    assert diagnostic["legacy_id"] == "D_source_conflict"
    assert diagnostic["type"] == "critical_rupture"
    assert diagnostic["recommended_action"] == "block"
    assert diagnostic["occurrences"] == ["alignment", "risk_observer", "action"]
    assert diagnostic["global"] is True


def test_hybrid_xiris_scenario_config_has_diagnostics_summary() -> None:
    scenario = _scenario("hybrid_xiris")
    summary = scenario["diagnostics_summary"]

    assert len(summary) == 1
    assert summary[0]["diagnostic_id"] == "D_quality_source_conflict"
    assert summary[0]["legacy_id"] == "D_source_conflict"
    assert summary[0]["occurrences"] == ["alignment", "risk_observer", "action"]
    assert summary[0]["global"] is True


def test_ecosystem_entities_have_valid_links() -> None:
    entities = _entities()

    for entity in entities.values():
        for link in entity.get("links", []):
            assert link["targetId"] in entities, f"Broken link from {entity.get('id')} to {link['targetId']}"


def test_hybrid_xiris_is_connected_to_models_and_operators() -> None:
    scenario = _entities()["hybrid_xiris"]
    linked_ids = {link["targetId"] for link in scenario["links"]}

    assert "model_iris_quality" in linked_ids
    assert "model_iris_matcher" in linked_ids
    assert "build_Ek" in linked_ids
    assert "T_ij" in linked_ids
    assert "select_F" in linked_ids
    assert "risk_observer" in linked_ids


def test_raw_json_is_only_in_technical_trace_drawer() -> None:
    source = Path("apps/fuzzyxai_studio.py").read_text(encoding="utf-8")

    assert "Технический след" in source
    assert source.count("ui.code(") == 1
    assert source.index("Технический след") < source.index("ui.code(")


def test_studio_source_contains_demo_ready_labels() -> None:
    source = Path("apps/fuzzyxai_studio.py").read_text(encoding="utf-8")

    for label in ["HYBRID-XIRIS", "БЛОКИРОВКА", "Риск-наблюдатель", "Критические пропуски", "Технический след", "выбран"]:
        assert label in source
