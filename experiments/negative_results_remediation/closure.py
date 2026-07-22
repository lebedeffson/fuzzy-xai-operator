from __future__ import annotations

import argparse
import zipfile

from .common import ARTIFACTS, CONFIG, FROZEN_NEGATIVE_CLAIMS, ROOT, evidence_record, git_commit, read_json, require_file, sha256_file, verify_protocol, write_json


def statistics() -> None:
    required = (
        ARTIFACTS / "h3" / "confirmatory_status.json",
        ARTIFACTS / "h5" / "summary.json",
        ARTIFACTS / "h6" / "detectability_summary.json",
        ARTIFACTS / "h6" / "real_rule_effects.json",
        ARTIFACTS / "replay" / "summary.json",
    )
    for path in required:
        require_file(path, "statistics require all registered result families")
    replay = read_json(required[-1])
    h5 = read_json(required[1])
    h6_envelope = read_json(required[2])
    summary = {
        "protocol_sha256": verify_protocol(),
        "unit_of_analysis": {"H3-R4": "replay_event_clustered_by_phase", "H5-A2": "registered_fault_case", "H6-envelope": "registered_grid_configuration"},
        "H3-R4": replay["statistics"],
        "H5-A2": {
            "typed_registered_recall": h5["typed_validator"]["registered_recall"],
            "simple_or_registered_recall": h5["simple_or"]["registered_recall"],
            "absolute_recall_gain": h5["typed_validator"]["registered_recall"] - h5["simple_or"]["registered_recall"],
            "false_certification": h5["typed_validator"]["false_certification"],
            "minimal_cut_exact_rate": h5["typed_validator"]["minimal_cut_solver_exact_rate"],
        },
        "H6-envelope": h6_envelope,
        "multiple_comparison": "Holm adjustment is reported for the single preregistered H3-R4 primary comparison; no unadjusted positive H6 claim is enabled.",
        "independent_confirmatory_claims_available": False,
    }
    write_json(ARTIFACTS / "statistics" / "summary.json", summary)
    print("PASS remediation-statistics independent_confirmatory=false")


def claims() -> None:
    require_file(ARTIFACTS / "statistics" / "summary.json", "claim generation requires statistics")
    h3 = read_json(ARTIFACTS / "h3" / "confirmatory_status.json")
    h5 = read_json(ARTIFACTS / "h5" / "summary.json")
    h6 = read_json(ARTIFACTS / "h6" / "real_rule_effects.json")
    replay = read_json(ARTIFACTS / "replay" / "summary.json")
    registry = {
        "schema_version": "1.0",
        "protocol_sha256": verify_protocol(),
        "commit": git_commit(),
        "immutable_claims": FROZEN_NEGATIVE_CLAIMS,
        "new_claims": {
            "H3-R1": h3["H3-R1"],
            "H3-R2": h3["H3-R2"],
            "H3-R3": h3["H3-R3"],
            "H3-R4": replay["H3-R4"],
            "H5-P2": h5["H5-P2"],
            "H5-P3": h5["H5-P3"],
            "H5-A2": h5["H5-A2"],
            "H6-R1": h6["H6-R1"],
            "H6-R2": h6["H6-R2"],
            "H6-R3": h6["H6-R3"],
            "H6-R4": h6["H6-R4"],
            "H6-R5": h6["H6-R5"],
        },
        "manual_positive_override_allowed": False,
        "allowed_summary": "The remediation architecture is implemented. H3-R4 is bounded to controlled replay and H5-A2 to the registered fault library. Independent H3 and real-rule confirmatory claims remain unavailable.",
        "forbidden_summary": "The new controller universally fixes the negative v1.3.0 results.",
    }
    if registry["immutable_claims"] != FROZEN_NEGATIVE_CLAIMS:
        raise RuntimeError("immutable negative claim changed")
    write_json(ARTIFACTS / "claims" / "negative_remediation_claim_registry.json", registry)
    print("PASS remediation-claims immutable_negatives=3 manual_override=false")


def evidence_map() -> None:
    records = []
    mapping = {
        "data/dataset_manifest.json": ("NR-E01", "descriptive", ["H6-R1", "H6-R2", "H6-R3"]),
        "data/leakage_audit.json": ("NR-E02", "audit", ["H3-R1", "H3-R2", "H3-R3"]),
        "lock/negative_remediation_lock.json": ("NR-E03", "protocol_lock", []),
        "h3/confirmatory_status.json": ("NR-E04", "blocked", ["H3-R1", "H3-R2", "H3-R3"]),
        "replay/summary.json": ("NR-E05", "controlled_replay", ["H3-R4"]),
        "replay/raw_results.npz": ("NR-E06", "raw_controlled_replay", ["H3-R4"]),
        "h5/summary.json": ("NR-E07", "controlled_confirmatory", ["H5-A2", "H5-P2", "H5-P3"]),
        "h5/raw_cases.json": ("NR-E08", "raw_controlled_faults", ["H5-A2"]),
        "h6/detectability_summary.json": ("NR-E09", "controlled_formative", ["H6-R1"]),
        "h6/detectability_rows.json": ("NR-E10", "raw_controlled_formative", ["H6-R1"]),
        "h6/real_rule_effects.json": ("NR-E11", "exploratory", ["H6-R1", "H6-R2", "H6-R3", "H6-R4", "H6-R5"]),
        "statistics/summary.json": ("NR-E12", "statistical_summary", ["H3-R4", "H5-A2"]),
        "claims/negative_remediation_claim_registry.json": ("NR-E13", "claim_registry", list(FROZEN_NEGATIVE_CLAIMS)),
    }
    for relative, (evidence_id, status, claim_ids) in mapping.items():
        path = ARTIFACTS / relative
        require_file(path, "evidence map is incomplete")
        records.append(evidence_record(evidence_id, path, status=status, claim_ids=claim_ids))
    write_json(ARTIFACTS / "evidence_map.json", {"schema_version": "1.0", "records": records})
    print(f"PASS remediation-evidence-map records={len(records)}")


def chapter() -> None:
    require_file(ARTIFACTS / "evidence_map.json", "chapter supplement requires evidence map")
    claims_value = read_json(ARTIFACTS / "claims" / "negative_remediation_claim_registry.json")
    replay = read_json(ARTIFACTS / "replay" / "summary.json")
    h5 = read_json(ARTIFACTS / "h5" / "summary.json")
    h6 = read_json(ARTIFACTS / "h6" / "real_rule_effects.json")
    lines = [
        "# Дополнение к главе 4: устранение методологических причин отрицательных результатов",
        "",
        f"Кодовая основа: `{git_commit()}`. Протокол: `{verify_protocol()}`.",
        "",
        "Старые результаты версии v1.3.0 не пересматривались: H3-original, H5-P-original и H6-general остаются неподтвержденными.",
        "",
        "## Иерархический контроллер",
        "",
        "Реализованы четыре независимо калибруемые головы риска, детерминированный hard guard и глобальное распределение бюджета по ожидаемому снижению потерь.",
        f"Независимая проверка H3-R1-H3-R3: **{claims_value['new_claims']['H3-R1']}**. Метки независимого sealed-набора не открывались.",
        f"В контролируемом временном replay из {replay['events']:,} событий H3-R4 получила статус **{replay['H3-R4']}**. Этот результат не переносится на наблюдаемый производственный поток.",
        "",
        "## Сертификат и диагностический разрез",
        "",
        f"H5-A2: **{h5['H5-A2']}**. Область вывода ограничена зарегистрированной библиотекой; специфическая идентификация произвольных неизвестных типов отказа не подтверждена.",
        f"H5-P2: **{h5['H5-P2']}**; H5-P3: **{h5['H5-P3']}**.",
        "",
        "## Раздельные эффекты правил",
        "",
        f"Non-refit, refit и conditional estimands рассчитаны на двух реальных наборах как exploratory evidence. H6-R4: **{h6['H6-R4']}**; H6-R5: **{h6['H6-R5']}**.",
        "",
        "## Итоговая граница",
        "",
        "Новая версия устраняет смешение targets и estimands на уровне архитектуры, но не превращает отсутствие независимого подтверждения в положительный научный результат.",
    ]
    output = ARTIFACTS / "chapter4" / "negative_results_remediation_supplement.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("PASS remediation-chapter4 generated_from_claim_registry=true")


def release_check() -> None:
    verify_protocol()
    require_file(ARTIFACTS / "chapter4" / "negative_results_remediation_supplement.md", "release check")
    registry = read_json(ARTIFACTS / "claims" / "negative_remediation_claim_registry.json")
    lock = read_json(ARTIFACTS / "lock" / "negative_remediation_lock.json")
    leakage = read_json(ARTIFACTS / "data" / "leakage_audit.json")
    if registry["immutable_claims"] != FROZEN_NEGATIVE_CLAIMS:
        raise RuntimeError("old negative results were relabeled")
    if registry["manual_positive_override_allowed"]:
        raise RuntimeError("manual positive override is enabled")
    if lock["formative_iteration_count"] > lock["maximum_formative_iterations"]:
        raise RuntimeError("formative stop rule exceeded")
    if leakage["test_labels_in_feature_manifest"] or lock["confirmatory_test_opened"]:
        raise RuntimeError("test leakage or unauthorized opening detected")
    source = (ARTIFACTS / "chapter4" / "negative_results_remediation_supplement.md").read_text(encoding="utf-8")
    if "universally fixes" in source or "универсально подтверж" in source:
        raise RuntimeError("unsupported universal claim detected")
    report = {
        "status": "pass_technical_research_candidate",
        "protocol_hash_valid": True,
        "immutable_negative_claims_valid": True,
        "formative_stop_rule_valid": True,
        "test_leakage_absent": True,
        "independent_confirmatory_h3_complete": False,
        "technical_release_allowed": True,
        "positive_scientific_release_allowed": False,
    }
    write_json(ARTIFACTS / "validation_report.json", report)
    print("PASS remediation-release-check technical=true independent_confirmatory=false")


def one_zip() -> None:
    release_check()
    short = git_commit()[:12]
    output = ROOT / "release_artifacts" / f"fuzzyxai-negative-results-remediation-{short}.zip"
    output.parent.mkdir(parents=True, exist_ok=True)
    include = [
        CONFIG / "negative_remediation_protocol.json",
        CONFIG / "negative_remediation_dataset_manifest.json",
        CONFIG / "negative_remediation_split_manifest.json",
        CONFIG / "negative_remediation_feature_manifest.json",
        CONFIG / "negative_remediation_claim_registry.json",
        ROOT / "Makefile",
    ]
    include.extend(path for path in (ROOT / "framework/fuzzyxai/fuzzyxai").glob("{audit_certificate,diagnostic_cut,practical_controller_v2,rule_effects_v2,replay}/*") if path.is_file())
    # pathlib glob does not expand braces; add module trees explicitly.
    for module in ("audit_certificate", "diagnostic_cut", "practical_controller_v2", "rule_effects_v2", "replay"):
        include.extend(path for path in (ROOT / "framework/fuzzyxai/fuzzyxai" / module).rglob("*.py"))
    include.extend((ROOT / "experiments/negative_results_remediation").rglob("*.py"))
    include.extend((ROOT / "tests/negative_remediation").rglob("*.py"))
    include.extend(path for path in ARTIFACTS.rglob("*") if path.is_file())
    unique = sorted(set(include), key=lambda path: str(path.relative_to(ROOT)))
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in unique:
            archive.write(path, path.relative_to(ROOT))
    digest = sha256_file(output)
    checksum = output.with_suffix(".sha256")
    checksum.write_text(f"{digest}  {output.name}\n", encoding="utf-8")
    print(f"PASS remediation-one-zip path={output} sha256={digest} files={len(unique)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("statistics", "claims", "evidence", "chapter", "check", "zip"))
    args = parser.parse_args()
    globals()[{"statistics": "statistics", "claims": "claims", "evidence": "evidence_map", "chapter": "chapter", "check": "release_check", "zip": "one_zip"}[args.stage]]()


if __name__ == "__main__":
    main()
