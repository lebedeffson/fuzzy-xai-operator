"""Generate leakage-free reviewer records and a separate encrypted answer key."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from .contracts import METHODS, FinalStudyError, canonical_json, digest, read_jsonl, sha256_file, write_jsonl


def build_final_blind_study(root: Path, *, secret_path: Path | None = None) -> dict[str, Any]:
    config = json.loads((root / "configs/ai_pre_review_final/config.json").read_text(encoding="utf-8"))
    old_source = {row["case_id"]: row for row in read_jsonl(root / "study/ai_pre_review/source_case_evidence.jsonl")}
    evidence = []
    for modality in config["modalities"]:
        evidence.extend(read_jsonl(root / f"study/ai_pre_review_final/evidence/{modality}.jsonl"))
    if len(evidence) != 360 or len({row["case_id"] for row in evidence}) != 360:
        raise FinalStudyError("final evidence must contain 360 unique cases")
    secret_file = secret_path or Path.home() / ".local/share/fuzzyxai/ai_pre_review_final_blinding_secret"
    secret = _load_or_create_secret(secret_file)
    commit = _git(root, "rev-parse", "HEAD")
    public_rows: list[dict[str, Any]] = []
    hidden_rows: list[dict[str, Any]] = []
    for evidence_row in sorted(evidence, key=lambda row: str(row["case_id"])):
        case_id = str(evidence_row["case_id"])
        source = old_source[case_id]
        order = _blind_order(secret, case_id)
        for position, method in enumerate(order, 1):
            variant_id = f"X{position}"
            public_rows.append(_public_record(evidence_row, source, method, variant_id))
            hidden_rows.append(_hidden_record(source, method, variant_id))
    _validate_counts(public_rows)
    study = root / "study/ai_pre_review_final"
    reviewer_path = study / "reviewer_cases.jsonl"
    write_jsonl(reviewer_path, public_rows)
    _publish_formative_inputs(root, study, public_rows, old_source)
    private_dir = study / "private"
    private_dir.mkdir(parents=True, exist_ok=True)
    plaintext = private_dir / "hidden_scoring_key.tmp.jsonl"
    write_jsonl(plaintext, hidden_rows)
    encrypted = private_dir / "hidden_scoring_key.enc"
    _encrypt(root, secret_file, plaintext, encrypted)
    plaintext.unlink()
    batches = _build_batches(root, public_rows, int(config["batch_case_limit"]))
    manifest = {
        "schema_version": "2.0",
        "study_id": config["study_id"],
        "stage": "formative",
        "frozen_q1_commit": config["frozen_q1_commit"],
        "selective_observer_commit": config["selective_observer_commit"],
        "technical_ai_pipeline_commit": config["technical_ai_pipeline_commit"],
        "final_ai_human_commit": commit,
        "reviewer_cases": len({row["case_id"] for row in public_rows}),
        "reviewer_variants": len(public_rows),
        "reviewer_cases_sha256": sha256_file(reviewer_path),
        "hidden_scoring_key_sha256": sha256_file(encrypted),
        "hidden_scoring_key_in_public_bundle": False,
        "method_identity_in_public_records": False,
        "generated_from_commit_timestamp": _git(root, "show", "-s", "--format=%cI", "HEAD"),
        "batches": batches,
    }
    manifest_path = study / "blind_batch_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _build_master_markdown(study / "BLIND_REVIEW_MASTER_LOG.md", public_rows, manifest)
    return manifest


def _publish_formative_inputs(
    root: Path,
    study: Path,
    rows: list[dict[str, Any]],
    old_source: dict[str, dict[str, Any]],
) -> None:
    public_dir = study / "public_formative"
    if public_dir.exists():
        shutil.rmtree(public_dir)
    public_dir.mkdir(parents=True)
    formative_ids = {case_id for case_id, source in old_source.items() if source["split"] == "formative"}
    public_rows: list[dict[str, Any]] = []
    for source_row in rows:
        if source_row["case_id"] not in formative_ids:
            continue
        row = json.loads(json.dumps(source_row, ensure_ascii=False))
        asset = row.get("observable_asset")
        if isinstance(asset, dict) and asset.get("thumbnail_ref"):
            original_ref = Path(str(asset["thumbnail_ref"]))
            source_asset = study / original_ref
            public_ref = Path("assets/image") / original_ref.name
            target_asset = public_dir / public_ref
            target_asset.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_asset, target_asset)
            asset["thumbnail_ref"] = public_ref.as_posix()
            row["record_sha256"] = ""
            row["record_sha256"] = digest(canonical_json(row))
        public_rows.append(row)
    if len(formative_ids) != 240 or len(public_rows) != 720:
        raise FinalStudyError("public formative input must contain 240 cases and 720 variants")
    output = public_dir / "reviewer_cases.jsonl"
    write_jsonl(output, public_rows)
    manifest = {
        "schema_version": "2.0",
        "stage": "formative",
        "case_count": len(formative_ids),
        "variant_count": len(public_rows),
        "modalities": sorted({str(row["modality"]) for row in public_rows}),
        "reviewer_cases_sha256": sha256_file(output),
        "confirmatory_material_included": False,
        "hidden_scoring_key_included": False,
        "source_commit": _git(root, "rev-parse", "HEAD"),
    }
    (public_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _public_record(evidence: dict[str, Any], source: dict[str, Any], method: str, variant_id: str) -> dict[str, Any]:
    items = list(evidence["interpretable_evidence"])
    agreement = float(evidence["source_summary"]["agreement"])
    stability = float(evidence["source_summary"]["stability"])
    confidence = float(evidence["prediction"]["confidence"])
    unavailable = _observable_unavailable(source, method)
    conditions = {
        "available_channels": _available_channels(method),
        "unavailable_channels": unavailable,
        "detected_shift_summary": _shift_summary(evidence, method),
        "stability_summary": (
            "Отдельная проверка устойчивости причин не предоставлена."
            if method == METHODS[0]
            else _quality_sentence("Устойчивость причин", stability)
        ),
        "source_agreement_summary": (
            "Сопоставление нескольких источников не предоставлено."
            if method == METHODS[0]
            else _quality_sentence("Согласие источников", agreement)
        ),
        "observable_condition_refs": ["O-CONFIDENCE", "O-AGREEMENT", "O-STABILITY", "O-CHANNELS", "O-SHIFT"],
    }
    if method == METHODS[0]:
        shown_items = items[:3]
        reasons = _reason_cards(shown_items, max_items=3)
        concerns: list[dict[str, Any]] = []
        limitations = [{"text": "Локальные вклады описывают чувствительность модели, а не причинное устройство объекта.", "claim_id": "C-L1", "evidence_ids": ["E1"]}]
        action = "Использовать прогноз как информационный сигнал и проверить его по правилам предметной задачи."
        provenance = []
        counterfactuals = []
        claim_links = _claim_links(evidence, reasons, concerns, limitations, action, "O-CONFIDENCE")
    elif method == METHODS[1]:
        shown_items = items[:4]
        reasons = _reason_cards(shown_items, max_items=4)
        concerns = _observable_concerns(confidence, agreement, stability, unavailable, full=False)
        limitations = _limitations(unavailable, extended=False)
        action, action_ref = _prospective_action(confidence, agreement, stability, unavailable)
        provenance = ["Версия набора и объекта зафиксирована хэшами.", "Причины связаны с видимыми evidence ID."]
        counterfactuals = [{"text": f"Наиболее чувствительный наблюдаемый фактор: {shown_items[0]['display_name']}.", "evidence_ids": [shown_items[0]["evidence_id"]], "kind": "sensitivity_not_recommendation"}]
        claim_links = _claim_links(evidence, reasons, concerns, limitations, action, action_ref)
    else:
        shown_items = items[:5]
        reasons = _reason_cards(shown_items, max_items=5)
        concerns = _observable_concerns(confidence, agreement, stability, unavailable, full=True)
        limitations = _limitations(unavailable, extended=True)
        action, action_ref = _prospective_action(confidence, agreement, stability, unavailable)
        provenance = [
            f"Dataset SHA256: {evidence['dataset_sha256']}",
            f"Evidence SHA256: {evidence['evidence_sha256']}",
            "Каждая показанная причина связана с одним или несколькими evidence ID.",
        ]
        counterfactuals = [{"text": f"Замена области или признака «{shown_items[0]['display_name']}» reference-значением меняет поддержку прогноза; это анализ чувствительности, не рекомендация.", "evidence_ids": shown_items[0]["evidence_refs"], "kind": "sensitivity_not_recommendation"}]
        claim_links = _claim_links(evidence, reasons, concerns, limitations, action, action_ref)
    decision = f"Модель отнесла объект к категории «{evidence['prediction']['display_label']}» с уверенностью {confidence:.2f}."
    decision_claim = {"claim_id": "C-D1", "text": decision, "evidence_ids": ["PREDICTION"]}
    claim_links.insert(0, decision_claim)
    explanation = {
        "decision": {"text": decision, "claim_id": "C-D1"},
        "main_reasons": reasons,
        "concerns": concerns,
        "limitations": limitations,
        "recommended_action": {"text": action, "claim_id": "C-A1"},
        "counterfactuals": counterfactuals,
        "provenance_summary": provenance,
    }
    row: dict[str, Any] = {
        "schema_version": "2.0",
        "case_id": evidence["case_id"],
        "variant_id": variant_id,
        "modality": evidence["modality"],
        "task_description": evidence["task_description"],
        "audience": "domain_user",
        "prediction": evidence["prediction"],
        "observable_conditions": conditions,
        "interpretable_evidence": shown_items,
        "fidelity_metadata": evidence["fidelity_metadata"] if method != METHODS[0] else _simple_fidelity(evidence["fidelity_metadata"]),
        "candidate_explanation": explanation,
        "claim_evidence_links": claim_links,
        "claim_evidence_coverage": 1.0,
        "presentation": {"language": "ru", "detail": "full" if method == METHODS[2] else "short", "word_count": _word_count(explanation)},
        "record_sha256": "",
    }
    if "observable_asset" in evidence:
        row["observable_asset"] = evidence["observable_asset"]
    row["record_sha256"] = digest(canonical_json(row))
    return row


def _reason_cards(items: list[dict[str, Any]], *, max_items: int) -> list[dict[str, Any]]:
    cards = []
    for index, item in enumerate(items[:max_items], 1):
        verb = "поддерживает" if item["direction"] == "supports" else "противоречит" if item["direction"] == "opposes" else "не меняет"
        cards.append({
            "claim_id": f"C-R{index}",
            "title": str(item["display_name"]),
            "text": f"{item['display_name']} {verb} выбранную категорию; относительная величина вклада {float(item['magnitude_normalized']):.2f}, ранг {item['rank']}.",
            "direction": item["direction"],
            "magnitude_normalized": item["magnitude_normalized"],
            "rank": item["rank"],
            "stability": item["stability"],
            "evidence_ids": [item["evidence_id"]],
        })
    return cards


def _observable_concerns(confidence: float, agreement: float, stability: float, unavailable: list[str], *, full: bool) -> list[dict[str, Any]]:
    rows = []
    if confidence < 0.70:
        rows.append({"claim_id": "C-C1", "text": "Уверенность прогноза ниже 0,70, поэтому решение требует дополнительной проверки.", "evidence_ids": ["O-CONFIDENCE"]})
    if agreement < 0.70:
        rows.append({"claim_id": "C-C2", "text": "Наблюдаемые источники не полностью согласуются по важности причин.", "evidence_ids": ["O-AGREEMENT"]})
    if stability < 0.70 and full:
        rows.append({"claim_id": "C-C3", "text": "Порядок причин недостаточно устойчив при доступных проверках.", "evidence_ids": ["O-STABILITY"]})
    if unavailable:
        rows.append({"claim_id": "C-C4", "text": "Часть проверочных каналов не предоставлена в доступном следе.", "evidence_ids": ["O-CHANNELS"]})
    return rows[:3]


def _limitations(unavailable: list[str], *, extended: bool) -> list[dict[str, Any]]:
    rows = [{"claim_id": "C-L1", "text": "Показанные связи характеризуют поведение модели и не доказывают причинность.", "evidence_ids": ["E1"]}]
    if unavailable:
        rows.append({"claim_id": "C-L2", "text": "Недоступные каналы ограничивают полноту проверки объяснения.", "evidence_ids": ["O-CHANNELS"]})
    if extended:
        rows.append({"claim_id": "C-L3", "text": "Предметная корректность действия не подтверждена независимым специалистом.", "evidence_ids": ["O-APPLICABILITY"]})
    return rows


def _claim_links(
    evidence: dict[str, Any],
    reasons: list[dict[str, Any]],
    concerns: list[dict[str, Any]],
    limitations: list[dict[str, Any]],
    action: str,
    action_ref: str,
) -> list[dict[str, Any]]:
    del evidence
    links = []
    for row in [*reasons, *concerns, *limitations]:
        links.append({"claim_id": row["claim_id"], "text": row["text"], "evidence_ids": row["evidence_ids"]})
    links.append({"claim_id": "C-A1", "text": action, "evidence_ids": [action_ref]})
    return links


def _prospective_action(confidence: float, agreement: float, stability: float, unavailable: list[str]) -> tuple[str, str]:
    if unavailable and (confidence < 0.70 or agreement < 0.60):
        return "Не применять результат автоматически; выполнить полную проверку доступных данных и происхождения объяснения.", "O-CHANNELS"
    if confidence < 0.70 or agreement < 0.65 or stability < 0.65:
        return "Перед применением выполнить полную проверку специалистом.", "O-AGREEMENT"
    if unavailable or agreement < 0.80:
        return "Выполнить краткую проверку причин и недоступных каналов.", "O-CHANNELS" if unavailable else "O-AGREEMENT"
    return "Использовать результат только в пределах исследовательского контракта и стандартного предметного контроля.", "O-CONFIDENCE"


def _observable_unavailable(source: dict[str, Any], method: str) -> list[str]:
    if method == METHODS[0]:
        return ["сопоставление нескольких источников", "история формирования объяснения", "проверка маршрута"]
    unavailable = []
    condition = source.get("controlled_condition")
    if condition == "missing_provenance":
        unavailable.append("часть источников происхождения")
    if method == METHODS[1]:
        unavailable.extend(["полная история формирования объяснения", "расширенный аудиторский след"])
    else:
        unavailable.extend(["история обучения модели", "измеренная потеря редукции"])
    return unavailable


def _available_channels(method: str) -> list[str]:
    if method == METHODS[0]:
        return ["прогноз", "локальные причины одного источника"]
    if method == METHODS[1]:
        return ["прогноз", "ранжированные причины", "согласие источников", "оценка устойчивости", "краткий след"]
    return ["прогноз", "многоканальные причины", "согласие", "устойчивость", "чувствительность", "расширенный след"]


def _shift_summary(evidence: dict[str, Any], method: str) -> str:
    if method == METHODS[0]:
        return "Отдельная проверка отличия от reference-профиля не предоставлена."
    percentiles = [
        float(item["reference_percentile"])
        for item in evidence.get("interpretable_evidence", [])
        if isinstance(item, dict) and item.get("reference_percentile") is not None
    ]
    if any(value <= 0.05 or value >= 0.95 for value in percentiles):
        return "Один или несколько показанных признаков находятся вне центральных 90% reference-выборки; предметная причина отклонения не установлена."
    if percentiles:
        return "Показанные признаки находятся внутри центральных 90% доступной reference-выборки."
    return "Числовой reference-профиль для этой модальности не предоставлен; вывод о сдвиге не формируется."


def _quality_sentence(label: str, value: float) -> str:
    band = "высокое" if value >= 0.80 else "умеренное" if value >= 0.60 else "низкое"
    return f"{label}: {band} ({value:.2f}) по доступным вычислительным проверкам."


def _simple_fidelity(value: object) -> object:
    if not isinstance(value, dict):
        return None
    return {key: value[key] for key in ("value", "scale_min", "scale_max", "higher_is_better", "interpretation_band", "metric_name")}


def _hidden_record(source: dict[str, Any], method: str, variant_id: str) -> dict[str, Any]:
    expected_action = _hidden_expected_action(source)
    return {
        "case_id": source["case_id"],
        "variant_id": variant_id,
        "method_identity": method,
        "true_label": source["prediction"]["true_label"],
        "is_correct": source["prediction"]["is_correct"],
        "original_stratum": source["stratum"],
        "expected_action": expected_action,
        "hidden_rupture_type": source.get("controlled_condition") if source.get("controlled_condition") == "controlled_structural_rupture" else None,
        "known_contradictions": [],
        "known_unsupported_claims": [],
        "original_model_decision": source["prediction"]["display_label"],
        "answer_key_annotations": ["controlled condition is not external-domain ground truth"] if source.get("controlled_condition") else [],
    }


def _hidden_expected_action(source: dict[str, Any]) -> str:
    condition = source.get("controlled_condition")
    if condition == "controlled_structural_rupture":
        return "block"
    if condition == "missing_provenance" or source.get("cross_model_conflict"):
        return "full_review"
    if float(source["prediction"]["score"]) < 0.70:
        return "short_review"
    return "accept"


def _build_batches(root: Path, rows: list[dict[str, Any]], case_limit: int) -> list[dict[str, Any]]:
    output = root / "study/ai_pre_review_final/blind_batches"
    rubric = (root / "study/ai_pre_review/rubric_v1.yaml").read_text(encoding="utf-8")
    batches = []
    hidden_source = {row["case_id"]: row for row in read_jsonl(root / "study/ai_pre_review/source_case_evidence.jsonl")}
    for split in ("formative", "confirmatory"):
        case_ids = sorted(case_id for case_id, source in hidden_source.items() if source["split"] == split)
        for number, offset in enumerate(range(0, len(case_ids), case_limit), 1):
            selected = set(case_ids[offset : offset + case_limit])
            batch_rows = [row for row in rows if row["case_id"] in selected]
            batch_id = f"{split}_batch_{number:03d}"
            jsonl = output / split / f"batch_{number:03d}.jsonl"
            write_jsonl(jsonl, batch_rows)
            batch_hash = sha256_file(jsonl)
            markdown = output / split / f"batch_{number:03d}.md"
            markdown.parent.mkdir(parents=True, exist_ok=True)
            markdown.write_text(_batch_markdown(batch_id, batch_hash, rubric, batch_rows), encoding="utf-8")
            batches.append({
                "batch_id": batch_id,
                "split": split,
                "case_count": len(selected),
                "variant_count": len(batch_rows),
                "jsonl": jsonl.relative_to(root).as_posix(),
                "jsonl_sha256": batch_hash,
                "markdown": markdown.relative_to(root).as_posix(),
                "markdown_sha256": sha256_file(markdown),
            })
    return batches


def _batch_markdown(batch_id: str, batch_hash: str, rubric: str, rows: list[dict[str, Any]]) -> str:
    lines = [
        f"# Blind explanation review packet: {batch_id}", "", f"Input SHA256: `{batch_hash}`", "",
        "Оценивайте только видимые наблюдения. Фактический исход, ожидаемое действие и способ построения варианта скрыты.",
        "Предлагаемое действие оценивается prospective: по сведениям, доступным в карточке, без знания истинного исхода.",
        "Не угадывайте способ построения варианта и не используйте внешние предметные сведения.", "", "## Rubric", "", "```yaml", rubric.rstrip(), "```",
    ]
    for case_id in sorted({str(row["case_id"]) for row in rows}):
        lines.extend(["", f"## {case_id}"])
        for row in [item for item in rows if item["case_id"] == case_id]:
            lines.extend(["", f"### {row['variant_id']}", "", "```json", json.dumps(row, ensure_ascii=False, indent=2, sort_keys=True), "```"])
    lines.extend(["", "Верните только JSONL по фиксированной response schema.", ""])
    return "\n".join(lines)


def _build_master_markdown(path: Path, rows: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    lines = [
        "# Blind Review Master Log v2", "", "Outcome, answer key, method identity and expected action are not included.",
        f"Cases: {manifest['reviewer_cases']}; variants: {manifest['reviewer_variants']}.",
        f"Reviewer JSONL SHA256: `{manifest['reviewer_cases_sha256']}`.", "",
    ]
    for offset in range(0, len(rows), 60):
        lines.extend([f"## Block {offset // 60 + 1}", ""])
        for row in rows[offset : offset + 60]:
            lines.extend([f"### {row['case_id']} / {row['variant_id']}", "", "```json", json.dumps(row, ensure_ascii=False, indent=2, sort_keys=True), "```", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _validate_counts(rows: list[dict[str, Any]]) -> None:
    if len(rows) != 1080:
        raise FinalStudyError("reviewer log must contain 1080 variants")
    keys = Counter((row["case_id"], row["variant_id"]) for row in rows)
    if len(keys) != 1080 or any(value != 1 for value in keys.values()):
        raise FinalStudyError("reviewer variants are missing or duplicated")


def _blind_order(secret: bytes, case_id: str) -> tuple[str, ...]:
    return tuple(sorted(METHODS, key=lambda method: hmac.new(secret, f"{case_id}:{method}".encode(), hashlib.sha256).digest()))


def _load_or_create_secret(path: Path) -> bytes:
    if path.exists():
        value = path.read_bytes().strip()
        if len(value) < 32:
            raise FinalStudyError("blinding secret is too short")
        return value
    path.parent.mkdir(parents=True, exist_ok=True)
    value = secrets.token_hex(32).encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(value + b"\n")
    return value


def _encrypt(root: Path, secret_path: Path, plaintext: Path, output: Path) -> None:
    secret = secret_path.read_bytes().strip()
    salt = hmac.new(secret, b"FXAI-FINAL-AI-HUMAN-CLOSURE", hashlib.sha256).hexdigest()[:16]
    subprocess.run(
        ["openssl", "enc", "-aes-256-cbc", "-pbkdf2", "-S", salt, "-in", str(plaintext), "-out", str(output), "-pass", f"file:{secret_path}"],
        cwd=root,
        check=True,
        capture_output=True,
    )


def _word_count(value: object) -> int:
    return len(canonical_json(value).replace('"', " ").split())


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()
