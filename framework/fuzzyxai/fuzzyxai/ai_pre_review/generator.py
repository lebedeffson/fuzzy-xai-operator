"""Build deterministic blind explanation logs and review packets."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from .contracts import StudyBoundaryError, canonical_json, contains_method_identity, read_jsonl, sha256_bytes, sha256_file, write_jsonl


METHODS = ("strong_local_baseline", "selective_risk_observer", "full_operator_route")


def build_study_inputs(root: Path, *, secret_path: Path | None = None) -> dict[str, object]:
    config = json.loads((root / "configs/ai_pre_review/config.json").read_text(encoding="utf-8"))
    source_path = root / "study/ai_pre_review/source_case_evidence.jsonl"
    source = read_jsonl(source_path)
    _validate_source(source, config)
    secret_file = secret_path or Path.home() / ".local/share/fuzzyxai/ai_pre_review_blinding_secret"
    secret = _load_or_create_secret(secret_file)
    commit = _git(root, "rev-parse", "HEAD")
    records: list[dict[str, Any]] = []
    identity_map: list[dict[str, str]] = []
    for case in source:
        order = _blind_order(secret, str(case["case_id"]))
        for position, method in enumerate(order, 1):
            variant_id = f"X{position}"
            token = hmac.new(secret, f"{case['case_id']}:{variant_id}:{method}".encode(), hashlib.sha256).hexdigest()
            row = _build_variant(case, method, variant_id, position, token, commit)
            if contains_method_identity(_blind_payload(row)):
                raise StudyBoundaryError(f"method identity leaked in {case['case_id']} {variant_id}")
            records.append(row)
            identity_map.append({"case_id": str(case["case_id"]), "variant_id": variant_id, "method": method, "token": token})
    _validate_master(records, config)
    study = root / "study/ai_pre_review"
    master = study / "master_explanation_log.jsonl"
    write_jsonl(master, records)
    _encrypt_identity_map(root, secret_file, identity_map, study / "method_identity_key.encrypted")
    batch_rows = _build_batches(root, records, config)
    manifest = {
        "schema_version": "1.0",
        "study_id": config["study_id"],
        "stage": "formative",
        "frozen_q1_commit": config["frozen_q1_commit"],
        "formative_observer_commit": config["formative_observer_commit"],
        "ai_review_commit": commit,
        "generated_at": _git(root, "show", "-s", "--format=%cI", "HEAD"),
        "source_case_sha256": sha256_file(source_path),
        "master_log_sha256": sha256_file(master),
        "cases": len(source),
        "variants": len(records),
        "method_identity": "encrypted_out_of_band_secret_not_in_repository",
        "frozen_evidence_limitations": [
            "Fashion-MNIST frozen benchmark marks every evaluated image class as rare; common_class is unavailable for image modality.",
            "Controlled route conditions are explicitly labeled and are not external-domain observations.",
        ],
        "batches": batch_rows,
    }
    manifest_path = study / "batch_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _build_master_markdown(study / "AI_REVIEW_MASTER_LOG.md", records, manifest)
    return manifest


def _build_variant(
    case: dict[str, Any], method: str, variant_id: str, position: int, token: str, commit: str
) -> dict[str, Any]:
    prediction = dict(case["prediction"])
    score = float(prediction["score"])
    condition = case.get("controlled_condition")
    conflict = bool(case.get("cross_model_conflict"))
    unstable = float(case.get("explainer_disagreement", 0.0)) >= 0.10 or condition == "controlled_attribution_instability"
    missing = condition == "missing_provenance"
    rupture = condition == "controlled_structural_rupture"
    action = _action(method, score, conflict, unstable, missing, rupture, bool(case.get("rare_class")))
    reason = f"Прогноз: {prediction['display_label']}; уверенность {score:.2f}."
    concerns: list[str] = []
    limitations: list[str] = []
    if score < 0.70:
        concerns.append("Уверенность ниже заранее заданного порога 0,70.")
    if conflict:
        concerns.append("Проверенные модели дают разные классы для этого объекта.")
    if unstable:
        concerns.append("Локальные источники объяснения заметно расходятся.")
    if missing:
        limitations.append("Часть происхождения объяснения недоступна.")
    if rupture:
        limitations.append("Контролируемая проверка обнаружила нарушение обязательного маршрута.")
    if case.get("rare_class"):
        limitations.append("Истинный класс относится к редким в frozen benchmark.")
    if method == "strong_local_baseline":
        text = reason + " Локальный вклад показан как вспомогательное свидетельство. " + (
            limitations[0] if limitations else "Результат не является предметным заключением."
        )
        shown_concerns = concerns[:1]
        shown_limitations = limitations[:1] or ["Локальное объяснение не подтверждает причинность."]
        channels = ["prediction", "local_attribution"]
        provenance = ["frozen model output", "anonymized local attribution"]
        detail = "short"
    elif method == "selective_risk_observer":
        text = _human_text(reason, concerns, limitations, action)
        shown_concerns = concerns[:2]
        shown_limitations = limitations[:2] or ["Предметная корректность требует независимого специалиста."]
        channels = ["prediction", "local_attribution", "risk_observer", "limitations"]
        provenance = ["frozen model output", "cross-model check", "selective action contract"]
        detail = "short"
    else:
        text = _human_text(reason, concerns, limitations, action) + " Технический след позволяет проверить источники каждого вывода."
        shown_concerns = concerns
        shown_limitations = limitations or ["Предметная корректность и практическое действие не подтверждены экспертами."]
        channels = ["prediction", "local_attribution", "model_disagreement", "route_diagnostics", "provenance"]
        provenance = ["frozen model output", "anonymized explainer evidence", "route condition", "action contract"]
        detail = "full"
    claims = [
        {"claim_id": f"{case['case_id']}:prediction", "statement": reason, "evidence_refs": ["prediction"]},
        {"claim_id": f"{case['case_id']}:action", "statement": f"Рекомендуемое действие: {action}.", "evidence_refs": ["action_contract"]},
    ]
    snapshot = {
        "available_channels": channels,
        "missing_channels": ["complete_provenance"] if missing else [],
        "local_attributions": case.get("explainer_evidence", []),
        "rules": [],
        "similar_cases": [],
        "training_history": [],
        "diagnostics": concerns,
        "structural_ruptures": [condition] if rupture else [],
    }
    row: dict[str, Any] = {
        "schema_version": "1.0",
        "case_id": case["case_id"],
        "object_id_hash": case["object_id_hash"],
        "dataset_id": case["dataset_id"],
        "modality": case["modality"],
        "task": case["task"],
        "model_family": "blinded_model_family",
        "model_version_hash": case["model_version_hash"],
        "split": case["split"],
        "stratum": [*case["stratum"], f"action_{action}"],
        "variant_id": variant_id,
        "variant_position": position,
        "method_identity_encrypted": token,
        "prediction": {key: prediction[key] for key in ("display_label", "score", "is_correct")},
        "action": action,
        "explanation_text": text,
        "reasons": [reason],
        "concerns": shown_concerns,
        "limitations": shown_limitations,
        "counterfactuals": [],
        "provenance_summary": provenance,
        "evidence_snapshot": snapshot,
        "grounded_checks": {
            "claims": claims,
            "claim_evidence_links": [{"claim_id": claim["claim_id"], "evidence_refs": claim["evidence_refs"]} for claim in claims],
            "known_contradictions": [],
            "known_unsupported_claims": [],
        },
        "presentation": {
            "language": "ru",
            "audience": "domain_user",
            "detail": detail,
            "word_count": len(text.split()),
        },
        "provenance": {
            "source_commit": case["source_commit"],
            "model_sha256": case["model_version_hash"],
            "explanation_sha256": sha256_bytes(text.encode()),
            "record_sha256": "",
        },
    }
    row["provenance"]["record_sha256"] = sha256_bytes(canonical_json(row).encode())
    return row


def _human_text(reason: str, concerns: list[str], limitations: list[str], action: str) -> str:
    blocks = [reason]
    if concerns:
        blocks.append("Сомнение: " + " ".join(concerns[:2]))
    if limitations:
        blocks.append("Ограничение: " + " ".join(limitations[:2]))
    labels = {
        "accept": "Результат можно использовать только в пределах указанного исследовательского контракта.",
        "short_review": "Рекомендуется краткая проверка специалистом.",
        "full_review": "Требуется полная проверка специалистом до применения.",
        "block": "Автоматическое применение заблокировано до восстановления обязательных свидетельств.",
    }
    blocks.append(labels[action])
    return " ".join(blocks)


def _action(method: str, score: float, conflict: bool, unstable: bool, missing: bool, rupture: bool, rare: bool) -> str:
    if method == "strong_local_baseline":
        return "accept" if score >= 0.70 else "short_review"
    if rupture or (missing and conflict):
        return "block"
    if conflict or unstable:
        return "full_review"
    if score < 0.70 or missing or rare:
        return "short_review"
    return "accept"


def _blind_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key not in {"method_identity_encrypted"}}


def _build_batches(root: Path, records: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, object]]:
    output = root / "study/ai_pre_review/chatgpt_batches"
    batches: list[dict[str, object]] = []
    case_limit = int(config["batch_case_limit"])
    rubric = (root / "study/ai_pre_review/rubric_v1.yaml").read_text(encoding="utf-8")
    for split in ("formative", "confirmatory"):
        case_ids = sorted({str(row["case_id"]) for row in records if row["split"] == split})
        for number, offset in enumerate(range(0, len(case_ids), case_limit), 1):
            selected = set(case_ids[offset : offset + case_limit])
            rows = [_blind_payload(row) for row in records if row["case_id"] in selected]
            batch_id = f"{split}_batch_{number:03d}"
            jsonl = output / split / f"batch_{number:03d}.jsonl"
            write_jsonl(jsonl, rows)
            input_hash = sha256_file(jsonl)
            markdown = output / split / f"batch_{number:03d}.md"
            markdown.parent.mkdir(parents=True, exist_ok=True)
            markdown.write_text(_batch_markdown(batch_id, input_hash, rubric, rows), encoding="utf-8")
            batches.append(
                {
                    "batch_id": batch_id,
                    "split": split,
                    "case_count": len(selected),
                    "variant_count": len(rows),
                    "jsonl": jsonl.relative_to(root).as_posix(),
                    "jsonl_sha256": input_hash,
                    "markdown": markdown.relative_to(root).as_posix(),
                    "markdown_sha256": sha256_file(markdown),
                }
            )
    return batches


def _batch_markdown(batch_id: str, batch_hash: str, rubric: str, rows: list[dict[str, Any]]) -> str:
    lines = [
        f"# Blind AI pre-review packet: {batch_id}",
        "",
        f"Input JSONL SHA256: `{batch_hash}`",
        "",
        "Оцените только предоставленные evidence. Не угадывайте метод, не используйте внешние предметные сведения и верните только JSONL по ai_review_schema.json.",
        "Вычислительная согласованность не является предметной или человеческой валидацией.",
        "",
        "## Rubric",
        "",
        "```yaml",
        rubric.rstrip(),
        "```",
    ]
    for case_id in sorted({str(row["case_id"]) for row in rows}):
        lines.extend(["", f"## {case_id}"])
        for row in [item for item in rows if item["case_id"] == case_id]:
            lines.extend(["", f"### {row['variant_id']}", "", "```json", json.dumps(row, ensure_ascii=False, indent=2, sort_keys=True), "```"])
    lines.extend(["", "## Response", "", "Верните ровно одну JSONL-строку на каждый вариант. Не добавляйте Markdown.", ""])
    return "\n".join(lines)


def _build_master_markdown(path: Path, records: list[dict[str, Any]], manifest: dict[str, object]) -> None:
    lines = [
        "# AI Review Master Log",
        "",
        "Статус: blind AI pre-review input; human confirmation not run.",
        f"Cases: {manifest['cases']}; variants: {manifest['variants']}.",
        f"Master SHA256: `{manifest['master_log_sha256']}`.",
        "",
        "Методы скрыты. Каждое число происходит из frozen Q1 evidence или явно помеченного controlled condition.",
    ]
    for offset in range(0, len(records), 60):
        chunk = records[offset : offset + 60]
        lines.extend(["", f"## Cases {offset // 3 + 1}-{offset // 3 + len(chunk) // 3}"])
        for row in chunk:
            blind = _blind_payload(row)
            lines.extend(["", f"### {row['case_id']} / {row['variant_id']}", "", "```json", json.dumps(blind, ensure_ascii=False, indent=2, sort_keys=True), "```"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _validate_source(rows: list[dict[str, Any]], config: dict[str, Any]) -> None:
    expected = 4 * (int(config["formative_cases_per_modality"]) + int(config["confirmatory_cases_per_modality"]))
    if len(rows) != expected or len({row.get("object_id_hash") for row in rows}) != expected:
        raise StudyBoundaryError(f"source snapshot must contain {expected} unique objects")


def _validate_master(rows: list[dict[str, Any]], config: dict[str, Any]) -> None:
    if len(rows) != 1080:
        raise StudyBoundaryError("master log must contain 1080 blind variants")
    counts = Counter((row["case_id"], row["variant_id"]) for row in rows)
    if any(value != 1 for value in counts.values()) or len(counts) != 1080:
        raise StudyBoundaryError("master log contains duplicate or missing variants")
    for row in rows:
        expected = row["provenance"]["record_sha256"]
        copy = json.loads(canonical_json(row))
        copy["provenance"]["record_sha256"] = ""
        if expected != sha256_bytes(canonical_json(copy).encode()):
            raise StudyBoundaryError("record SHA256 mismatch")


def _blind_order(secret: bytes, case_id: str) -> tuple[str, ...]:
    return tuple(sorted(METHODS, key=lambda name: hmac.new(secret, f"{case_id}:{name}".encode(), hashlib.sha256).digest()))


def _load_or_create_secret(path: Path) -> bytes:
    if path.exists():
        secret = path.read_bytes().strip()
        if len(secret) < 32:
            raise StudyBoundaryError("blinding secret is too short")
        return secret
    path.parent.mkdir(parents=True, exist_ok=True)
    secret = secrets.token_hex(32).encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(secret + b"\n")
    return secret


def _encrypt_identity_map(root: Path, secret_path: Path, rows: list[dict[str, str]], output: Path) -> None:
    plaintext = output.with_suffix(".tmp.json")
    plaintext.write_text(json.dumps(rows, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    try:
        secret = secret_path.read_bytes().strip()
        salt = hmac.new(secret, b"FXAI-AI-REVIEW-HUMAN-CONFIRMATION-FINAL", hashlib.sha256).hexdigest()[:16]
        subprocess.run(
            ["openssl", "enc", "-aes-256-cbc", "-pbkdf2", "-S", salt, "-in", str(plaintext), "-out", str(output), "-pass", f"file:{secret_path}"],
            cwd=root,
            check=True,
            capture_output=True,
        )
    finally:
        plaintext.unlink(missing_ok=True)


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()
