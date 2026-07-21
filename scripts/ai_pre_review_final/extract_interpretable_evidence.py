#!/usr/bin/env python3
"""Recompute interpretable reviewer evidence for the frozen 360-case cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from fuzzyxai.ai_pre_review.contracts import canonical_json, read_jsonl, write_jsonl
from fuzzyxai.q1_final.multiclass import _stratified_cap, load_native_dataset

ROOT = Path(__file__).resolve().parents[2]
OLD_SOURCE = ROOT / "study/ai_pre_review/source_case_evidence.jsonl"
OUTPUT_DIR = ROOT / "study/ai_pre_review_final/evidence"
CACHE_ROOT = ROOT / ".cache/q1-final"
MODALITIES = ("tabular", "image", "text", "timeseries")

TABULAR_NAMES = (
    "Высота участка", "Ориентация склона", "Уклон поверхности", "Расстояние до воды по горизонтали",
    "Перепад высоты до воды", "Расстояние до дороги", "Освещённость утром", "Освещённость днём",
    "Освещённость вечером", "Расстояние до зоны возгорания",
    *(f"Признак территории {chr(65 + index)}" for index in range(4)),
    *(f"Характеристика почвы {index + 1}" for index in range(40)),
)
FASHION_LABELS = ("футболка или топ", "брюки", "свитер", "платье", "верхняя одежда", "сандалии", "рубашка", "кроссовки", "сумка", "ботинки")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", choices=MODALITIES, required=True)
    args = parser.parse_args()
    rows = [row for row in read_jsonl(OLD_SOURCE) if row["modality"] == args.modality]
    benchmark = json.loads((ROOT / f"release_evidence/q1_final/real_jobs/{args.modality}.json").read_text(encoding="utf-8"))
    dataset = load_native_dataset(args.modality, CACHE_ROOT / args.modality)
    if dataset.raw_sha256 != rows[0]["dataset_sha256"]:
        raise RuntimeError(f"dataset hash mismatch for {args.modality}")
    object_ids = _resolve_ids(rows, benchmark)
    if args.modality == "tabular":
        evidence = _tabular(dataset, rows, object_ids)
    elif args.modality == "image":
        evidence = _image(dataset, rows, object_ids)
    elif args.modality == "text":
        evidence = _text(dataset, rows, object_ids)
    else:
        evidence = _timeseries(dataset, rows, object_ids)
    _validate(evidence, args.modality)
    output = OUTPUT_DIR / f"{args.modality}.jsonl"
    write_jsonl(output, evidence)
    print(f"PASS: interpretable_evidence modality={args.modality} cases={len(evidence)} sha256={hashlib.sha256(output.read_bytes()).hexdigest()}")


def _resolve_ids(rows: list[dict[str, Any]], benchmark: dict[str, Any]) -> list[int]:
    candidates = {int(row["object_id"]) for row in benchmark["object_predictions"]}
    by_hash = {hashlib.sha256(f"{rows[0]['modality']}:{object_id}".encode()).hexdigest(): object_id for object_id in candidates}
    resolved = []
    for row in rows:
        object_id = by_hash.get(str(row["object_id_hash"]))
        if object_id is None:
            raise RuntimeError(f"cannot resolve frozen object {row['case_id']}")
        resolved.append(object_id)
    return resolved


def _base_record(row: dict[str, Any], object_id: int, items: list[dict[str, Any]], label: str, task: str) -> dict[str, Any]:
    return {
        "schema_version": "2.0",
        "case_id": row["case_id"],
        "object_id_hash": row["object_id_hash"],
        "modality": row["modality"],
        "dataset_id": row["dataset_id"],
        "task_description": task,
        "prediction": {"display_label": label, "confidence": row["prediction"]["score"]},
        "interpretable_evidence": items,
        "dataset_sha256": row["dataset_sha256"],
        "evidence_extraction": {
            "source_commit": row["source_commit"],
            "object_reference_hash": hashlib.sha256(f"{row['dataset_id']}:{object_id}".encode()).hexdigest(),
            "result_origin": "recomputed_from_frozen_dataset_object_and_declared_model_seed",
        },
    }


def _tabular(dataset: Any, rows: list[dict[str, Any]], object_ids: list[int]) -> list[dict[str, Any]]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    values = np.asarray(dataset.values, dtype=np.float32)
    labels = np.asarray(dataset.labels, dtype=int)
    indices = np.arange(len(labels))
    train_validation, _ = train_test_split(indices, test_size=0.2, random_state=4201, stratify=labels)
    train, _ = train_test_split(train_validation, test_size=0.25, random_state=4301, stratify=labels[train_validation])
    train = _stratified_cap(train, labels, 100_000, 4201)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=250, random_state=4201)).fit(values[train], labels[train])
    scaler = model.named_steps["standardscaler"]
    estimator = model.named_steps["logisticregression"]
    sample = values[np.asarray(object_ids)]
    transformed = scaler.transform(sample)
    predicted = model.predict(sample).astype(int)
    coefficients = estimator.coef_[predicted]
    contributions = transformed * coefficients
    reference = np.median(values[train], axis=0)
    base_probability = model.predict_proba(sample)
    records = []
    for row_id, (source, object_id) in enumerate(zip(rows, object_ids, strict=True)):
        top = np.argsort(np.abs(contributions[row_id]))[-5:][::-1]
        items = []
        maximum = max(float(np.max(np.abs(contributions[row_id]))), 1e-12)
        for rank, feature in enumerate(top, 1):
            changed = sample[row_id : row_id + 1].copy()
            changed[0, feature] = reference[feature]
            altered = model.predict_proba(changed)[0, predicted[row_id]]
            perturbation = float(base_probability[row_id, predicted[row_id]] - altered)
            direction = "supports" if contributions[row_id, feature] >= 0 else "opposes"
            percentile = float(np.mean(values[train, feature] <= sample[row_id, feature]))
            items.append({
                "evidence_id": f"E{rank}", "type": "feature_attribution", "display_name": TABULAR_NAMES[int(feature)],
                "direction": direction, "magnitude_normalized": abs(float(contributions[row_id, feature])) / maximum,
                "rank": rank, "stability": _agreement(abs(float(contributions[row_id, feature])) / maximum, abs(perturbation)),
                "source_agreement": _agreement(abs(float(contributions[row_id, feature])) / maximum, min(abs(perturbation) * 5.0, 1.0)),
                "observed_value_anonymized": round(float(sample[row_id, feature]), 4),
                "reference_percentile": round(percentile, 6),
                "reference_description": f"Значение находится примерно на {round(percentile * 100)}-м процентиле обучающей выборки.",
                "evidence_refs": [f"tabular:coefficient:{int(feature)}", f"tabular:reference_perturbation:{int(feature)}"],
                "limitations": ["Вклад описывает связь модели, а не причинное влияние на реальный объект."],
            })
        label = f"тип покрытия {chr(65 + int(predicted[row_id]))}"
        records.append(_with_sources(_base_record(source, object_id, items, label, "Определение типа растительного покрытия по характеристикам участка"), source))
    return records


def _image(dataset: Any, rows: list[dict[str, Any]], object_ids: list[int]) -> list[dict[str, Any]]:
    import onnxruntime as ort
    from PIL import Image

    model_path = ROOT / "release_evidence/q1_final/real_jobs/fashion_compact_cnn.onnx"
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    images = np.asarray(dataset.values, dtype=np.float32)[np.asarray(object_ids)]
    tensors = images[:, None, :, :]
    logits = np.asarray(session.run(None, {input_name: tensors})[0], dtype=float)
    probabilities = _softmax(logits)
    predicted = probabilities.argmax(axis=1)
    zero_effect, mean_effect = _onnx_occlusion(session, input_name, tensors, predicted, probabilities)
    asset_dir = OUTPUT_DIR / "assets/image"
    asset_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for row_id, (source, object_id) in enumerate(zip(rows, object_ids, strict=True)):
        thumb = asset_dir / f"{source['case_id']}.png"
        Image.fromarray(np.uint8(np.clip(images[row_id] * 255, 0, 255)), mode="L").resize((112, 112)).save(thumb)
        items = _image_regions(zero_effect[row_id], mean_effect[row_id])
        record = _base_record(source, object_id, items, FASHION_LABELS[int(predicted[row_id])], "Классификация предмета одежды на изображении")
        record["observable_asset"] = {"thumbnail_ref": thumb.relative_to(ROOT / "study/ai_pre_review_final").as_posix(), "image_shape": [28, 28]}
        records.append(_with_sources(record, source))
    return records


def _onnx_occlusion(
    session: Any,
    input_name: str,
    images: np.ndarray,
    predicted: np.ndarray,
    base_probabilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    zero_effect = np.zeros((len(images), 28, 28), dtype=float)
    mean_effect = np.zeros_like(zero_effect)
    reference = float(images.mean())
    for row in range(4):
        for column in range(4):
            y0, y1, x0, x1 = row * 7, (row + 1) * 7, column * 7, (column + 1) * 7
            effects = []
            for replacement in (0.0, reference):
                changed = images.copy()
                changed[:, 0, y0:y1, x0:x1] = replacement
                logits = np.asarray(session.run(None, {input_name: changed})[0], dtype=float)
                probability = _softmax(logits)
                effects.append(base_probabilities[np.arange(len(images)), predicted] - probability[np.arange(len(images)), predicted])
            zero_effect[:, y0:y1, x0:x1] = effects[0][:, None, None]
            mean_effect[:, y0:y1, x0:x1] = effects[1][:, None, None]
    return zero_effect, mean_effect


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _image_regions(primary: np.ndarray, secondary: np.ndarray) -> list[dict[str, Any]]:
    combined = np.abs(primary) + np.abs(secondary)
    cells = []
    for row in range(4):
        for column in range(4):
            y0, y1 = row * 7, (row + 1) * 7
            x0, x1 = column * 7, (column + 1) * 7
            cells.append((float(combined[y0:y1, x0:x1].mean()), row, column))
    maximum = max(max(value for value, _, _ in cells), 1e-12)
    items = []
    for rank, (value, row, column) in enumerate(sorted(cells, reverse=True)[:5], 1):
        y0, y1, x0, x1 = row * 7, (row + 1) * 7, column * 7, (column + 1) * 7
        primary_value = float(primary[y0:y1, x0:x1].mean())
        secondary_value = float(secondary[y0:y1, x0:x1].mean())
        items.append({
            "evidence_id": f"E{rank}", "type": "image_region", "display_name": f"Область изображения {rank}",
            "region_id": f"region_{row}_{column}", "bounding_box": [x0, y0, x1, y1],
            "direction": "supports" if primary_value + secondary_value >= 0 else "opposes",
            "magnitude_normalized": value / maximum, "rank": rank, "stability": _agreement(primary_value, secondary_value),
            "source_agreement": _agreement(primary_value, secondary_value),
            "reference_description": "Квадрат 7x7 пикселей проверен двумя вариантами маскирования: нулевым и средним фоном.",
            "evidence_refs": [f"image:zero_occlusion:{row}:{column}", f"image:mean_occlusion:{row}:{column}"],
            "limitations": ["Область показывает чувствительность модели и не является медицинской или причинной локализацией."],
        })
    return items


def _text(dataset: Any, rows: list[dict[str, Any]], object_ids: list[int]) -> list[dict[str, Any]]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    documents = dataset.values
    labels = np.asarray(dataset.labels)
    indices = np.arange(len(labels))
    train_validation, _ = train_test_split(indices, test_size=0.2, random_state=4201, stratify=labels)
    train, _ = train_test_split(train_validation, test_size=0.25, random_state=4301, stratify=labels[train_validation])
    vectorizer = TfidfVectorizer(max_features=10_000, min_df=2, sublinear_tf=True)
    train_matrix = vectorizer.fit_transform([documents[int(index)] for index in train])
    model = LogisticRegression(max_iter=250, random_state=4201).fit(train_matrix, labels[train])
    sample = vectorizer.transform([documents[index] for index in object_ids])
    predicted = model.predict(sample).astype(int)
    base_probabilities = model.predict_proba(sample)
    values = np.asarray(sample.multiply(model.coef_[predicted]).toarray())
    names = vectorizer.get_feature_names_out()
    target_names = _newsgroup_names()
    records = []
    for row_id, (source, object_id) in enumerate(zip(rows, object_ids, strict=True)):
        top = np.argsort(np.abs(values[row_id]))[-5:][::-1]
        maximum = max(float(np.max(np.abs(values[row_id]))), 1e-12)
        document_lower = str(documents[object_id]).lower()
        items = []
        for rank, feature in enumerate(top, 1):
            token = str(names[feature])
            contribution = float(values[row_id, feature])
            position = document_lower.find(token.lower())
            changed = sample[row_id : row_id + 1].tolil(copy=True)
            changed[0, int(feature)] = 0.0
            perturbation = float(base_probabilities[row_id, predicted[row_id]] - model.predict_proba(changed.tocsr())[0, predicted[row_id]])
            normalized_contribution = abs(contribution) / maximum
            normalized_perturbation = min(abs(perturbation) * 5.0, 1.0)
            items.append({
                "evidence_id": f"E{rank}", "type": "text_phrase", "display_name": f"Фраза «{token}»",
                "phrase": token, "character_position": position if position >= 0 else None,
                "direction": "supports" if contribution >= 0 else "opposes", "magnitude_normalized": normalized_contribution,
                "rank": rank, "stability": _agreement(normalized_contribution, normalized_perturbation),
                "source_agreement": _agreement(normalized_contribution, normalized_perturbation),
                "semantic_group": "лексический признак", "reference_description": "Вклад TF-IDF признака в линейное решение для выбранной темы.",
                "evidence_refs": [f"text:token_contribution:{int(feature)}", f"text:token_deletion:{int(feature)}"],
                "limitations": ["Лексическая связь не подтверждает причинность и может зависеть от корпуса."],
            })
        label = f"тематическая категория «{target_names[int(predicted[row_id])]}»"
        records.append(_with_sources(_base_record(source, object_id, items, label, "Определение тематической категории текстового сообщения"), source))
    return records


def _newsgroup_names() -> list[str]:
    from sklearn.datasets import fetch_20newsgroups
    return [name.replace(".", " / ") for name in fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"), shuffle=False).target_names]


def _timeseries(dataset: Any, rows: list[dict[str, Any]], object_ids: list[int]) -> list[dict[str, Any]]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from fuzzyxai.q1_final.explainers import _timeseries_features

    values = np.asarray(dataset.values, dtype=np.float32)
    labels = np.asarray(dataset.labels)
    indices = np.arange(len(labels))
    train_validation, _ = train_test_split(indices, test_size=0.2, random_state=4201, stratify=labels)
    train, _ = train_test_split(train_validation, test_size=0.25, random_state=4301, stratify=labels[train_validation])
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=250, random_state=4201)).fit(_timeseries_features(values[train]), labels[train])
    sample = values[np.asarray(object_ids)]
    base = model.predict_proba(_timeseries_features(sample))
    predicted = base.argmax(axis=1)
    windows = np.array_split(np.arange(values.shape[1]), 12)
    reference = values[train].mean(axis=0)
    attribution = np.zeros((len(sample), len(windows)), dtype=float)
    zero_attribution = np.zeros_like(attribution)
    for window_id, window in enumerate(windows):
        masked = sample.copy()
        masked[:, window] = reference[window]
        probabilities = model.predict_proba(_timeseries_features(masked))
        attribution[:, window_id] = base[np.arange(len(sample)), predicted] - probabilities[np.arange(len(sample)), predicted]
        zero_masked = sample.copy()
        zero_masked[:, window] = 0.0
        zero_probabilities = model.predict_proba(_timeseries_features(zero_masked))
        zero_attribution[:, window_id] = base[np.arange(len(sample)), predicted] - zero_probabilities[np.arange(len(sample)), predicted]
    records = []
    for row_id, (source, object_id) in enumerate(zip(rows, object_ids, strict=True)):
        top = np.argsort(np.abs(attribution[row_id]))[-5:][::-1]
        maximum = max(float(np.max(np.abs(attribution[row_id]))), 1e-12)
        items = []
        for rank, window_id in enumerate(top, 1):
            window = windows[int(window_id)]
            effect = float(attribution[row_id, window_id])
            zero_effect = float(zero_attribution[row_id, window_id])
            normalized_effect = abs(effect) / maximum
            normalized_zero = min(abs(zero_effect) / maximum, 1.0)
            items.append({
                "evidence_id": f"E{rank}", "type": "time_interval", "display_name": f"Интервал сигнала {int(window[0])}-{int(window[-1])}",
                "interval_start": int(window[0]), "interval_end": int(window[-1]), "signal_channel": "канал энергопотребления",
                "direction": "supports" if effect >= 0 else "opposes", "magnitude_normalized": normalized_effect,
                "rank": rank, "stability": _agreement(normalized_effect, normalized_zero),
                "source_agreement": _agreement(normalized_effect, normalized_zero),
                "observed_mean_anonymized": round(float(sample[row_id, window].mean()), 4),
                "reference_description": "Изменение вероятности после замены интервала средним профилем обучающей выборки.",
                "evidence_refs": [f"timeseries:reference_window_mask:{int(window_id)}", f"timeseries:zero_window_mask:{int(window_id)}"],
                "limitations": ["Маскирование показывает чувствительность модели, а не допустимое физическое воздействие."],
            })
        label = f"профиль энергопотребления {chr(65 + int(predicted[row_id]))}"
        records.append(_with_sources(_base_record(source, object_id, items, label, "Классификация формы временного профиля энергопотребления"), source))
    return records


def _with_sources(record: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    fidelities = [float(item["fidelity"]) for item in source.get("explainer_evidence", [])]
    value = max(fidelities) if fidelities else None
    record["fidelity_metadata"] = None if value is None else {
        "value": value, "scale_min": -1.0, "scale_max": 1.0, "higher_is_better": True,
        "interpretation_band": "high" if value >= 0.7 else "medium" if value >= 0.3 else "low",
        "metric_name": "deletion fidelity or native explainer precision",
        "comparability": "within the same source family and modality only",
    }
    item_agreement = [float(item["source_agreement"]) for item in record["interpretable_evidence"]]
    item_stability = [float(item["stability"]) for item in record["interpretable_evidence"]]
    record["source_summary"] = {
        "source_count": max(2, len(source.get("explainer_evidence", []))),
        "agreement": float(np.median(item_agreement)),
        "stability": float(np.median(item_stability)),
    }
    record["evidence_sha256"] = hashlib.sha256(canonical_json(record).encode()).hexdigest()
    return record


def _agreement(left: float, right: float) -> float:
    scale = max(abs(left), abs(right), 1e-12)
    return max(0.0, 1.0 - abs(left - right) / scale)


def _validate(rows: list[dict[str, Any]], modality: str) -> None:
    if len(rows) != 90:
        raise RuntimeError(f"{modality}: expected 90 records")
    for row in rows:
        items = row["interpretable_evidence"]
        if len(items) < 2:
            raise RuntimeError(f"{row['case_id']}: fewer than two evidence items")
        for item in items:
            required = {"evidence_id", "display_name", "direction", "magnitude_normalized", "rank", "stability", "source_agreement", "evidence_refs", "limitations"}
            if not required.issubset(item):
                raise RuntimeError(f"{row['case_id']}: incomplete evidence item")
            if item["direction"] not in {"supports", "opposes", "neutral"}:
                raise RuntimeError(f"{row['case_id']}: invalid direction")


if __name__ == "__main__":
    main()
