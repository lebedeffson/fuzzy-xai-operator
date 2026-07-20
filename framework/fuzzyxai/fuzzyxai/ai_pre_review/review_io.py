"""Import validation and agreement statistics for AI and human reviews."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median, pvariance
from typing import Any, Iterable

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

from .contracts import (
    AI_RUN_IDS,
    COMMIT_RE,
    CRITICAL_FLAGS,
    METHOD_NAMES,
    SCORE_KEYS,
    SHA_RE,
    StudyBoundaryError,
    canonical_json,
    read_jsonl,
    sha256_bytes,
    validate_flag_rows,
    validate_score_map,
    write_jsonl,
)


def validate_ai_review_directory(
    root: Path, review_dir: Path, *, split: str, run_id: str | None = None
) -> list[dict[str, Any]]:
    manifest = _manifest(root)
    expected_batches = {row["batch_id"]: row for row in manifest["batches"] if row["split"] == split}
    if not expected_batches:
        raise StudyBoundaryError(f"no batches registered for split {split}")
    files = [review_dir] if review_dir.is_file() else sorted(review_dir.rglob("*.jsonl"))
    if not files:
        raise StudyBoundaryError("no raw AI review JSONL files found")
    rows = [row for path in files for row in read_jsonl(path)]
    expected_pairs: set[tuple[str, str, str]] = set()
    for batch_id, batch in expected_batches.items():
        input_rows = read_jsonl(root / str(batch["jsonl"]))
        expected_pairs.update((batch_id, str(row["case_id"]), str(row["variant_id"])) for row in input_rows)
    observed_pairs: set[tuple[str, str, str]] = set()
    for row in rows:
        _validate_ai_row(row, expected_batches, manifest, run_id)
        key = (str(row["batch_id"]), str(row["case_id"]), str(row["variant_id"]))
        if key in observed_pairs:
            raise StudyBoundaryError(f"duplicate AI review: {key}")
        observed_pairs.add(key)
    missing = expected_pairs - observed_pairs
    extra = observed_pairs - expected_pairs
    if missing or extra:
        raise StudyBoundaryError(f"AI review coverage mismatch: missing={len(missing)} extra={len(extra)}")
    return rows


def _validate_ai_row(
    row: dict[str, Any], batches: dict[str, dict[str, Any]], manifest: dict[str, Any], run_id: str | None
) -> None:
    batch_id = str(row.get("batch_id", ""))
    batch = batches.get(batch_id)
    if batch is None:
        raise StudyBoundaryError(f"unknown batch_id: {batch_id}")
    if row.get("review_schema_version") != "1.0" or row.get("reviewer_type") != "ai_pre_reviewer":
        raise StudyBoundaryError("AI review schema or reviewer_type is invalid")
    if row.get("input_batch_sha256") != batch["jsonl_sha256"]:
        raise StudyBoundaryError("AI review input batch hash was changed")
    if run_id is not None and row.get("ai_run_id") != run_id:
        raise StudyBoundaryError("AI review run id mismatch")
    if row.get("ai_run_id") not in AI_RUN_IDS:
        raise StudyBoundaryError("unknown AI run id")
    if not COMMIT_RE.fullmatch(str(row.get("ai_review_commit", ""))):
        raise StudyBoundaryError("AI review commit must be a complete Git hash")
    if row.get("ai_review_commit") != manifest["ai_review_commit"]:
        raise StudyBoundaryError("AI review belongs to another commit")
    validate_score_map(row.get("scores"))
    validate_flag_rows(row.get("critical_flags"))
    if not 1 <= int(row.get("preferred_variant_rank", 0)) <= 3:
        raise StudyBoundaryError("preferred variant rank is invalid")
    confidence = row.get("confidence_in_review")
    if not isinstance(confidence, (int, float)) or not 0 <= float(confidence) <= 1:
        raise StudyBoundaryError("confidence_in_review must be from 0 to 1")
    if len(str(row.get("summary", ""))) > 600:
        raise StudyBoundaryError("AI review summary exceeds 600 characters")
    if len(row.get("required_changes", [])) > 8 or len(row.get("optional_changes", [])) > 5:
        raise StudyBoundaryError("AI review change list exceeds rubric limits")
    text = canonical_json(row).lower()
    if any(name in text for name in METHOD_NAMES):
        raise StudyBoundaryError("method identity leaked in AI response")


def aggregate_ai_reviews(root: Path, run_dirs: dict[str, Path]) -> dict[str, Any]:
    runs = {run_id: validate_ai_review_directory(root, path, split="confirmatory", run_id=run_id) for run_id, path in run_dirs.items()}
    if set(runs) != set(AI_RUN_IDS):
        raise StudyBoundaryError("confirmatory aggregation requires AI_RUN_1, AI_RUN_2 and AI_RUN_3")
    indexed = {run: {(row["case_id"], row["variant_id"]): row for row in rows} for run, rows in runs.items()}
    keys = set.intersection(*(set(rows) for rows in indexed.values()))
    if len(keys) != 360:
        raise StudyBoundaryError("each confirmatory AI run must contain 360 variant reviews")
    criterion_kappa: dict[str, float] = {}
    for criterion in SCORE_KEYS:
        pairwise = []
        for left, right in (("AI_RUN_1", "AI_RUN_2"), ("AI_RUN_1", "AI_RUN_3"), ("AI_RUN_2", "AI_RUN_3")):
            a = [indexed[left][key]["scores"][criterion] for key in sorted(keys)]
            b = [indexed[right][key]["scores"][criterion] for key in sorted(keys)]
            pairwise.append(float(cohen_kappa_score(a, b, weights="quadratic")))
        criterion_kappa[criterion] = _finite_mean(pairwise)
    totals = np.asarray(
        [[sum(indexed[run][key]["scores"].values()) for run in AI_RUN_IDS] for key in sorted(keys)], dtype=float
    )
    preferred = _preferred_by_case(indexed)
    flag_sets = {
        run: {key: {flag["flag"] for flag in indexed[run][key]["critical_flags"]} for key in keys} for run in AI_RUN_IDS
    }
    unstable = []
    for key, row in zip(sorted(keys), totals, strict=True):
        if pvariance(row.tolist()) > 16 or any(criterion_kappa[name] < 0.60 for name in SCORE_KEYS):
            unstable.append({"case_id": key[0], "variant_id": key[1], "total_score_variance": pvariance(row.tolist())})
    result = {
        "schema_version": "1.0",
        "status": "ai_pre_reviewed" if not unstable else "ai_pre_reviewed_with_instability",
        "run_count": 3,
        "review_count_per_run": len(keys),
        "weighted_kappa_by_criterion": criterion_kappa,
        "weighted_kappa_mean": _finite_mean(criterion_kappa.values()),
        "icc_total_score": _icc_two_way_absolute(totals),
        "preferred_variant_agreement": preferred,
        "critical_flag_agreement": _flag_agreement(flag_sets, keys),
        "score_variance_mean": float(np.mean(np.var(totals, axis=1))),
        "unstable_cases": unstable,
        "human_confirmation": "not_run",
    }
    return result


def validate_human_review_directory(root: Path, review_dir: Path, *, min_experts: int = 3) -> list[dict[str, Any]]:
    commitment_path = root / "study/ai_pre_review/human_confirmation/ai_scores_commitment.json"
    lock_path = root / "study/ai_pre_review/confirmatory_protocol_lock.json"
    if not commitment_path.is_file() or not lock_path.is_file():
        raise StudyBoundaryError("human import requires protocol lock and AI score commitment")
    files = [review_dir] if review_dir.is_file() else sorted(review_dir.rglob("*.jsonl"))
    if not files:
        raise StudyBoundaryError("human results are external inputs and were not found")
    rows = [row for path in files for row in read_jsonl(path)]
    experts = {str(row.get("reviewer_hash", "")) for row in rows}
    if len(experts) < min_experts:
        raise StudyBoundaryError(f"at least {min_experts} independent human reviewers are required")
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        if row.get("reviewer_type", "human_expert") != "human_expert":
            raise StudyBoundaryError("AI review cannot be imported as a human response")
        validate_score_map(row.get("scores"))
        validate_flag_rows(row.get("critical_flags"))
        for field in ("reviewer_hash", "signed_record_sha256", "input_batch_sha256"):
            if not SHA_RE.fullmatch(str(row.get(field, ""))):
                raise StudyBoundaryError(f"invalid human response field: {field}")
        for field in ("domain_correctness", "action_acceptability", "domain_terminology", "real_world_risk"):
            if type(row.get(field)) is not int or not 0 <= row[field] <= 4:
                raise StudyBoundaryError(f"invalid human domain score: {field}")
        key = (str(row["reviewer_hash"]), str(row["case_id"]), str(row["variant_id"]))
        if key in seen:
            raise StudyBoundaryError(f"duplicate human response: {key}")
        seen.add(key)
        unsigned = {key: value for key, value in row.items() if key != "signed_record_sha256"}
        if row["signed_record_sha256"] != sha256_bytes(canonical_json(unsigned).encode()):
            raise StudyBoundaryError("human response record hash mismatch")
    per_expert = Counter(str(row["reviewer_hash"]) for row in rows)
    if any(count != 360 for count in per_expert.values()):
        raise StudyBoundaryError("each expert must review all 120 cases and 360 variants")
    return rows


def compare_ai_human(ai_rows: list[dict[str, Any]], human_rows: list[dict[str, Any]], thresholds: dict[str, float]) -> dict[str, Any]:
    ai = _consensus(ai_rows, reviewer_field="ai_run_id")
    human = _consensus(human_rows, reviewer_field="reviewer_hash")
    keys = sorted(set(ai) & set(human))
    if len(keys) != 360:
        raise StudyBoundaryError("AI-human comparison requires all 360 confirmatory variants")
    ai_total = [ai[key]["total"] for key in keys]
    human_total = [human[key]["total"] for key in keys]
    rho = float(spearmanr(ai_total, human_total).statistic)
    kappas = {
        criterion: float(cohen_kappa_score([ai[key][criterion] for key in keys], [human[key][criterion] for key in keys], weights="quadratic"))
        for criterion in SCORE_KEYS
    }
    critical = _critical_metrics(ai, human, keys)
    preferred = _consensus_preferred_agreement(ai_rows, human_rows)
    passed = (
        _finite_mean(kappas.values()) >= thresholds["weighted_kappa"]
        and rho >= thresholds["spearman_overall_usability"]
        and critical["recall"] >= thresholds["critical_defect_recall"]
        and critical["precision"] >= thresholds["critical_defect_precision"]
        and preferred >= thresholds["preferred_variant_agreement"]
    )
    return {
        "schema_version": "1.0",
        "status": "human_confirmed" if passed else "formative_assistance_only",
        "n_variants": len(keys),
        "spearman_total_score": rho,
        "weighted_kappa_by_criterion": kappas,
        "weighted_kappa_mean": _finite_mean(kappas.values()),
        "mean_absolute_error_total": float(np.mean(np.abs(np.asarray(ai_total) - np.asarray(human_total)))),
        "critical_defects": critical,
        "preferred_variant_agreement": preferred,
        "thresholds": thresholds,
        "threshold_gate_passed": passed,
        "bias_audits_required": ["modality", "stratum", "explanation_length", "blind_method_identity"],
    }


def write_review_import(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    write_jsonl(path, rows)


def _manifest(root: Path) -> dict[str, Any]:
    path = root / "study/ai_pre_review/batch_manifest.json"
    if not path.is_file():
        raise StudyBoundaryError("build blind batches before importing reviews")
    return json.loads(path.read_text(encoding="utf-8"))


def _preferred_by_case(indexed: dict[str, dict[tuple[str, str], dict[str, Any]]]) -> dict[str, object]:
    cases = sorted({key[0] for key in indexed["AI_RUN_1"]})
    unanimous = 0
    pairwise_matches = 0
    pairwise_total = 0
    for case in cases:
        choices = []
        for run in AI_RUN_IDS:
            rows = [row for (case_id, _), row in indexed[run].items() if case_id == case]
            choices.append(min(rows, key=lambda row: int(row["preferred_variant_rank"]))["variant_id"])
        unanimous += len(set(choices)) == 1
        for left, right in ((0, 1), (0, 2), (1, 2)):
            pairwise_matches += choices[left] == choices[right]
            pairwise_total += 1
    return {"unanimous_rate": unanimous / len(cases), "pairwise_rate": pairwise_matches / pairwise_total}


def _flag_agreement(
    flag_sets: dict[str, dict[tuple[str, str], set[str]]], keys: set[tuple[str, str]]
) -> dict[str, float]:
    output: dict[str, float] = {}
    for flag in CRITICAL_FLAGS:
        matches = total = 0
        for key in keys:
            values = [flag in flag_sets[run][key] for run in AI_RUN_IDS]
            matches += int(values[0] == values[1]) + int(values[0] == values[2]) + int(values[1] == values[2])
            total += 3
        output[flag] = matches / total
    return output


def _icc_two_way_absolute(values: np.ndarray) -> float:
    n, k = values.shape
    if n < 2 or k < 2:
        return 0.0
    grand = float(np.mean(values))
    row_means = np.mean(values, axis=1)
    col_means = np.mean(values, axis=0)
    ms_rows = k * float(np.sum((row_means - grand) ** 2)) / (n - 1)
    ms_cols = n * float(np.sum((col_means - grand) ** 2)) / (k - 1)
    residual = values - row_means[:, None] - col_means[None, :] + grand
    ms_error = float(np.sum(residual**2)) / ((n - 1) * (k - 1))
    denominator = ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n
    return (ms_rows - ms_error) / denominator if denominator else 0.0


def _consensus(rows: list[dict[str, Any]], *, reviewer_field: str) -> dict[tuple[str, str], dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["case_id"]), str(row["variant_id"]))].append(row)
    result = {}
    for key, values in grouped.items():
        scores = {criterion: int(round(median([row["scores"][criterion] for row in values]))) for criterion in SCORE_KEYS}
        flags = Counter(flag["flag"] for row in values for flag in row["critical_flags"])
        result[key] = {**scores, "total": sum(scores.values()), "flags": {flag for flag, count in flags.items() if count > len(values) / 2}}
    return result


def _critical_metrics(ai: dict[tuple[str, str], dict[str, Any]], human: dict[tuple[str, str], dict[str, Any]], keys: list[tuple[str, str]]) -> dict[str, float]:
    tp = fp = fn = 0
    reference_flags = set(CRITICAL_FLAGS[:5])
    for key in keys:
        ai_flags = ai[key]["flags"] & reference_flags
        human_flags = human[key]["flags"] & reference_flags
        tp += len(ai_flags & human_flags)
        fp += len(ai_flags - human_flags)
        fn += len(human_flags - ai_flags)
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    return {"true_positive": tp, "false_positive": fp, "false_negative": fn, "precision": precision, "recall": recall, "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}


def _consensus_preferred_agreement(ai_rows: list[dict[str, Any]], human_rows: list[dict[str, Any]]) -> float:
    def choices(rows: list[dict[str, Any]], reviewer: str) -> dict[str, str]:
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[(str(row[reviewer]), str(row["case_id"]))].append(row)
        votes: dict[str, Counter[str]] = defaultdict(Counter)
        for (_, case), values in grouped.items():
            winner = min(values, key=lambda row: int(row["preferred_variant_rank"]))["variant_id"]
            votes[case][winner] += 1
        return {case: counter.most_common(1)[0][0] for case, counter in votes.items()}
    ai = choices(ai_rows, "ai_run_id")
    human = choices(human_rows, "reviewer_hash")
    common = set(ai) & set(human)
    return sum(ai[case] == human[case] for case in common) / len(common) if common else 0.0


def _finite_mean(values: Iterable[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return mean(finite) if finite else 0.0
