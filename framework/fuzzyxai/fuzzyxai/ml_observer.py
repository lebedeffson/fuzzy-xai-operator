"""Read-only, model-agnostic audit of a tabular ML run.

Observes supplied evidence about data, training, and outputs; never calls
fit(), predict(), repair(), changes policy, or certifies predictive truth.
Classification, regression, and clustering are supported.  Raw images,
text, generative systems, and other modalities need explicit adapters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score

from .fit_provenance import audit_fit_trace
from .output_provenance import audit_output_calls
from .transform_provenance import audit_transform_chain

Task = Literal["classification", "regression", "clustering"]
Stage = Literal["data", "training", "output"]
CheckStatus = Literal["ok", "warning", "problem", "unknown", "not_applicable"]


@dataclass(frozen=True)
class MLRunEvidence:
    task: Task
    model_family: str
    train_features: Any | None = None
    evaluation_features: Any | None = None
    train_targets: Any | None = None
    train_feature_names: tuple[str, ...] | None = None
    evaluation_feature_names: tuple[str, ...] | None = None
    evaluation_targets: Any | None = None
    predictions: Any | None = None
    probabilities: Any | None = None
    probability_classes: tuple[str, ...] | None = None
    train_ids: tuple[str, ...] | None = None
    evaluation_ids: tuple[str, ...] | None = None
    model_fit_ids: tuple[str, ...] | None = None
    preprocessor_fit_ids: tuple[str, ...] | None = None
    preprocessor_applied: bool | None = None
    fit_records: Any | None = None
    output_records: Any | None = None
    transform_records: Any | None = None
    train_loss: tuple[float, ...] | None = None
    validation_loss: tuple[float, ...] | None = None
    model_version: str | None = None
    random_seed: int | None = None


@dataclass(frozen=True)
class MLCheck:
    stage: Stage
    code: str
    status: CheckStatus
    detail: str
    observed: float | int | str | None = None


@dataclass(frozen=True)
class MLObservation:
    task: Task
    model_family: str
    overall_status: str
    checks: tuple[MLCheck, ...]
    metrics: dict[str, float | int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def observe_ml_run(evidence: MLRunEvidence, *, target_hmac_key: bytes | None = None) -> MLObservation:
    """Inspect evidence without mutating arrays, models or stored artifacts.

    'ok' means only that a *specific declared check* passed.  No overall
    health score, guaranteed absence of leakage, or automatic action is
    calculated. Unknown is different from not_applicable and from failure.
    """

    if evidence.task not in ("classification", "regression", "clustering"):
        raise ValueError("supported tasks are classification, regression and clustering")
    if not evidence.model_family:
        raise ValueError("model_family must identify the claimed model family")

    findings: list[MLCheck] = []
    metrics: dict[str, float | int] = {}

    def note(stage: Stage, code: str, status: CheckStatus, detail: str, observed: float | str | None = None) -> None:
        findings.append(MLCheck(stage, code, status, detail, observed))

    def matrix(value: Any | None, name: str) -> np.ndarray | None:
        if value is None:
            note("data", name, "unknown", "Feature matrix was not supplied")
            return None
        try:
            a = np.asarray(value)
        except (TypeError, ValueError) as exc:
            note("data", name, "problem", f"Cannot inspect feature matrix: {type(exc).__name__}")
            return None
        if a.ndim != 2 or 0 in a.shape:
            note("data", name, "problem", "Expected a nonempty two-dimensional feature matrix")
            return None
        if not np.issubdtype(a.dtype, np.number):
            note("data", name, "unknown", "Only numeric tabular features can be checked by this adapter")
            return None
        try:
            numeric = np.asarray(a, dtype=float)
        except (TypeError, ValueError, OverflowError):
            note("data", name, "problem", "Numeric features cannot be represented as finite floats")
            return None
        if not np.isfinite(numeric).all():
            note("data", name, "problem", "NaN or infinity is present in feature values", int(np.size(numeric) - np.isfinite(numeric).sum()))
            return None
        note("data", name, "ok", "Numeric feature matrix has valid shape and finite values", len(numeric))
        metrics[f"{name}_rows"] = len(numeric)
        metrics[f"{name}_features"] = numeric.shape[1]
        return numeric

    x_train = matrix(evidence.train_features, "train_features")
    x_eval = matrix(evidence.evaluation_features, "evaluation_features")
    if x_train is not None and x_eval is not None:
        if x_train.shape[1] != x_eval.shape[1]:
            note("data", "feature_width", "problem", "Training and evaluation feature counts differ")
        else:
            note("data", "feature_width", "ok", "Feature counts match; feature *order* is not independently verified")
            spread = x_train.std(axis=0)
            shift = np.abs(x_train.mean(axis=0) - x_eval.mean(axis=0))
            # Zero variance means changes in a constant training feature are
            # suspicious, but this is only a descriptive drift diagnostic.
            shift = np.where(spread > 1e-12, shift / np.maximum(spread, 1e-12), np.where(shift < 1e-12, 0.0, 1e12))
            max_shift = float(np.max(shift))
            metrics["max_standardized_mean_shift"] = max_shift
            note(
                "data", "mean_shift",
                "warning" if max_shift > 2.0 else "ok",
                "Exploratory mean-shift indicator (threshold 2 train SD); not a distribution-shift test",
                max_shift,
            )
    else:
        note("data", "feature_width", "unknown", "Cannot compare training and evaluation schemas")
        note("data", "mean_shift", "unknown", "Cannot compare train and evaluation means")

    # A duplicated numeric input may be a legitimate repeated observation.
    # It is therefore only an investigation signal, not proof of leakage.
    # Unlike row-ID comparison, this can catch byte-identical examples
    # copied into another split under newly assigned identifiers.
    if x_train is None or x_eval is None or x_train.shape[1] != x_eval.shape[1]:
        note("data", "cross_split_duplicate_rows", "unknown", "Both compatible numeric matrices are required")
    elif x_train.size + x_eval.size > 8_000_000:
        note("data", "cross_split_duplicate_rows", "unknown", "Exact duplicate scan exceeds the configured 8M-value limit")
    else:
        def row_digest(row: np.ndarray) -> bytes:
            raw = np.ascontiguousarray(row, dtype="<f8")
            return hashlib.sha256(raw.tobytes()).digest()

        train_signatures = {row_digest(row) for row in x_train}
        duplicates = sum(row_digest(row) in train_signatures for row in x_eval)
        metrics["cross_split_duplicate_rows"] = int(duplicates)
        note(
            "data",
            "cross_split_duplicate_rows",
            "warning" if duplicates else "ok",
            "Exact numeric rows appear in both partitions (possibly legitimate repeats, not proven leakage)"
            if duplicates else "No identical numeric rows found across partitions",
            duplicates,
        )

    if evidence.train_feature_names is None or evidence.evaluation_feature_names is None:
        note("data", "ordered_feature_schema", "unknown", "Both ordered feature-name lists are needed to verify feature semantics")
    else:
        tr_names = evidence.train_feature_names
        ev_names = evidence.evaluation_feature_names
        valid_train = bool(tr_names) and len(tr_names) == len(set(tr_names)) and all(isinstance(v, str) and v.strip() for v in tr_names)
        valid_eval = bool(ev_names) and len(ev_names) == len(set(ev_names)) and all(isinstance(v, str) and v.strip() for v in ev_names)
        valid_width = (x_train is None or len(tr_names) == x_train.shape[1]) and (x_eval is None or len(ev_names) == x_eval.shape[1])
        if not valid_train or not valid_eval or not valid_width or tr_names != ev_names:
            note("data", "ordered_feature_schema", "problem", "Ordered feature names are invalid or differ across train/evaluation")
        else:
            note("data", "ordered_feature_schema", "ok", "Declared feature names and order agree; transformed schema is not independently authenticated")

    def ids(name: str, raw: tuple[str, ...] | None, n: int | None) -> set[str] | None:
        if raw is None:
            note("training", name, "unknown", "Source row identifiers are not available")
            return None
        if not all(isinstance(x, str) and x for x in raw):
            note("training", name, "problem", "Every source row ID must be a nonempty string")
            return None
        if n is not None and len(raw) != n:
            note("training", name, "problem", "ID count differs from feature-matrix row count")
            return None
        if len(raw) != len(set(raw)):
            note("training", name, "problem", "Row identifiers repeat within a declared partition")
            return None
        return set(raw)

    train_ids = ids("train_ids", evidence.train_ids, len(x_train) if x_train is not None else None)
    eval_ids = ids("evaluation_ids", evidence.evaluation_ids, len(x_eval) if x_eval is not None else None)
    if train_ids is not None and eval_ids is not None:
        overlap = len(train_ids & eval_ids)
        note("training", "split_isolation", "problem" if overlap else "ok",
             "Overlapping source row identifiers" if overlap else "Declared train/evaluation row IDs do not overlap; source identity remains unverified",
             overlap)
    else:
        note("training", "split_isolation", "unknown", "Cannot establish split separation without both row-ID lists")

    for field_name, fit_raw in (("model_fit", evidence.model_fit_ids), ("preprocessor_fit", evidence.preprocessor_fit_ids)):
        if field_name == "preprocessor_fit" and evidence.preprocessor_applied is False:
            if fit_raw is None:
                note("training", "preprocessor_fit_ids", "not_applicable", "The declared pipeline has no fitted preprocessor")
                note("training", "preprocessor_fit_isolation", "not_applicable", "No preprocessor fit was declared; the claim is not authenticated")
            else:
                note("training", "preprocessor_fit_isolation", "problem", "A pipeline declared no preprocessor but supplies preprocessor fit IDs")
            continue
        fit_ids = ids(field_name + "_ids", fit_raw, None)
        if fit_ids is None or eval_ids is None:
            note("training", field_name + "_isolation", "unknown", "Cannot inspect fit/evaluation overlap")
            continue
        overlap = len(fit_ids & eval_ids)
        if overlap:
            note("training", field_name + "_isolation", "problem", "Declared fit inputs include evaluation row IDs", overlap)
        elif train_ids is not None and not fit_ids.issubset(train_ids):
            note("training", field_name + "_isolation", "warning", "Fit IDs include rows outside the declared training partition")
        else:
            note("training", field_name + "_isolation", "ok", "No overlap in declared IDs; actual fit calls cannot be authenticated")

    for observed in audit_fit_trace(
        evidence.fit_records,
        train_features=evidence.train_features,
        evaluation_features=evidence.evaluation_features,
        train_ids=evidence.train_ids,
        evaluation_ids=evidence.evaluation_ids,
        model_fit_ids=evidence.model_fit_ids,
        preprocessor_fit_ids=evidence.preprocessor_fit_ids,
        preprocessor_applied=evidence.preprocessor_applied,
        train_targets=evidence.train_targets,
        evaluation_targets=evidence.evaluation_targets,
        target_hmac_key=target_hmac_key,
    ):
        note("training", observed.code, observed.status, observed.detail, observed.observed)

    for transformed in audit_transform_chain(
        evidence.transform_records,
        train_features=evidence.train_features,
        evaluation_features=evidence.evaluation_features,
        train_ids=evidence.train_ids,
        evaluation_ids=evidence.evaluation_ids,
        fit_records=evidence.fit_records,
        output_records=evidence.output_records,
        preprocessor_applied=evidence.preprocessor_applied,
        task=evidence.task,
    ):
        note("training" if transformed.code in (
            "transform_train_input", "transform_preprocessor_fit_link",
            "transform_model_fit_link", "transform_identity", "transform_trace",
        ) else "output", transformed.code, transformed.status, transformed.detail)

    if evidence.model_version is None or evidence.random_seed is None:
        note("training", "run_identity", "unknown", "Version and seed are both needed for a minimal run identity; they do not guarantee reproducibility")
    else:
        note("training", "run_identity", "ok", "Version and seed were declared, not independently authenticated")

    for name, raw in (("train_loss", evidence.train_loss), ("validation_loss", evidence.validation_loss)):
        if raw is None or len(raw) == 0:
            note("training", name, "unknown", "No observed loss history was supplied")
            continue
        try:
            values = np.asarray(raw, dtype=float)
        except (TypeError, ValueError, OverflowError):
            note("training", name, "problem", "Loss history is not numeric")
            continue
        if values.ndim != 1 or not np.isfinite(values).all():
            note("training", name, "problem", "Loss history must be one-dimensional and finite")
            continue
        metrics[name + "_first"] = float(values[0])
        metrics[name + "_last"] = float(values[-1])
        metrics[name + "_minimum"] = float(np.min(values))
        note("training", name, "ok", "Observed loss history recorded; change alone does not establish convergence or overfitting", len(values))

    def vector(raw: Any | None, stage: Stage, name: str, expected: int | None = None) -> np.ndarray | None:
        if raw is None:
            note(stage, name, "unknown", "Values were not supplied")
            return None
        try:
            arr = np.asarray(raw)
        except (ValueError, TypeError):
            note(stage, name, "problem", "Cannot interpret one-dimensional values")
            return None
        if arr.ndim != 1 or len(arr) == 0 or (expected is not None and len(arr) != expected):
            note(stage, name, "problem", "Expected nonempty 1D values matching evaluation row count")
            return None
        if np.issubdtype(arr.dtype, np.number):
            try:
                finite = np.isfinite(arr.astype(float))
            except (ValueError, TypeError, OverflowError):
                finite = np.zeros(arr.shape, dtype=bool)
            if not finite.all():
                note(stage, name, "problem", "Numeric values include NaN, infinity, or unrepresentable numbers")
                return None
        note(stage, name, "ok", "One-dimensional values have valid shape", len(arr))
        return arr

    expected = len(x_eval) if x_eval is not None else None
    y_pred = vector(evidence.predictions, "output", "predictions", expected)
    y_true = vector(evidence.evaluation_targets, "output", "evaluation_targets", expected)
    if x_eval is None:
        note("output", "prediction_binding", "unknown", "Without evaluation features, sample alignment cannot be verified")
    elif y_pred is not None:
        note("output", "prediction_binding", "ok", "Prediction count matches evaluation feature rows; row ordering is declared, not authenticated")

    for witnessed in audit_output_calls(
        evidence.output_records,
        evaluation_features=evidence.evaluation_features,
        evaluation_ids=evidence.evaluation_ids,
        predictions=evidence.predictions,
        probabilities=evidence.probabilities,
        probability_classes=evidence.probability_classes,
        task=evidence.task,
    ):
        note("output", witnessed.code, witnessed.status, witnessed.detail, witnessed.observed)

    if evidence.task == "classification":
        train_y = vector(evidence.train_targets, "data", "train_targets", len(x_train) if x_train is not None else None)
        if train_y is not None:
            try:
                values, counts = np.unique(train_y, return_counts=True)
                metrics["train_classes"] = len(values)
                metrics["minimum_class_share"] = float(np.min(counts) / len(train_y))
                if len(values) < 2:
                    note("data", "train_class_diversity", "problem", "Training targets have only one class")
                else:
                    note("data", "train_class_diversity", "ok", "Training class frequencies were inspected")
            except TypeError:
                note("data", "train_class_diversity", "unknown", "Classes are not consistently comparable")
        else:
            note("data", "train_class_diversity", "unknown", "Cannot determine training class frequencies")

        if evidence.probabilities is None:
            note("output", "probabilities", "unknown", "Probability output was not supplied")
        else:
            try:
                proba = np.asarray(evidence.probabilities, dtype=float)
                good = proba.ndim == 2 and proba.shape[1] >= 2 and (expected is None or len(proba) == expected)
                good = good and bool(np.isfinite(proba).all()) and bool(((proba >= 0) & (proba <= 1)).all())
                good = good and bool(np.allclose(proba.sum(axis=1), 1.0, atol=1e-5, rtol=0))
            except (TypeError, ValueError, OverflowError):
                good = False
            if not good:
                note("output", "probabilities", "problem", "Probabilities must be finite n×k values in [0,1] with rows summing to one")
            else:
                note("output", "probabilities", "ok", "Probability matrix is structurally valid")
                if evidence.probability_classes is None:
                    note("output", "class_mapping", "unknown", "Probability column-to-class mapping was not provided")
                elif len(evidence.probability_classes) != proba.shape[1] or len(set(evidence.probability_classes)) != len(evidence.probability_classes):
                    note("output", "class_mapping", "problem", "Probability classes must be unique and match column count")
                elif y_pred is not None and len(y_pred) == len(proba):
                    predicted = np.asarray(evidence.probability_classes, dtype=str)[np.argmax(proba, axis=1)]
                    conflicts = int(np.sum(predicted != y_pred.astype(str)))
                    note("output", "class_mapping", "warning" if conflicts else "ok",
                         "Argmax differs from declared classes; may reflect a separate decision policy" if conflicts else "Predictions agree with declared probability classes",
                         conflicts)
                else:
                    note("output", "class_mapping", "unknown", "Predictions or class mapping are insufficient for consistency check")
        if y_true is not None and y_pred is not None and len(y_true) == len(y_pred):
            try:
                metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
                metrics["macro_f1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
                note("output", "label_metrics", "ok", "Accuracy and macro-F1 were measured; no quality threshold is assumed")
            except (TypeError, ValueError):
                note("output", "label_metrics", "problem", "True and predicted class labels are incompatible")
        else:
            note("output", "label_metrics", "unknown", "Metrics require aligned predictions and evaluation labels")
    elif evidence.task == "regression":
        note("output", "probabilities", "not_applicable", "Classification probability check does not apply to regression")
        if y_true is not None and y_pred is not None and len(y_true) == len(y_pred):
            try:
                truth = np.asarray(y_true, dtype=float)
                estimated = np.asarray(y_pred, dtype=float)
                metrics["mae"] = float(mean_absolute_error(truth, estimated))
                metrics["rmse"] = float(np.sqrt(np.mean((truth - estimated) ** 2)))
                if len(truth) >= 2 and np.var(truth) > 0:
                    metrics["r2"] = float(r2_score(truth, estimated))
                note("output", "label_metrics", "ok", "Regression errors measured without assigning a universal quality grade")
            except (TypeError, ValueError, OverflowError):
                note("output", "label_metrics", "problem", "Regression values are not numeric or finite")
        else:
            note("output", "label_metrics", "unknown", "Regression metrics require aligned predictions and evaluation targets")
    else:
        note("output", "probabilities", "not_applicable", "Class probability checks do not apply to clustering")
        note("output", "label_metrics", "not_applicable", "External labels are not assumed in unsupervised clustering")
        if y_pred is not None:
            try:
                unique, counts = np.unique(y_pred, return_counts=True)
                metrics["clusters_observed"] = len(unique)
                metrics["smallest_cluster_size"] = int(min(counts))
                note("output", "clusters", "warning" if len(unique) < 2 else "ok",
                     "Only one cluster was observed" if len(unique) < 2 else "Observed cluster counts; no cluster-quality guarantee")
            except TypeError:
                note("output", "clusters", "unknown", "Cluster labels are not consistently comparable")
        else:
            note("output", "clusters", "unknown", "Cluster assignments are unavailable")

    counts = {name: sum(f.status == name for f in findings) for name in ("problem", "warning", "unknown")}
    overall = "observed_with_problems" if counts["problem"] else "observed_with_warnings" if counts["warning"] else "incomplete_evidence" if counts["unknown"] else "checks_completed"
    return MLObservation(evidence.task, evidence.model_family, overall, tuple(findings), metrics)



def main() -> None:
    """CLI for already-collected JSON evidence; does not fit or query a model."""

    parser = argparse.ArgumentParser(description="Read-only ML lifecycle inspection: data, training, output")
    parser.add_argument("evidence", type=Path, help="Path to a JSON object with MLRunEvidence fields")
    parser.add_argument("--output", type=Path, help="Optional destination for the JSON report")
    args = parser.parse_args()
    document = json.loads(args.evidence.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        parser.error("evidence JSON must be an object")
    report = observe_ml_run(MLRunEvidence(**document))
    serialized = json.dumps(report.to_dict(), indent=2, ensure_ascii=False, allow_nan=False)
    if args.output is None:
        print(serialized)
    else:
        args.output.write_text(serialized + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
