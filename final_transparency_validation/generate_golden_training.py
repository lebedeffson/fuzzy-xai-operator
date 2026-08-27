"""Same-run SGD training observation exported through the public API."""

from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path

import numpy as np
from fuzzyxai import FuzzyXAI, ObservationContext
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

OUT = Path(__file__).resolve().parent / "golden_training"


def main() -> None:
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=.2, random_state=19, stratify=data.target)
    scaler = StandardScaler().fit(X_train)
    Xs, Xts = scaler.transform(X_train), scaler.transform(X_test)
    classifier = SGDClassifier(loss="log_loss", random_state=19, learning_rate="constant", eta0=.015, penalty="l2")
    rng = np.random.default_rng(19)
    histories: list[list[dict[str, object]]] = [[] for _ in range(len(X_test))]
    for epoch in range(1, 31):
        order = rng.permutation(len(Xs))
        classifier.partial_fit(Xs[order], y_train[order], classes=np.array([0, 1]))
        probabilities = classifier.predict_proba(Xts)
        predictions = classifier.predict(Xts)
        for index, (prediction, probability, truth) in enumerate(zip(predictions, probabilities, y_test)):
            confidence = float(probability[int(prediction)])
            true_probability = max(float(probability[int(truth)]), 1e-15)
            histories[index].append({
                "epoch": epoch, "predicted_class": int(prediction), "confidence": confidence,
                "correct": bool(prediction == truth), "loss": float(-np.log(true_probability)),
            })
    # Select a genuinely observed forgetting/confidence-drop trajectory when available.
    def has_event(history: list[dict[str, object]]) -> bool:
        return any(bool(a["correct"]) and (not bool(b["correct"]) or float(a["confidence"]) - float(b["confidence"]) >= .2) for a, b in pairwise(history))
    selected = next((index for index, history in enumerate(histories) if has_event(history)), 0)
    object_id = f"training_test_{selected}"
    final_model = Pipeline([("scaler", scaler), ("classifier", classifier)])
    probe = FuzzyXAI.wrap(final_model)
    training_run = probe.observe_training(
        history={"objects": {object_id: histories[selected]}},
        training_run_id="sgd-bcw-p19-run-19", training_method="SGDClassifier.partial_fit(log_loss)",
        epoch_source="measured after each partial_fit epoch", final_checkpoint_ref="epoch:30",
    )
    context = ObservationContext(
        reference_data=X_train, reference_labels=y_train, training_run=training_run,
        dataset_version="breast_cancer_wisconsin_sgd_p19", run_parameters={"split": {"random_state": 19, "test_size": .2}},
    )
    result = FuzzyXAI.wrap(final_model, observation_context=context).explain_one(
        X_test[selected], object_id=object_id, include_training_trace=True, feature_names=list(data.feature_names),
    )
    OUT.mkdir(exist_ok=True)
    (OUT / "full_report_reader_ru.txt").write_text(result.full_report(level="reader"), encoding="utf-8")
    (OUT / "full_report_audit_ru.txt").write_text(result.full_report(level="audit"), encoding="utf-8")
    result.export_json(OUT / "result.json", detail="audit")
    (OUT / "audit.json").write_text(json.dumps(result.audit(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "provenance.json").write_text(json.dumps(result.inspect("action").to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result.visualize(view="provenance", output=OUT / "provenance_action.png", selector="action")
    (OUT / "limitations.txt").write_text("History and final explanation share the fingerprint recorded by FuzzyXAI.observe_training. Loss is measured negative log-likelihood for the selected object.\n", encoding="utf-8")
    trace = result.view_model.layers["training"][0]
    print(json.dumps({"prediction": result.prediction.predictions, "training": trace}, ensure_ascii=False))


if __name__ == "__main__":
    main()
