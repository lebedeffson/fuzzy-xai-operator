from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, log_loss, recall_score

from fuzzyxai import CounterfactualEvidence, ExplanationEvidence, FuzzyXAI, LearnedRule, evaluate_rule_ablation


SEED = 42


def run(output_dir: str | Path) -> dict[str, object]:
    """Reproduce learning, forgetting, and restoration for rare object 85."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    negative = rng.normal([-1.0, -1.0], [0.45, 0.45], (120, 2))
    common_positive = rng.normal([1.0, 1.0], [0.45, 0.45], (100, 2))
    rare_positive = rng.normal([-0.8, 0.2], [0.12, 0.12], (8, 2))
    values = np.vstack([negative, common_positive, rare_positive])
    labels = np.asarray([0] * 120 + [1] * 100 + [1] * 8)
    rare_indices = np.arange(220, 228)
    object_index = 220
    object_id = "85"

    model = SGDClassifier(loss="log_loss", alpha=0.01, random_state=SEED, learning_rate="constant", eta0=0.05)
    object_metrics: list[dict[str, object]] = []
    global_accuracy: list[float] = []
    rare_recall: list[float] = []
    rule_history: list[list[str]] = []
    for epoch in range(40):
        if epoch < 12:
            train_indices = np.concatenate([np.arange(len(values)), rare_indices.repeat(30)])
        else:
            train_indices = np.arange(220)
        rng.shuffle(train_indices)
        model.partial_fit(values[train_indices], labels[train_indices], classes=[0, 1] if epoch == 0 else None)
        probabilities = model.predict_proba(values)
        predictions = model.predict(values)
        confidence = float(probabilities[object_index, 1])
        object_metrics.append(
            {
                "epoch": epoch,
                "predicted_class": int(predictions[object_index]),
                "confidence": confidence,
                "loss": float(log_loss([labels[object_index]], [probabilities[object_index]], labels=[0, 1])),
                "correct": bool(predictions[object_index] == labels[object_index]),
                "embedding": [float(item) for item in model.decision_function([values[object_index]])],
                "rule_activations": {"R_rare_subtype_surrogate": confidence},
            }
        )
        global_accuracy.append(float(accuracy_score(labels, predictions)))
        rare_recall.append(float(recall_score(labels[rare_indices], predictions[rare_indices], zero_division=0)))
        rule_history.append(["R_rare_subtype_surrogate"] if float(np.mean(probabilities[rare_indices, 1])) >= 0.5 else [])

    forgotten_model = deepcopy(model)
    restored_model = deepcopy(model)
    restore_indices = np.concatenate([np.arange(len(values)), rare_indices.repeat(30)])
    for _ in range(5):
        rng.shuffle(restore_indices)
        restored_model.partial_fit(values[restore_indices], labels[restore_indices])

    baseline_predictions = forgotten_model.predict(values)
    restored_predictions = restored_model.predict(values)
    baseline = {
        "test_accuracy": float(accuracy_score(labels, baseline_predictions)),
        "rare_subtype_recall": float(recall_score(labels[rare_indices], baseline_predictions[rare_indices], zero_division=0)),
    }
    restored = {
        "test_accuracy": float(accuracy_score(labels, restored_predictions)),
        "rare_subtype_recall": float(recall_score(labels[rare_indices], restored_predictions[rare_indices], zero_division=0)),
    }

    fx = FuzzyXAI.wrap(forgotten_model, adapter="sklearn")
    training = fx.observe_training(
        history={
            "objects": {object_id: object_metrics},
            "global_metric": [global_accuracy[10], global_accuracy[-1]],
            "subgroup_metrics": {"rare_positive": [rare_recall[10], rare_recall[-1]]},
            "subgroup_objects": {"rare_positive": [object_id, *[f"rare_{index}" for index in range(1, 8)]]},
            "subgroup_rule_history": {"rare_positive": [rule_history[10], rule_history[-1]]},
        }
    )
    monitored_rule = LearnedRule(
        rule_id="R_rare_subtype_surrogate",
        model_version=fx.model_adapter.model_fingerprint()[:12],
        antecedents=["marker_a is atypically low", "marker_b supports the positive rare subtype"],
        consequent="1",
        activation=float(object_metrics[-1]["confidence"]),
        coverage=len(rare_indices) / len(values),
        precision=None,
        support=len(rare_indices),
        stability=None,
        importance=None,
        counterfactual_effect={},
        source_objects=[object_id, *[f"rare_{index}" for index in range(1, 8)]],
        class_distribution={"1": 1.0},
        human_text="rare positive subtype identified by the monitored marker region",
        complexity=2.0,
        is_primary=True,
        is_redundant=False,
        is_conflicting=False,
        native=False,
        surrogate=True,
        fidelity=None,
        evidence_refs=["controlled_training_protocol:R_rare_subtype_surrogate"],
    )
    monitored_rule = evaluate_rule_ablation(
        monitored_rule,
        baseline_metrics={"test_accuracy": restored["test_accuracy"], "rare_subtype_recall": restored["rare_subtype_recall"]},
        ablated_metrics={"test_accuracy": baseline["test_accuracy"], "rare_subtype_recall": baseline["rare_subtype_recall"]},
    )
    restoration = CounterfactualEvidence(
        source_prediction=int(forgotten_model.predict([values[object_index]])[0]),
        target_prediction=int(restored_model.predict([values[object_index]])[0]),
        changed_features={},
        changed_regions=[],
        changed_rules=["R_rare_subtype_surrogate"],
        minimality=None,
        plausibility=1.0,
        stability=None,
        expected_effect=None,
        observed_effect=restored["rare_subtype_recall"] - baseline["rare_subtype_recall"],
        actionability="retrain and revalidate the rare subgroup",
        limitations=[
            f"overall test accuracy changed by {restored['test_accuracy'] - baseline['test_accuracy']:+.6f}",
            "controlled synthetic protocol; effect must be re-measured for another dataset",
        ],
        evidence_refs=["object_85_restoration_protocol"],
    )
    result = fx.explain_one(
        values[object_index],
        object_id=object_id,
        reference_data=values,
        reference_ids=[str(index) for index in range(len(values))],
        reference_labels=labels.tolist(),
        feature_names=["marker_a", "marker_b"],
        training_run=training,
        include_similar_cases=True,
        include_counterfactuals=True,
        include_training_trace=True,
        additional_evidence=ExplanationEvidence(rules=[monitored_rule], counterfactuals=[restoration]),
    )
    result.export_json(output / "object_85_explanation.json")
    result.export_html(output / "object_85_explanation.html")
    result.plot(output / "object_85_dashboard.png")
    (output / "object_85_user.md").write_text(result.summary("user"), encoding="utf-8")
    training.plot_object_trajectory(object_id, str(output / "object_85_training_trajectory.png"))
    report = {
        "protocol": "controlled rare-subtype forgetting and restoration",
        "seed": SEED,
        "object_id": object_id,
        "first_learned_epoch": training.traces[object_id].first_learned_epoch,
        "forgetting_events": list(training.traces[object_id].forgetting_events),
        "averaged_subgroups": [item.to_dict() for item in training.find_averaged_subgroups()],
        "before_restoration": baseline,
        "after_restoration": restored,
        "effect": {key: restored[key] - baseline[key] for key in baseline},
        "claim_scope": "controlled protocol; not a clinical or production benchmark",
    }
    (output / "object_85_restoration_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="release_evidence/object_85")
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
