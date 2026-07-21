#!/usr/bin/env python3
"""Build frozen blinded study materials without generating human responses."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "study/q1_final"
BASE_COMMIT = "41c32af25242164144fd907e4850fa9d4f426bd1"


SCENARIO_TYPES = (
    ("simple", 6),
    ("low_confidence", 6),
    ("incorrect_high_confidence", 4),
    ("incomplete_evidence", 4),
    ("conflict", 2),
    ("shift", 2),
)


def dump(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scenarios() -> list[dict[str, object]]:
    rows = []
    number = 1
    for scenario_type, count in SCENARIO_TYPES:
        for offset in range(count):
            confidence = round(0.93 - ((number * 7) % 38) / 100, 2)
            action = "review" if scenario_type != "simple" else "accept"
            limitation = {
                "simple": "No material limitation was detected in the available route.",
                "low_confidence": "The model confidence is close to the decision boundary.",
                "incorrect_high_confidence": "High confidence conflicts with held-out outcome evidence.",
                "incomplete_evidence": "A required provenance channel is absent.",
                "conflict": "Two evidence channels support different actions.",
                "shift": "The object differs from the training reference distribution.",
            }[scenario_type]
            rows.append(
                {
                    "stimulus_id": f"stimulus_{number:03d}",
                    "scenario_type": scenario_type,
                    "object_id": f"q1_object_{number:03d}",
                    "ground_truth_action": action,
                    "condition_a": {
                        "title": "Model output",
                        "prediction": "elevated risk" if number % 3 else "moderate risk",
                        "confidence": confidence,
                        "local_reason": "The strongest local contribution is fracture density.",
                        "warning": "This local explanation is associational.",
                    },
                    "condition_b": {
                        "title": "Decision review",
                        "decision": "elevated risk" if number % 3 else "moderate risk",
                        "reasons": ["fracture density", "water saturation", "distance to excavation"],
                        "limitations": [limitation],
                        "recommended_action": action,
                        "provenance_available": scenario_type != "incomplete_evidence",
                    },
                    "questions": {
                        "decision": "Which decision did the model produce?",
                        "reason": "Which factor most strongly supports the decision?",
                        "limitation": "What is the main limitation?",
                        "action": "Which action is appropriate?",
                        "overtrust": "Should this result be used automatically?",
                    },
                }
            )
            number += 1
    return rows


def assignments(stimuli: list[dict[str, object]]) -> list[dict[str, object]]:
    ids = [str(row["stimulus_id"]) for row in stimuli]
    rows = []
    for slot in range(40):
        selected = [ids[(slot + 2 * index) % len(ids)] for index in range(12)]
        order = "A_then_B" if slot % 2 == 0 else "B_then_A"
        rows.append(
            {
                "assignment_slot": f"slot_{slot + 1:03d}",
                "condition_order": order,
                "stimulus_ids": selected,
                "object_order_seed": 9200 + slot,
                "participant_id": None,
            }
        )
    return rows


def expert_objects(stimuli: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    labels = ("ordinary", "rare", "incorrect", "conflict", "incomplete", "shift", "boundary")
    for index in range(100):
        source = stimuli[index % len(stimuli)]
        rows.append(
            {
                "object_id": f"expert_object_{index + 1:03d}",
                "stratum": labels[index % len(labels)],
                "source_stimulus_id": source["stimulus_id"],
                "blinded_conditions": ["condition_1", "condition_2", "condition_3"],
                "response": None,
            }
        )
    return rows


def domain_cards(stimuli: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    levels = (("user", 30), ("expert", 20), ("audit", 10))
    counter = 0
    for level, count in levels:
        for _ in range(count):
            source = stimuli[counter % len(stimuli)]
            rows.append(
                {
                    "card_id": f"domain_card_{counter + 1:03d}",
                    "level": level,
                    "stimulus_id": source["stimulus_id"],
                    "decision": source["condition_b"]["decision"],
                    "limitation": source["condition_b"]["limitations"][0],
                    "action": source["condition_b"]["recommended_action"],
                    "associational_label": True,
                }
            )
            counter += 1
    return rows


def main() -> None:
    stimulus_rows = scenarios()
    dump(STUDY / "stimuli_manifest/stimuli.json", stimulus_rows)
    dump(STUDY / "comprehension/assignments.json", assignments(stimulus_rows))
    dump(STUDY / "expert_action_review/objects.json", expert_objects(stimulus_rows))
    dump(STUDY / "domain_language_review/cards.json", domain_cards(stimulus_rows))
    ethics_status = STUDY / "ethics/status.json"
    if not ethics_status.exists():
        dump(
            ethics_status,
            {
                "schema_version": "1.0",
                "status": "not_started",
                "recruitment_allowed": False,
                "public_record": None,
                "full_document": None,
            },
        )
    dump(
        STUDY / "comprehension/study_design.json",
        {
            "design": "blinded randomized within-subject A/B",
            "minimum_valid_participants": 24,
            "target_participants": 40,
            "stimuli_per_participant": 12,
            "minimum_effect_limitation_accuracy": 0.05,
            "minimum_effect_action_accuracy": 0.05,
            "unsafe_overtrust_noninferiority_margin": 0.02,
            "condition_a": "strong simple baseline",
            "condition_b": "FuzzyXAI human explanation",
            "frozen_before_recruitment": True,
        },
    )
    dump(
        STUDY / "comprehension/power_analysis.json",
        {
            "method": "paired binary-outcome design; conservative preregistered minimum",
            "minimum_valid_participants": 24,
            "target_participants": 40,
            "status": "design_frozen_not_recruited",
            "limitation": "final achieved power must be recomputed from genuine anonymized records",
        },
    )
    dump(
        STUDY / "expert_action_review/study_design.json",
        {
            "minimum_reviewers": 3,
            "shared_objects": 100,
            "minimum_agreement_gain": 0.02,
            "unsafe_accept_noninferiority_margin": 0.01,
            "gold_standard": "majority consensus with disagreement retained",
            "frozen_before_review": True,
        },
    )
    excluded_parts = {"raw_anonymized", "signed_records"}
    files = sorted(
        path
        for path in STUDY.rglob("*")
        if path.is_file() and not excluded_parts.intersection(path.relative_to(STUDY).parts)
    )
    scorer_paths = {
        "comprehension": ROOT / "scripts/q1_final/score_comprehension.py",
        "domain_language_review": ROOT / "scripts/q1_final/score_domain_review.py",
        "expert_action_review": ROOT / "scripts/q1_final/score_expert_action_review.py",
    }
    manifest = {
        "schema_version": "2.0",
        "base_commit": BASE_COMMIT,
        "record_origin": "frozen_stimuli_and_blank_templates_only",
        "human_responses_generated": False,
        "method_identity_blinded": True,
        "stimulus_count": len(stimulus_rows),
        "assignment_slots": 40,
        "expert_object_count": 100,
        "domain_card_count": 60,
        "scorer_sha256": {name: sha(path) for name, path in scorer_paths.items()},
        "files": [
            {"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path), "bytes": path.stat().st_size}
            for path in files
            if path.name != "manifest.json"
        ],
    }
    dump(STUDY / "stimuli_manifest/manifest.json", manifest)
    print("PASS: q1_final_external_study_pack stimuli=24 expert_objects=100 domain_cards=60")


if __name__ == "__main__":
    main()
