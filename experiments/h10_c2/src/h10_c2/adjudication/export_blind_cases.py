from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict

from ..data import generate_cases
from ..hashing import object_sha256, read_json, write_json
from ..paths import ARTIFACT_ROOT


FIELDS = (
    "case_id",
    "mutation_log_consistent",
    "optimal_cuts_valid",
    "repair_actions_valid",
    "additional_valid_variants_json",
    "ambiguous",
    "sufficient_evidence",
    "comments",
    "reviewer_signature",
)


def export_blind_cases(sample_size: int = 200) -> dict:
    manifest = read_json(ARTIFACT_ROOT / "data" / "protocol_validation" / "manifest.json")
    cases = generate_cases("protocol_validation", int(manifest["case_count"]), seed=221005)
    selected = random.Random(221099).sample(cases, k=min(sample_size, len(cases)))
    root = ARTIFACT_ROOT / "adjudication" / "blind"
    orders = {}
    for reviewer_index in (1, 2):
        reviewer = root / f"reviewer_{reviewer_index}"
        reviewer.mkdir(parents=True, exist_ok=True)
        ordered = list(selected)
        random.Random(221100 + reviewer_index).shuffle(ordered)
        orders[str(reviewer_index)] = [case.case_id for case in ordered]
        with (reviewer / "cases.jsonl").open("w", encoding="utf-8") as stream:
            for case in ordered:
                payload = {
                    "case_id": case.case_id,
                    "clean_route": case.clean_route,
                    "observed_route": case.observed_route,
                    "mutation_log": [asdict(item) for item in case.transactions],
                }
                stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        with (reviewer / "form.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            for case in ordered:
                writer.writerow({"case_id": case.case_id})
        (reviewer / "instructions.md").write_text(
            "# Инструкция рецензенту\n\n"
            "Проверьте согласованность mutation log, допустимые минимальные разрезы, "
            "действия восстановления, неоднозначность и достаточность сведений. "
            "Заполните все поля и укажите собственную подпись.\n",
            encoding="utf-8",
        )
    output = {
        "sample_size": len(selected),
        "same_cases": set(orders["1"]) == set(orders["2"]),
        "different_order": orders["1"] != orders["2"],
        "case_ids_sha256": object_sha256(sorted(orders["1"])),
        "answers_prefilled": False,
    }
    write_json(root / "manifest.json", output)
    write_json(ARTIFACT_ROOT / "adjudication" / "status.json", {"status": "BLOCKED_HUMAN_ADJUDICATION"})
    return output

