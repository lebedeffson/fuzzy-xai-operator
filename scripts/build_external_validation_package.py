"""Freeze the external validation inputs without fabricating human evidence."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from score_comprehension_pilot import score_rows


ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "release_evidence/user_study/comprehension_pilot"
DOMAIN = ROOT / "release_evidence/domain_language_review"
EMPIRICAL = ROOT / "release_evidence/empirical_experiments/breast_cancer_checkpoint"
MEDICAL = ROOT / "release_evidence/explanation_experience/medical_research_similarity_explanation.json"

RESPONSE_FIELDS = (
    "participant_id",
    "role",
    "condition_order",
    "scenario_id",
    "mode",
    "decision_correct",
    "reasons_correct",
    "concern_correct",
    "reliability_correct",
    "action_correct",
    "limitation_correct",
    "provenance_correct",
    "similarity_correct",
    "counterfactual_correct",
    "native_surrogate_correct",
    "overtrust_error",
    "iou_misinterpreted_as_probability",
    "sensitivity_misinterpreted_as_recommendation",
    "unsupported_inference_count",
    "completion_time_sec",
    "subjective_clarity_1_5",
    "cognitive_load_1_5",
    "notes",
)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def write_empty_csv(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.DictWriter(handle, fieldnames=RESPONSE_FIELDS, lineterminator="\n").writeheader()


def build_pilot() -> None:
    PILOT.mkdir(parents=True, exist_ok=True)
    sources = {
        "forgetting_case": EMPIRICAL / "selected_forgetting_case.json",
        "rule_ablation": EMPIRICAL / "rule_ablation.json",
        "image_similarity": MEDICAL,
    }
    missing = [str(path.relative_to(ROOT)) for path in sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"pilot source artifacts are missing: {missing}")

    scenario_manifest = {
        "schema_version": "1.0",
        "commit": git_head(),
        "randomization": "counterbalanced AB/BA within participant",
        "scenarios": [
            {
                "scenario_id": scenario_id,
                "artifact": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
                "claim_boundary": (
                    "research_only; IoU and embedding similarity are not diagnostic probabilities"
                    if scenario_id == "image_similarity"
                    else "measured evidence for this dataset, split, seed, and model configuration"
                ),
            }
            for scenario_id, path in sources.items()
        ],
    }
    write_json(PILOT / "scenario_manifest.json", scenario_manifest)
    (PILOT / "protocol.md").write_text(
        """# Independent A/B comprehension pilot

Status: `planned_not_run` until real anonymized responses are supplied.

## Participants

At least six independent people: at least three domain specialists and three model integrators.
No project author may be counted as an external participant.

## Design

Every participant evaluates `technical_baseline` and `human_explanation` for all three frozen scenarios.
Use counterbalanced `AB`/`BA` order. Record correctness, interpretation errors, completion time, clarity,
and cognitive load. The study evaluates the explanation, not participant ability.

## Prohibited conduct

Do not fabricate rows, names, identities, answers, timings, consent, or reviewer conclusions. Do not collect
names, emails, diagnoses, or unnecessary free text in the repository copy.
""",
        encoding="utf-8",
    )
    (PILOT / "participant_information.md").write_text(
        """# Participant information

This voluntary study compares two explanation formats. Participation can stop at any time. Only an anonymous
participant ID, role group, answers, timings, and optional non-identifying notes are retained.
""",
        encoding="utf-8",
    )
    (PILOT / "consent_template.md").write_text(
        """# Consent template

I voluntarily consent to the anonymous use of my task answers and timings for evaluation of the FuzzyXAI
explanation interface. I understand that the medical image scenario is research-only and is not a diagnosis.
""",
        encoding="utf-8",
    )
    write_empty_csv(PILOT / "response_template.csv")
    responses = PILOT / "anonymized_responses.csv"
    if not responses.exists():
        write_empty_csv(responses)
    shutil.copy2(ROOT / "scripts/score_comprehension_pilot.py", PILOT / "score_comprehension_pilot.py")
    with responses.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    report = score_rows(rows)
    write_json(PILOT / "scoring_report.json", report)
    (PILOT / "analysis.md").write_text(
        "# Pilot analysis\n\n"
        f"Status: `{report['status']}`. Participant count: {report['participant_count']}. "
        "No demonstrated-comprehensibility claim is allowed unless the scorer returns `pass`.\n",
        encoding="utf-8",
    )
    (PILOT / "limitations.md").write_text(
        "# Limitations\n\nA prepared protocol is not user evidence. External responses and independent "
        "domain review remain mandatory release gates.\n",
        encoding="utf-8",
    )


def build_domain_review() -> None:
    DOMAIN.mkdir(parents=True, exist_ok=True)
    language = EMPIRICAL / "domain_language.json"
    if not language.is_file():
        raise FileNotFoundError(language)
    record = DOMAIN / "review_record.json"
    if not record.exists():
        write_json(
            record,
            {
                "schema_version": "1.0",
                "status": "pending_external_review",
                "reviewer_id": None,
                "reviewer_role": None,
                "independent_of_project": None,
                "domain_language_artifact": str(language.relative_to(ROOT)),
                "domain_language_sha256": sha256(language),
                "review_date": None,
                "approved_terms": [],
                "rejected_terms": [],
                "comments": [],
                "claim_allowed": False,
            },
        )
    (DOMAIN / "review_protocol.md").write_text(
        """# External domain-language review

An independent subject-matter expert checks every user-facing label, meaning, comparison, direction, limitation,
and action. Approval applies only to the exact SHA256 in `review_record.json`. The reviewer must record an
anonymous reviewer ID, role, date, decisions, and comments. A project author cannot self-approve this gate.
""",
        encoding="utf-8",
    )


def write_manifests() -> None:
    for directory in (PILOT, DOMAIN):
        files = {
            str(path.relative_to(directory)): sha256(path)
            for path in sorted(directory.rglob("*"))
            if path.is_file() and path.name != "manifest_sha256.json"
        }
        write_json(directory / "manifest_sha256.json", {"schema_version": "1.0", "files": files})


def main() -> None:
    build_pilot()
    build_domain_review()
    write_manifests()
    print("PASS: external_validation_package")
    print("BLOCKED: comprehension_pilot planned_not_run")
    print("BLOCKED: domain_language pending_external_review")


if __name__ == "__main__":
    main()
