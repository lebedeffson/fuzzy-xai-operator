"""Build Chapter 4 empirical evidence from measured runs, never edited goldens."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EMPIRICAL = ROOT / "release_evidence/empirical_experiments/breast_cancer_checkpoint"
CONTROLLED = ROOT / "release_evidence/controlled_fixtures/object_85_controlled_story_fixture"
OUTPUT = ROOT / "release_evidence/chapter4_empirical_validation"
USER_STUDY = ROOT / "release_evidence/user_study/comprehension_pilot"
ARCHIVE = ROOT / "release_evidence/chapter4_empirical_validation_evidence.zip"


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def deterministic_zip(source: Path, target: Path) -> None:
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source.rglob("*")):
            if not path.is_file():
                continue
            info = zipfile.ZipInfo(
                str(Path(source.name) / path.relative_to(source)),
                date_time=(2020, 1, 1, 0, 0, 0),
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes())


def write_checkpoint_table(checkpoints: list[dict[str, object]], path: Path) -> None:
    fields = (
        "checkpoint_id",
        "epoch",
        "model_fingerprint",
        "train_metric",
        "validation_metric",
        "test_metric",
        "subgroup_metric",
        "random_seed",
        "result_origin",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in checkpoints)


def write_cross_model_table(rows: list[dict[str, object]], path: Path) -> None:
    fields = (
        "model",
        "model_fingerprint",
        "prediction",
        "explanation_level",
        "native_channels",
        "surrogate_channels",
        "missing_channels",
        "native_rule_count",
        "action",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: ";".join(str(item) for item in row[field])
                    if isinstance(row.get(field), list)
                    else row.get(field)
                    for field in fields
                }
            )


def build_figures(summary: dict[str, object], selected: dict[str, object], checkpoints: list[dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    figure_dir = OUTPUT / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    history = selected["history"]
    epochs = [int(item["epoch"]) for item in history]
    confidence = [float(item["confidence"]) for item in history]
    loss = [float(item["loss"]) for item in history]
    correct = [bool(item["correct"]) for item in history]
    figure, axes = plt.subplots(3, 1, figsize=(16, 9), sharex=True)
    axes[0].plot(epochs, confidence, marker="o", color="#275d73", label="true-class confidence")
    axes[0].axhline(0.5, color="#9b3d2c", linestyle="--", label="decision boundary")
    axes[0].legend(loc="best")
    axes[0].set_ylabel("Confidence")
    axes[1].plot(epochs, loss, marker="o", color="#b26a2b")
    axes[1].set_ylabel("Object loss")
    axes[2].step(epochs, [int(item) for item in correct], where="mid", color="#3d7652")
    axes[2].set_yticks((0, 1), ("wrong", "correct"))
    axes[2].set_xlabel("Measured checkpoint epoch")
    for event in selected["forgetting_events"]:
        for axis in axes:
            axis.axvline(event, color="#9b3d2c", linewidth=2, alpha=0.7)
    figure.suptitle("Automatically selected measured forgetting case: case_real_001", fontsize=18, fontweight="bold")
    figure.tight_layout()
    figure.savefig(figure_dir / "01_measured_training_trace.png", dpi=150, facecolor="white")
    plt.close(figure)

    ablation = summary["rule_ablation"]
    with_rule = ablation["test_metrics_with_rule"]
    without_rule = ablation["test_metrics_without_rule"]
    names = ["accuracy", "recall", "subgroup_recall"]
    baseline = [float(with_rule[name]) for name in names]
    suppressed = [float(without_rule[name]) for name in names]
    x = range(len(names))
    figure, axis = plt.subplots(figsize=(16, 9))
    axis.bar([item - 0.2 for item in x], baseline, width=0.4, label="with native leaf rule", color="#275d73")
    axis.bar([item + 0.2 for item in x], suppressed, width=0.4, label="leaf suppressed", color="#b26a2b")
    axis.set_xticks(list(x), ["Test accuracy", "Test recall", "Test subgroup recall"])
    axis.set_ylim(0, 1.05)
    axis.legend()
    axis.set_title("Measured native tree-rule ablation", loc="left", fontsize=20, fontweight="bold")
    figure.tight_layout()
    figure.savefig(figure_dir / "02_measured_rule_ablation.png", dpi=150, facecolor="white")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(16, 9))
    axis.plot(
        [int(item["epoch"]) for item in checkpoints],
        [float(item["validation_metric"]) for item in checkpoints],
        label="validation accuracy",
        color="#275d73",
    )
    axis.plot(
        [int(item["epoch"]) for item in checkpoints],
        [float(item["subgroup_metric"]) for item in checkpoints],
        label="pre-defined subgroup accuracy",
        color="#9b3d2c",
    )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Measured accuracy")
    axis.legend()
    axis.set_title("Global and pre-defined subgroup trajectories", loc="left", fontsize=20, fontweight="bold")
    figure.tight_layout()
    figure.savefig(figure_dir / "03_subgroup_trajectory.png", dpi=150, facecolor="white")
    plt.close(figure)


def prepare_user_study() -> None:
    USER_STUDY.mkdir(parents=True, exist_ok=True)
    protocol = """# Independent A/B comprehension pilot

Status: `planned_not_run`.

Required participants: at least three ML developers and three external domain researchers.
Each participant sees technical-baseline and FuzzyXAI conditions in randomized AB/BA order for:

1. the measured forgetting experiment;
2. measured native-rule ablation;
3. the controlled research-only image channel.

No names, email addresses, diagnoses, or other unnecessary personal data are collected. Participant IDs
must be anonymous. The scorer must not be run on fabricated rows.
"""
    (USER_STUDY / "protocol.md").write_text(protocol, encoding="utf-8")
    (USER_STUDY / "participant_information.md").write_text(
        "# Participant information\n\nThis study evaluates explanation comprehension, not participant ability. Participation is voluntary and anonymous.\n",
        encoding="utf-8",
    )
    (USER_STUDY / "consent_template.md").write_text(
        "# Consent\n\nI consent to anonymous use of my task answers and timing for research evaluation. I may stop at any time.\n",
        encoding="utf-8",
    )
    source_template = ROOT / "release_evidence/explanation_experience/comprehension_pilot/response_template.csv"
    if source_template.exists():
        shutil.copy2(source_template, USER_STUDY / "response_template.csv")
    shutil.copy2(ROOT / "scripts/score_comprehension_pilot.py", USER_STUDY / "score_comprehension_pilot.py")
    write_json(
        USER_STUDY / "scoring_report.json",
        {
            "status": "planned_not_run",
            "participant_count": 0,
            "required_participants": 6,
            "claim_allowed": False,
            "blocker": "independent participants have not supplied anonymized responses",
        },
    )
    (USER_STUDY / "limitations.md").write_text(
        "# Limitations\n\nNo user-comprehension claim is allowed while status is `planned_not_run`.\n",
        encoding="utf-8",
    )


def main() -> None:
    subprocess.run(
        [sys.executable, str(ROOT / "experiments/real_training_experiment/run_empirical_validation.py")],
        cwd=ROOT,
        check=True,
    )
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    (OUTPUT / "json").mkdir(parents=True)
    (OUTPUT / "tables").mkdir(parents=True)
    CONTROLLED.mkdir(parents=True, exist_ok=True)
    write_json(
        CONTROLLED / "origin.json",
        {
            "scenario_id": "object_85_controlled_story_fixture",
            "fixture_type": "controlled_research_fixture",
            "empirical_status": "not_real_training_result",
            "source_type": "controlled",
            "result_origin": "controlled_fixture",
            "canonical_artifact": "release_evidence/explanation_experience/object_85_controlled_story_fixture.json",
        },
    )
    prepare_user_study()

    for path in sorted(EMPIRICAL.glob("*.json")):
        if path.name != "manifest_sha256.json":
            shutil.copy2(path, OUTPUT / "json" / path.name)
    shutil.copy2(EMPIRICAL / "dataset_card.md", OUTPUT / "dataset_card.md")
    summary = json.loads((EMPIRICAL / "empirical_summary.json").read_text(encoding="utf-8"))
    selected = json.loads((EMPIRICAL / "selected_forgetting_case.json").read_text(encoding="utf-8"))
    checkpoints = json.loads((EMPIRICAL / "checkpoints.json").read_text(encoding="utf-8"))
    cross_model = json.loads((EMPIRICAL / "cross_model_matrix.json").read_text(encoding="utf-8"))
    write_checkpoint_table(checkpoints, OUTPUT / "tables/checkpoint_metrics.csv")
    write_cross_model_table(cross_model, OUTPUT / "tables/cross_model_matrix.csv")
    write_json(OUTPUT / "tables/rule_ablation.json", summary["rule_ablation"])
    write_json(OUTPUT / "tables/similar_cases.json", json.loads((EMPIRICAL / "similar_cases.json").read_text(encoding="utf-8")))
    build_figures(summary, selected, checkpoints)

    status = {
        "schema_version": "1.0",
        "computed_gates": {
            "real_training": "PASS",
            "checkpoint_count": len(checkpoints),
            "forgetting_selection": "PASS",
            "predefined_subgroup": "PASS",
            "measured_rule_ablation": "PASS",
            "cross_model": "PASS",
            "chapter4_artifacts": "PASS",
        },
        "external_gates": {
            "domain_semantic_review": summary["domain_language_validation"]["status"],
            "comprehension_pilot": "planned_not_run",
        },
        "release_gate": "BLOCKED",
        "tag_allowed": False,
        "reason": "independent comprehension pilot and regulated-domain dictionary review are incomplete",
    }
    write_json(OUTPUT / "release_gate_status.json", status)
    report = f"""# Chapter 4 empirical validation evidence

## Measured result

- Dataset: {summary['dataset']['name']} ({summary['dataset']['objects']} objects, {summary['dataset']['features']} features).
- Checkpoints: {summary['checkpoints']} unique measured model states.
- Selected case: `{summary['selected_case']['public_id']}`; forgetting epochs: {summary['selected_case']['forgetting_events']}.
- Rare subgroup: `{summary['subgroup']['subgroup_id']}`, fixed before training and case selection.
- Native rule: `{summary['rule_ablation']['rule_id']}`; target prediction changes from {summary['rule_ablation']['target_prediction_with_rule']} to {summary['rule_ablation']['target_prediction_without_rule']} after measured suppression.
- Cross-model contracts: {summary['cross_model_count']}.

## Controlled boundary

`object_85_controlled_story_fixture` remains a contract/visualization fixture and is not a real training result.
The empirical case is `case_real_001`; it is not renamed to object 85.

## Release boundary

The computational empirical gate passes. The release tag remains blocked because the independent pilot is
`planned_not_run` and the regulated-domain dictionary is awaiting external review. No human-comprehension or
clinical-validity claim is permitted.
"""
    (OUTPUT / "report.md").write_text(report, encoding="utf-8")
    files = {
        str(path.relative_to(OUTPUT)): sha256(path)
        for path in sorted(OUTPUT.rglob("*"))
        if path.is_file() and path.name != "manifest_sha256.json"
    }
    write_json(
        OUTPUT / "manifest_sha256.json",
        {
            "schema_version": "1.0",
            "result_origin": "measured",
            "controlled_fixture_separated": True,
            "files": files,
        },
    )
    deterministic_zip(OUTPUT, ARCHIVE)
    ARCHIVE.with_suffix(".zip.sha256").write_text(f"{sha256(ARCHIVE)}  {ARCHIVE.name}\n", encoding="ascii")
    print(f"PASS: chapter4_empirical_checkpoints {len(checkpoints)}")
    print("PASS: chapter4_measured_ablation")
    print(f"PASS: chapter4_cross_model {len(cross_model)}")
    print("PASS: chapter4_figures 3/3")
    print(f"PASS: chapter4_empirical_archive {ARCHIVE}")
    print("BLOCKED: release_tag independent pilot and domain review incomplete")


if __name__ == "__main__":
    main()
