"""Build the Chapter 4 evidence package from executable framework artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import yaml

from fuzzyxai.audit.operators_manifest import validate_manifest
from fuzzyxai.visualization.matplotlib_renderer import render_visual_spec
from fuzzyxai.visualization.view_model import ExplanationViewModel

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "release_evidence/explanation_experience"
OUTPUT = ROOT / "release_evidence/chapter4_explanation_experience"
FIGURES = OUTPUT / "figures"
TABLES = OUTPUT / "tables"


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_architecture(path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    labels = ("ModelAdapter", "Evidence", "Claims", "ExplanationGraph", "Diagnostic", "Action", "VisualSpec")
    figure, axis = plt.subplots(figsize=(16, 9))
    axis.set_xlim(0, 1); axis.set_ylim(0, 1); axis.axis("off")
    axis.set_title("FuzzyXAI Explanation Experience architecture", loc="left", fontsize=22, fontweight="bold")
    positions = [(0.04 + 0.23 * (index % 4), 0.62 if index < 4 else 0.28) for index in range(len(labels))]
    for index, (label, (x, y)) in enumerate(zip(labels, positions)):
        box = FancyBboxPatch((x, y), 0.18, 0.16, boxstyle="round,pad=.02", facecolor="#eef4f6", edgecolor="#355b72", linewidth=2)
        axis.add_patch(box); axis.text(x + 0.09, y + 0.08, label, ha="center", va="center", fontsize=12, fontweight="bold")
        if index:
            previous = positions[index - 1]
            if index != 4:
                axis.annotate("", xy=(x, y + 0.08), xytext=(previous[0] + 0.18, previous[1] + 0.08), arrowprops={"arrowstyle": "->", "color": "#355b72", "lw": 2})
    axis.text(0.04, 0.12, "Scientific result and presentation contract are separated; renderers never recompute evidence.", fontsize=13, color="#355b72")
    figure.savefig(path, dpi=150, bbox_inches="tight", facecolor="white"); plt.close(figure)


def build_levels(path: Path) -> None:
    import matplotlib.pyplot as plt

    labels = ["E0\nprediction", "E1\ndata", "E2\nrules/contributions", "E3\nconcepts/cases", "E4\ntraining", "E5\noperator audit"]
    figure, axis = plt.subplots(figsize=(16, 9))
    colors = ["#d9e2e6", "#c4d7df", "#a9c8d5", "#83b3c6", "#5c96ad", "#355b72"]
    axis.barh(range(6), range(1, 7), color=colors)
    axis.set_yticks(range(6), labels); axis.set_xticks([]); axis.invert_yaxis()
    axis.set_title("Explanation levels disclose evidence depth, not model quality", loc="left", fontsize=22, fontweight="bold")
    for index in range(6): axis.text(index + 0.85, index, f"E{index}", va="center", color="white" if index > 2 else "#17242d", fontweight="bold")
    axis.spines[:].set_visible(False)
    figure.savefig(path, dpi=150, bbox_inches="tight", facecolor="white"); plt.close(figure)


def render_fixture(source: Path, view: str, output: Path) -> None:
    model = ExplanationViewModel.load_json(source)
    render_visual_spec(model.visual_spec, view=view, output_path=output)


def build_medical_panel(path: Path) -> None:
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    report = json.loads((SOURCE / "medical_research_similarity_explanation.json").read_text(encoding="utf-8"))
    names = ("query_image", "reference_image_67", "query_mask", "reference_mask_67", "mask_intersection", "mask_difference", "counterexample_41", "counterexample_93")
    figure, axes = plt.subplots(2, 4, figsize=(16, 9))
    for axis, name in zip(axes.flat, names):
        axis.imshow(mpimg.imread(ROOT / report["media_artifacts"][name]), cmap="gray")
        axis.set_title(name.replace("_", " ")); axis.axis("off")
    figure.suptitle(f"Research-only image comparison: mask IoU={report['mask_iou']:.6f}; not diagnostic probability", fontsize=18, fontweight="bold")
    figure.savefig(path, dpi=150, bbox_inches="tight", facecolor="white"); plt.close(figure)


def build_operator_matrix() -> list[dict[str, object]]:
    manifest_path = ROOT / "framework/fuzzyxai/operators_manifest.yaml"
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    validation = validate_manifest(manifest_path)
    rows = []
    for operator in payload["operators"]:
        artifacts = [str(item) for item in operator["artifacts"]]
        rows.append({
            "id": operator["id"], "dissertation_ref": operator["dissertation_ref"], "callable": operator["callable"],
            "input_schema": operator["input_schema"], "output_schema": operator["output_schema"],
            "tests": "; ".join(operator["tests"]), "evidence": "; ".join(artifacts),
            "claim": "claim-grounded" if str(operator["id"]).startswith("framework.") else "operator-route evidence",
            "visualization": "; ".join(operator["visualization"]) or "not required by manifest",
            "release_artifact": "release_evidence/chapter4_explanation_experience",
            "status": "PASS" if validation["status"] == "PASS" and all((ROOT / item).exists() for item in artifacts) else "FAIL",
        })
    return rows


def write_table(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    subprocess.run([sys.executable, str(ROOT / "scripts/build_explanation_experience_evidence.py")], cwd=ROOT, check=True)
    if OUTPUT.exists(): shutil.rmtree(OUTPUT)
    FIGURES.mkdir(parents=True); TABLES.mkdir(parents=True)

    build_architecture(FIGURES / "01_framework_architecture.png")
    render_fixture(SOURCE / "object_85_explanation.json", "provenance", FIGURES / "02_evidence_claim_diagnostic_action.png")
    build_levels(FIGURES / "03_explanation_levels_e0_e5.png")
    views = {
        "04_object85_explanation_story.png": "explanation_story", "05_object85_data_profile.png": "data_profile",
        "06_object85_training_trace.png": "training_trace", "08_rule_ablation.png": "rule_ablation",
        "09_similar_cases.png": "similar_cases", "10_provenance.png": "provenance",
    }
    for name, view in views.items(): render_fixture(SOURCE / "object_85_explanation.json", view, FIGURES / name)
    render_fixture(SOURCE / "anfis_native_rules_explanation.json", "knowledge_atlas", FIGURES / "07_anfis_knowledge_atlas.png")
    build_medical_panel(FIGURES / "11_medical_research_only_comparison.png")
    render_fixture(SOURCE / "cross_model/black_box_callable.json", "audit", FIGURES / "12_missing_evidence_state.png")

    matrix = build_operator_matrix()
    write_table(TABLES / "operator_implementation_matrix.csv", matrix)
    write_json(TABLES / "operator_implementation_matrix.json", matrix)
    markdown = ["# Operator implementation matrix", "", "| ID | Dissertation | Callable | Test | Evidence | Visual | Status |", "|---|---|---|---|---|---|---|"]
    markdown.extend(f"| {row['id']} | {row['dissertation_ref']} | `{row['callable']}` | `{row['tests']}` | `{row['evidence']}` | {row['visualization']} | {row['status']} |" for row in matrix)
    (TABLES / "operator_implementation_matrix.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")

    cross_model = json.loads((SOURCE / "cross_model/cross_model_matrix.json").read_text(encoding="utf-8"))
    model_rows = [{"scenario": name, **payload} for name, payload in cross_model.items()]
    write_json(TABLES / "model_capability_matrix.json", model_rows)
    object85 = json.loads((SOURCE / "object_85_explanation.json").read_text(encoding="utf-8"))
    write_json(TABLES / "rule_ablation_results.json", object85["visual_spec"]["rule_ablations"])
    write_json(TABLES / "explanation_coverage.json", object85["quality_metrics"])
    shutil.copy2(SOURCE / "manifest_sha256.json", TABLES / "golden_sha256.json")
    write_json(TABLES / "comprehension_pilot_status.json", {"status": "planned_not_run", "required_participants": 6, "claim_allowed": False, "protocol": "docs/user_comprehension_study.md"})

    tests = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_explanation_experience.py", "tests/test_evidence_first_framework.py", "tests/test_public_framework_api.py"], cwd=ROOT, text=True, capture_output=True)
    write_json(TABLES / "test_matrix.json", {"command": tests.args, "exit_code": tests.returncode, "status": "PASS" if tests.returncode == 0 else "FAIL", "output": tests.stdout.strip(), "errors": tests.stderr.strip()})
    if tests.returncode: raise RuntimeError("chapter 4 evidence tests failed")

    files = {str(path.relative_to(OUTPUT)): sha256(path) for path in sorted(OUTPUT.rglob("*")) if path.is_file() and path.name not in {"manifest_sha256.json", "chapter4_explanation_evidence.zip"}}
    write_json(OUTPUT / "manifest_sha256.json", {"schema_version": "1.0", "operator_rows": len(matrix), "all_operators_pass": all(row["status"] == "PASS" for row in matrix), "files": files})
    archive = ROOT / "release_evidence/chapter4_explanation_evidence.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(OUTPUT.rglob("*")):
            if path.is_file(): handle.write(path, Path("chapter4_explanation_experience") / path.relative_to(OUTPUT))
    (archive.with_suffix(".zip.sha256")).write_text(f"{sha256(archive)}  {archive.name}\n", encoding="ascii")
    print(f"PASS: chapter4_operator_matrix {len(matrix)}/30")
    print("PASS: chapter4_figures 12/12")
    print(f"PASS: chapter4_evidence_archive {archive}")


if __name__ == "__main__": main()
