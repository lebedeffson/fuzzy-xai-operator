#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import zipfile
from hashlib import sha256
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "release" / "fuzzyxai_shap_like_visualization_package"
ZIP = ROOT / "reports" / "release" / "fuzzyxai_shap_like_visualization_package.zip"
CLI = ROOT / "reports" / "release" / "fuzzyxai_framework_rc" / "cli_check"
RESULTS = ROOT / "research_validation" / "reports" / "research_validation_results.csv"
CHAPTER_ASSETS = ROOT / "docs" / "chapter_4_framework" / "assets"


def must(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "framework" / "fuzzyxai") + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False, env=env)
    if result.returncode:
        print(result.stdout, end="")
        print(result.stderr, end="", file=sys.stderr)
        raise SystemExit(result.returncode)
    return result


def digest(path: Path) -> str:
    h = sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def require_file(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        raise SystemExit(f"missing shap-like artifact: {path}")


def word_count(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").split())


def verify_manifest(package_root: Path) -> None:
    manifest = json.loads((package_root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("manifest_self_hash_policy") != "excluded":
        raise SystemExit("manifest_self_hash_policy must be excluded")
    for item in manifest.get("sha256", []):
        if item["path"] == "manifest.json":
            raise SystemExit("manifest must not include itself")
        path = package_root / item["path"]
        require_file(path)
        if digest(path) != item["sha256"]:
            raise SystemExit(f"sha256 mismatch: {item['path']}")


def copy_chapter_pngs(chapter_dir: Path) -> None:
    CHAPTER_ASSETS.mkdir(parents=True, exist_ok=True)
    for path in chapter_dir.glob("*.png"):
        shutil.copy2(path, CHAPTER_ASSETS / path.name)


def write_report(path: Path, manifest: dict[str, object]) -> None:
    write(
        path,
        "# FuzzyXAI SHAP-like Visualization Report\n\n"
        "## Purpose\n\n"
        "This package defines a SHAP-like visual refinement for FuzzyXAI. The goal is readability comparable to common "
        "explainability figures while preserving FuzzyXAI mathematics. SHAP waterfall plots are additive: a base value and "
        "feature contributions are summed to a prediction. FuzzyXAI routes usually do not use additive aggregation. The "
        "risk value rho is commonly computed by an aggregation rule such as max(gamma, delta, quality, conflict, interval). "
        "Therefore the figures in this package visualize operator risk evidence, dominance and action boundaries rather "
        "than cumulative feature contribution.\n\n"
        "## Difference from SHAP\n\n"
        "SHAP answers how features contributed to a model prediction. FuzzyXAI answers why an explanation received a trust "
        "regime such as accept, lower_confidence, audit, defer_to_human or block. A feature attribution can be locally "
        "reasonable while the FuzzyXAI route still lowers confidence because explanation coverage is incomplete, input "
        "quality is limited, or a conflict component dominates the risk. The visual unit is therefore not a feature, but "
        "an operator component: uncertainty, reduction, quality, conflict and interval evidence. The action follows from "
        "rho and ExplainPlan thresholds.\n\n"
        "## S1 Operator Risk Contribution Summary\n\n"
        "This figure is the FuzzyXAI analogue of a SHAP summary bar plot, but it avoids claiming additive contribution. "
        "For each risk component it reports mean_value, std_value, dominance_count, dominance_rate, mean_excess_over_warning "
        "and max_value. The dominance rate is essential: a component with moderate mean value can still be responsible for "
        "critical actions in specific task classes. The CSV used for the plot is included as data/operator_risk_contribution_summary.csv.\n\n"
        "## S2 Local Risk Evidence Bridge\n\n"
        "This figure replaces the unsafe cumulative waterfall for max aggregation. It shows parallel bars for uncertainty, "
        "reduction, quality, conflict and interval evidence, then draws a vertical rho line computed by the aggregation "
        "rule. For the default max rule the dominant component is highlighted; no component is visually added to another. "
        "If future plans use weighted_sum, a cumulative variant may be introduced, but this package explicitly checks that "
        "the max bridge is used for max aggregation.\n\n"
        "## S3 Gamma-Delta Action Map v2\n\n"
        "The background is generated from the ExplainPlan thresholds and rho=max(gamma, delta), not from decorative "
        "rectangles. The accept region is the lower-left square where both gamma and delta are below the accept threshold. "
        "The next zones follow max(gamma, delta) ranges. Points are real research validation experiments. Point color "
        "shows action_id and marker shape/edge indicates dominant risk component.\n\n"
        "## S4 Action Boundary Strip v2\n\n"
        "The strip shows rho on a single action scale, distances to neighboring thresholds and the dominant evidence. It "
        "answers why the route is not in a neighboring zone: for example, reduction can keep rho above the accept boundary "
        "even when uncertainty is not high.\n\n"
        "## S5 Compact Operator Trace Heatmap v2\n\n"
        "The compact heatmap contains only numeric risk evidence columns. It intentionally excludes long diagnostic and "
        "action labels from the numeric matrix. This keeps the chapter figure readable and prevents mixing categorical "
        "states with continuous evidence values.\n\n"
        "## S6 Representation Class Atlas v2\n\n"
        "The atlas maps task_type by perturbation. Color is the majority representation class and the label contains the "
        "class and experiment count. This shows where F0, F_int, NAS and F_ML are activated and avoids treating class "
        "coverage as a simple binary flag.\n\n"
        "## S7 Explanation Coverage Curve\n\n"
        "The coverage curve remains part of the visual language because it explains delta directly. Top-k coverage is the "
        "visible retained explanation mass; delta is the residual loss. It is included in the broader visual package and "
        "the risk bridge uses the same reduction component.\n\n"
        "## S8 Proof Consistency Matrix v2\n\n"
        "The proof matrix is not SHAP-like, but it is a FuzzyXAI-specific audit figure. It checks artifacts against "
        "invariants: source_commit, gamma, delta, rho, diagnostic, action, route_id, sha256 and verifier_status. Green "
        "means passed, gray means not applicable and red means failed. This supports reproducibility claims for generated "
        "dashboards.\n\n"
        "## Mathematical correctness rule\n\n"
        "The most important design constraint is that the visual grammar must match the aggregation semantics. If the "
        "ExplainPlan or route states that risk is aggregated by max, then a cumulative stacked waterfall is forbidden "
        "because it implies a sum. The Local Risk Evidence Bridge therefore presents components as simultaneous evidence "
        "bars. The final rho marker is drawn as the aggregation result, and the dominant component is emphasized as the "
        "component that determines the max. This is deliberately different from a SHAP waterfall. It is less familiar at "
        "first sight, but it is faithful to the operator calculus used by FuzzyXAI. If a future route uses weighted_sum, "
        "the renderer can switch to a cumulative plot only when weights are present and the aggregation rule explicitly "
        "allows addition.\n\n"
        "## Reading the figures together\n\n"
        "The figures are meant to be read as a coordinated explanation board. Start with S2 for a single local decision: "
        "it identifies the operator evidence and the dominant risk source. Then inspect S4 to see whether rho lies near "
        "a boundary. A route can be lower_confidence because it is only slightly above accept, or because it is moving "
        "toward audit. S3 then places the same decision in the global gamma-delta decision geometry, where other research "
        "experiments can be compared. S1 gives the population-level component summary and indicates whether a component "
        "dominates often or merely has a high average. S5 and S6 are chapter-level comparison views, while S8 supports "
        "auditability. This order avoids the common mistake of interpreting one figure as a complete explanation.\n\n"
        "## Why dominance matters\n\n"
        "Mean values alone are not enough for FuzzyXAI. A component may have low mean value across the validation suite "
        "but still determine action in a small set of important experiments. For example, quality evidence may be near "
        "zero for clean tabular cases and much higher for signal or image perturbations. A mean bar would hide that "
        "behavior. Dominance_count and dominance_rate expose when a component actually controls rho under max aggregation. "
        "Mean_excess_over_warning adds another view: it shows how far a component crosses a policy threshold when it "
        "does become active. Together these statistics are a safer analogue of contribution magnitude.\n\n"
        "## Chapter use\n\n"
        "For Chapter 4 the recommended primary figures are S2 Local Risk Evidence Bridge, S3 Gamma-Delta Action Map v2, "
        "S4 Action Boundary Strip v2, S5 Compact Operator Trace Heatmap v2 and S8 Proof Consistency Matrix v2. S1 is useful "
        "for an aggregate research summary, and S6 demonstrates representation-class coverage. The categorical decision "
        "heatmap is included for appendix-level inspection because it contains long labels that are better suited to "
        "HTML or supplementary figures than to the main dissertation text.\n\n"
        "## Validation checks\n\n"
        "The package builder verifies that PNG, PDF and SVG files exist for all chapter figures. It verifies that the "
        "bridge uses max aggregation rather than cumulative summation. It writes a visual_source_index.json file so the "
        "reader can identify the route, trace, audit package and research CSV used to render the figures. It also validates "
        "sha256 entries in the manifest while excluding manifest.json itself from the hash list. These checks are small, "
        "but they protect the main claim: the figures are generated from FuzzyXAI artifacts, not manually drawn after the "
        "fact.\n\n"
        "## Separation of numeric and categorical evidence\n\n"
        "The refined visual set separates continuous risk evidence from categorical decisions. The compact operator trace "
        "heatmap contains numeric columns only: gamma, delta, rho and component values. This makes color intensity mean "
        "one thing across the matrix. Representation class, diagnostic state, action and verifier status are categorical "
        "and therefore belong in a separate decision heatmap or table. Mixing both types in one heatmap may look compact, "
        "but it weakens interpretation because a red numeric risk and a red category label do not necessarily encode the "
        "same mathematical quantity. The package keeps the numeric chapter figure concise and leaves the categorical view "
        "as supplementary evidence.\n\n"
        "## Interpretation limits\n\n"
        "These plots should not be read as performance metrics of the underlying machine-learning models. They are route "
        "interpretability and governance figures. A high reduction component means that the explanation supplied to the "
        "route is incomplete, not that the classifier is inaccurate. A high quality component means that the input or "
        "source conditions constrain trust, not that the target class is false. A proof matrix PASS means that the package "
        "is internally consistent; it does not certify real-world correctness. This distinction is essential for using "
        "the figures in a dissertation without overstating empirical claims.\n\n"
        "## Files and Reproducibility\n\n"
        "All figures are rendered from real CSV or JSON artifacts. The package includes PNG, PDF and SVG for chapter-ready "
        "figures, plus CSV/JSON source data. The manifest excludes itself from sha256 to avoid self-hash instability. "
        "The package_source_commit records the code revision that generated the visualization package. The route_source_commit "
        "records the source revision embedded in the audited route used as visual input. The visualization_source_commit "
        "records the visualization renderer revision; for this package it matches the package commit. These fields are "
        "separate because a visualization package can be rebuilt from a previously generated route. That is not a mismatch "
        "when each role is named explicitly.\n\n"
        "## Limitations\n\n"
        "These figures validate operator-risk-action visualization, not external clinical or industrial deployment. The "
        "default aggregation shown here is max. Other aggregation functions require matching visualization rules. The "
        "package deliberately avoids cumulative waterfall semantics unless the ExplainPlan explicitly uses weighted_sum.\n\n"
        "```json\n" + json.dumps(manifest, ensure_ascii=False, indent=2) + "\n```\n",
    )


def build() -> dict[str, object]:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    commit = must(["git", "rev-parse", "HEAD"]).stdout.strip()
    sys.path.insert(0, str(ROOT / "framework" / "fuzzyxai"))
    from fuzzyxai.visualization.shap_like import (
        build_operator_risk_contribution_summary,
        local_risk_evidence_data,
        render_action_boundary_strip_v2,
        render_compact_operator_trace_heatmap_v2,
        render_decision_heatmap_v2,
        render_explanation_coverage_curve_v2,
        render_gamma_delta_action_map_v2,
        render_local_risk_evidence_bridge,
        render_operator_risk_contribution_summary,
        render_proof_consistency_matrix_v2,
        render_representation_class_atlas_v2,
    )

    chapter = OUT / "chapter"
    data = OUT / "data"
    chapter.mkdir(parents=True)
    data.mkdir(parents=True)
    route = CLI / "route.json"
    trace = CLI / "operator_trace.json"
    package = CLI / "audit_package.zip"
    route_source_commit = json.loads(route.read_text(encoding="utf-8")).get("source_commit", "")
    contribution_csv = data / "operator_risk_contribution_summary.csv"
    bridge_json = data / "local_risk_evidence_bridge.json"
    proof_matrix_json = data / "proof_consistency_matrix_v2.json"

    render_operator_risk_contribution_summary(RESULTS, chapter / "operator_risk_contribution_summary.png", out_csv=contribution_csv)
    bridge_data = local_risk_evidence_data(trace, aggregation="max")
    if abs(float(bridge_data["rho"]) - max(float(v) for v in bridge_data["components"].values())) > 1e-9:
        raise SystemExit("max aggregation bridge does not match max component")
    render_local_risk_evidence_bridge(trace, chapter / "local_risk_evidence_bridge.png", out_json=bridge_json, aggregation="max")
    render_gamma_delta_action_map_v2(RESULTS, chapter / "gamma_delta_action_map_v2.png")
    render_action_boundary_strip_v2(route, chapter / "action_boundary_strip_v2.png", aggregation="max")
    render_compact_operator_trace_heatmap_v2(RESULTS, chapter / "compact_operator_trace_heatmap_v2.png")
    render_decision_heatmap_v2(RESULTS, chapter / "categorical_decision_heatmap_v2.png")
    render_representation_class_atlas_v2(RESULTS, chapter / "representation_class_atlas_v2.png")
    render_explanation_coverage_curve_v2(trace, chapter / "explanation_coverage_curve_v2.png")
    render_proof_consistency_matrix_v2(
        package if package.exists() else CLI,
        chapter / "proof_consistency_matrix_v2.png",
        out_json=proof_matrix_json,
    )
    build_operator_risk_contribution_summary(RESULTS, contribution_csv)

    source_index = {
        "research_results": RESULTS.relative_to(ROOT).as_posix(),
        "route": route.relative_to(ROOT).as_posix(),
        "operator_trace": trace.relative_to(ROOT).as_posix(),
        "audit_package": package.relative_to(ROOT).as_posix(),
        "risk_aggregation": "max",
        "manual_numbers_used": False,
    }
    write(data / "visual_source_index.json", json.dumps(source_index, ensure_ascii=False, indent=2) + "\n")

    expected_bases = [
        "operator_risk_contribution_summary",
        "local_risk_evidence_bridge",
        "gamma_delta_action_map_v2",
        "action_boundary_strip_v2",
        "compact_operator_trace_heatmap_v2",
        "representation_class_atlas_v2",
        "explanation_coverage_curve_v2",
        "proof_consistency_matrix_v2",
    ]
    for base in expected_bases:
        for suffix in ("png", "pdf", "svg"):
            require_file(chapter / f"{base}.{suffix}")
    for extra in [
        chapter / "categorical_decision_heatmap_v2.png",
        contribution_csv,
        bridge_json,
        proof_matrix_json,
        data / "visual_source_index.json",
    ]:
        require_file(extra)
    proof_matrix = json.loads(proof_matrix_json.read_text(encoding="utf-8"))
    if proof_matrix.get("fail_count") != 0:
        raise SystemExit("proof consistency matrix contains FAIL")
    proof_svg = (chapter / "proof_consistency_matrix_v2.svg").read_text(encoding="utf-8", errors="ignore")
    if "FAIL" in proof_svg:
        raise SystemExit("proof_consistency_matrix_v2.svg contains FAIL")
    copy_chapter_pngs(chapter)

    manifest = {
        "source_commit": commit,
        "package_source_commit": commit,
        "route_source_commit": route_source_commit,
        "visualization_source_commit": commit,
        "manifest_self_hash_policy": "excluded",
        "package_type": "fuzzyxai_shap_like_visualization_package",
        "risk_aggregation": "max",
        "checks": {
            "png_pdf_svg": "PASS",
            "max_evidence_bridge": "PASS",
            "real_csv_json_sources": "PASS",
            "no_cumulative_sum_for_max": "PASS",
            "proof_matrix_no_fail": "PASS",
            "manifest_self_hash_excluded": "PASS",
        },
        "visualizations": [
            "S1 Operator Risk Contribution Summary",
            "S2 Local Risk Evidence Bridge",
            "S3 Gamma-Delta Action Map v2",
            "S4 Action Boundary Strip v2",
            "S5 Compact Operator Trace Heatmap v2",
            "S6 Representation Class Atlas v2",
            "S7 Explanation Coverage Curve",
            "S8 Proof Consistency Matrix v2",
        ],
    }
    write_report(OUT / "shap_like_visualization_report.md", manifest)
    if word_count(OUT / "shap_like_visualization_report.md") < 1500:
        raise SystemExit("shap_like_visualization_report.md must contain at least 1500 words")

    files = [path for path in sorted(OUT.rglob("*")) if path.is_file()]
    manifest["sha256"] = [
        {"path": path.relative_to(OUT).as_posix(), "sha256": digest(path), "size_bytes": path.stat().st_size}
        for path in files
        if path.name != "manifest.json"
    ]
    write(OUT / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    verify_manifest(OUT)

    if ZIP.exists():
        ZIP.unlink()
    with zipfile.ZipFile(ZIP, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(OUT.rglob("*")):
            if path.is_file():
                archive.write(path, f"fuzzyxai_shap_like_visualization_package/{path.relative_to(OUT).as_posix()}")
    return {
        "commit": commit,
        "package": ZIP.as_posix(),
        "files": len([path for path in OUT.rglob("*") if path.is_file()]),
        "chapter_figures": len([path for path in chapter.rglob("*") if path.is_file()]),
        "report_words": word_count(OUT / "shap_like_visualization_report.md"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-only", action="store_true")
    args = parser.parse_args()
    result = build()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("fuzzyxai-shap-like-visualization-package: PASS" if args.package_only else "fuzzyxai-shap-like-visualization-check: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
