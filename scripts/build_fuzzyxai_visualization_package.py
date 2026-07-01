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
OUT = ROOT / "reports" / "release" / "fuzzyxai_visualization_package"
ZIP = ROOT / "reports" / "release" / "fuzzyxai_visualization_package.zip"
CLI = ROOT / "reports" / "release" / "fuzzyxai_framework_rc" / "cli_check"
RESULTS = ROOT / "research_validation" / "reports" / "research_validation_results.csv"
CHAPTER_ASSETS = ROOT / "docs" / "chapter_4_framework" / "assets"
CHAPTER_FIGURES_DOC = ROOT / "docs" / "chapter_4_framework" / "visualization_figures.md"


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
        raise SystemExit(f"required visualization artifact missing: {path}")


def word_count(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").split())


def verify_manifest(package_root: Path) -> None:
    manifest_path = package_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for item in manifest.get("sha256", []):
        rel = item["path"]
        if rel == "manifest.json":
            raise SystemExit("manifest must not include itself in sha256")
        path = package_root / rel
        require_file(path)
        if digest(path) != item["sha256"]:
            raise SystemExit(f"sha256 mismatch: {rel}")


def copy_chapter_assets(chapter_dir: Path) -> None:
    CHAPTER_ASSETS.mkdir(parents=True, exist_ok=True)
    for path in chapter_dir.glob("*.png"):
        shutil.copy2(path, CHAPTER_ASSETS / path.name)


def write_chapter_figures_doc() -> None:
    write(
        CHAPTER_FIGURES_DOC,
        "# Visualization Figures for Chapter 4\n\n"
        "This note fixes the chapter-ready visual set for the FuzzyXAI visual analytics layer. "
        "The figures visualize the operator-risk-action trace rather than feature contribution alone.\n\n"
        "## Captions\n\n"
        "- Figure 4.x -- Gamma-Delta Action Map. Shows experiments in the gamma/delta risk space with action zones.\n"
        "- Figure 4.x -- Risk Waterfall. Shows uncertainty, reduction, quality, conflict and interval components leading to rho.\n"
        "- Figure 4.x -- Action Boundary Plot. Shows rho relative to accept, lower_confidence and audit thresholds.\n"
        "- Figure 4.x -- Operator Trace Heatmap Compact. Compares experiments by gamma, delta, rho and dominant components.\n"
        "- Figure 4.x -- Representation Class Atlas. Shows activation of F0, F_int, NAS and F_ML by task and perturbation.\n"
        "- Figure 4.x -- Proof Consistency Matrix. Shows consistency of route, trace, proof, dashboard data, verifier and manifest.\n\n"
        "## Files\n\n"
        "- `assets/gamma_delta_action_map_chapter.png`\n"
        "- `assets/risk_waterfall_chapter.png`\n"
        "- `assets/action_boundary_chapter.png`\n"
        "- `assets/operator_trace_heatmap_compact.png`\n"
        "- `assets/representation_atlas_chapter.png`\n"
        "- `assets/proof_consistency_matrix_chapter.png`\n",
    )


def check_chapter_pngs(paths: list[Path]) -> None:
    from fuzzyxai.visualization.utils import png_size

    for path in paths:
        width, _ = png_size(path)
        if width < 1200:
            raise SystemExit(f"chapter PNG width below 1200: {path} ({width})")


def build() -> dict[str, object]:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    commit = must(["git", "rev-parse", "HEAD"]).stdout.strip()
    sys.path.insert(0, str(ROOT / "framework" / "fuzzyxai"))
    from fuzzyxai.visualization.export import render_chapter_visuals, render_research_visuals, render_single_case_visuals, write_visualization_report

    single = OUT / "single_case"
    research = OUT / "research"
    chapter = OUT / "chapter"
    render_single_case_visuals(CLI, single)
    render_research_visuals(RESULTS, research)
    chapter_outputs = render_chapter_visuals(CLI, RESULTS, chapter)
    cli_out = single / "cli_route_sankey.png"
    must([
        sys.executable,
        "-m",
        "fuzzyxai.cli",
        "visualize",
        "route-sankey",
        "--route",
        str(CLI / "route.json"),
        "--out",
        str(cli_out),
        "--html-out",
        str(single / "cli_route_sankey.html"),
    ])

    expected = [
        single / "operator_route_sankey.png",
        single / "operator_route_sankey.html",
        single / "risk_waterfall.png",
        single / "risk_waterfall.html",
        single / "explanation_coverage_curve.png",
        single / "explanation_coverage_curve.html",
        single / "action_boundary_plot.png",
        single / "action_boundary_plot.html",
        single / "proof_consistency_matrix.png",
        single / "proof_consistency_matrix.html",
        single / "operator_dashboard_v3.png",
        single / "operator_dashboard_v3.html",
        single / "cli_route_sankey.png",
        single / "cli_route_sankey.html",
        research / "gamma_delta_action_map.png",
        research / "gamma_delta_action_map.html",
        research / "operator_trace_heatmap.png",
        research / "operator_trace_heatmap.html",
        research / "representation_atlas.png",
        research / "representation_atlas.html",
        research / "research_visual_dashboard.html",
        research / "research_visual_dashboard.png",
        chapter / "gamma_delta_action_map_chapter.png",
        chapter / "risk_waterfall_chapter.png",
        chapter / "action_boundary_chapter.png",
        chapter / "operator_trace_heatmap_compact.png",
        chapter / "representation_atlas_chapter.png",
        chapter / "proof_consistency_matrix_chapter.png",
    ]
    for path in expected:
        require_file(path)
    check_chapter_pngs(list(chapter_outputs.values()))

    manifest = {
        "source_commit": commit,
        "package_type": "fuzzyxai_visualization_package",
        "visual_idea": "operator-risk-action trace",
        "checks": {
            "single_case_visuals": "PASS",
            "research_visuals": "PASS",
            "png_created": "PASS",
            "html_created": "PASS",
            "cli_visualize": "PASS",
            "chapter_figures": "PASS",
            "visual_quality": "PASS",
        },
        "canonical_visualizations": [
            "Operator Route Flow",
            "Gamma-Delta Action Map",
            "Risk Waterfall",
            "Operator Trace Heatmap",
            "Representation Class Atlas",
            "Explanation Coverage Curve",
            "Action Boundary Plot",
            "Proof Consistency Matrix",
        ],
    }
    write_visualization_report(OUT / "visualization_report.md", manifest)
    if word_count(OUT / "visualization_report.md") < 1200:
        raise SystemExit("visualization_report.md must contain at least 1200 words")
    copy_chapter_assets(chapter)
    write_chapter_figures_doc()
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
                archive.write(path, f"fuzzyxai_visualization_package/{path.relative_to(OUT).as_posix()}")
    return {
        "commit": commit,
        "package": ZIP.as_posix(),
        "files": len([p for p in OUT.rglob("*") if p.is_file()]),
        "single_case": len([p for p in single.rglob("*") if p.is_file()]),
        "research": len([p for p in research.rglob("*") if p.is_file()]),
        "chapter": len([p for p in chapter.rglob("*") if p.is_file()]),
        "visual_report_words": word_count(OUT / "visualization_report.md"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-only", action="store_true")
    args = parser.parse_args()
    result = build()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("fuzzyxai-visualization-package: PASS" if args.package_only else "fuzzyxai-visualization-check: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
