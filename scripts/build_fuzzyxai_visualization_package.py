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


def build() -> dict[str, object]:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    commit = must(["git", "rev-parse", "HEAD"]).stdout.strip()
    sys.path.insert(0, str(ROOT / "framework" / "fuzzyxai"))
    from fuzzyxai.visualization.export import render_research_visuals, render_single_case_visuals, write_visualization_report

    single = OUT / "single_case"
    research = OUT / "research"
    render_single_case_visuals(CLI, single)
    render_research_visuals(RESULTS, research)
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
    ]
    for path in expected:
        require_file(path)

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
        },
        "canonical_visualizations": [
            "Operator Route Sankey",
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
    files = [path for path in sorted(OUT.rglob("*")) if path.is_file()]
    manifest["sha256"] = [
        {"path": path.relative_to(OUT).as_posix(), "sha256": digest(path), "size_bytes": path.stat().st_size}
        for path in files
        if path.name != "manifest.json"
    ]
    write(OUT / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")

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
