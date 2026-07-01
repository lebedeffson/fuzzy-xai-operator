from __future__ import annotations

import json
from pathlib import Path

from .action_boundary import render_action_boundary
from .coverage_curve import render_coverage_curve
from .gamma_delta_map import render_gamma_delta_action_map
from .proof_matrix import render_proof_consistency_matrix
from .representation_atlas import render_representation_atlas
from .risk_waterfall import render_risk_waterfall
from .route_sankey import render_route_sankey
from .trace_heatmap import render_operator_trace_heatmap
from .utils import ensure_parent, read_json, write_html_with_image


def render_operator_dashboard_v3(
    route: str | Path,
    trace: str | Path,
    verifier: str | Path,
    package: str | Path,
    out_png: str | Path,
    out_html: str | Path | None = None,
) -> Path:
    out_png = ensure_parent(out_png)
    work = out_png.parent / "_dashboard_v3_parts"
    work.mkdir(parents=True, exist_ok=True)
    sankey = render_route_sankey(route, work / "route_sankey.png")
    waterfall = render_risk_waterfall(trace, work / "risk_waterfall.png")
    boundary = render_action_boundary(route, work / "action_boundary.png")
    proof = render_proof_consistency_matrix(package, work / "proof_matrix.png")
    data = read_json(route)
    verifier_data = read_json(verifier)
    computed = data.get("computed_result", {})
    try:
        import matplotlib.image as mpimg
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required") from exc
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    for ax, image, title in [
        (axes[0, 0], sankey, "Operator Route Sankey"),
        (axes[0, 1], waterfall, "Risk Waterfall"),
        (axes[1, 0], boundary, "Action Boundary"),
        (axes[1, 1], proof, "Proof Consistency"),
    ]:
        ax.imshow(mpimg.imread(image))
        ax.set_title(title, weight="bold")
        ax.axis("off")
    fig.suptitle(
        f"FuzzyXAI Dashboard v3: {computed.get('model_name')} / action={computed.get('action_id')} / verifier={verifier_data.get('overall_status')}",
        fontsize=16,
        weight="bold",
    )
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    if out_html:
        write_html_with_image(out_png, out_html, title="FuzzyXAI Operator Dashboard v3", summary=computed)
    return out_png


def render_research_visual_dashboard(
    results_csv: str | Path,
    out_html: str | Path,
    out_png: str | Path | None = None,
) -> Path:
    out_html = ensure_parent(out_html)
    work = out_html.parent / "_research_visual_parts"
    work.mkdir(parents=True, exist_ok=True)
    gamma_delta = render_gamma_delta_action_map(results_csv, work / "gamma_delta_action_map.png")
    heatmap = render_operator_trace_heatmap(results_csv, work / "operator_trace_heatmap.png")
    atlas = render_representation_atlas(results_csv, work / "representation_atlas.png")
    if out_png:
        try:
            import matplotlib.image as mpimg
            import matplotlib.pyplot as plt
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("matplotlib is required") from exc
        out_png = ensure_parent(out_png)
        fig, axes = plt.subplots(1, 3, figsize=(21, 7))
        for ax, image, title in [
            (axes[0], gamma_delta, "Gamma-Delta Action Map"),
            (axes[1], heatmap, "Operator Trace Heatmap"),
            (axes[2], atlas, "Representation Atlas"),
        ]:
            ax.imshow(mpimg.imread(image))
            ax.set_title(title, weight="bold")
            ax.axis("off")
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
    images = []
    for title, image in [
        ("Gamma-Delta Action Map", gamma_delta),
        ("Operator Trace Heatmap", heatmap),
        ("Representation Atlas", atlas),
    ]:
        images.append(f"<h2>{title}</h2><img src='_research_visual_parts/{Path(image).name}' alt='{title}'>")
    out_html.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>FuzzyXAI Research Visual Dashboard</title>"
        "<style>body{font-family:Arial,sans-serif;margin:28px}img{max-width:100%;border:1px solid #ddd;margin-bottom:24px}</style>"
        "</head><body><h1>FuzzyXAI Research Visual Dashboard</h1>"
        + "\n".join(images)
        + "</body></html>",
        encoding="utf-8",
    )
    return out_html


def render_single_case_visuals(base_dir: str | Path, out_dir: str | Path) -> dict[str, Path]:
    base = Path(base_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    route = base / "route.json"
    trace = base / "operator_trace.json"
    verifier = base / "verifier_report.json"
    package = base / "audit_package.zip"
    return {
        "operator_route_sankey": render_route_sankey(route, out / "operator_route_sankey.png", out / "operator_route_sankey.html"),
        "risk_waterfall": render_risk_waterfall(trace, out / "risk_waterfall.png", out / "risk_waterfall.html"),
        "explanation_coverage_curve": render_coverage_curve(trace, out / "explanation_coverage_curve.png", out / "explanation_coverage_curve.html"),
        "action_boundary_plot": render_action_boundary(route, out / "action_boundary_plot.png", out / "action_boundary_plot.html"),
        "proof_consistency_matrix": render_proof_consistency_matrix(package if package.exists() else base, out / "proof_consistency_matrix.png", out / "proof_consistency_matrix.html"),
        "operator_dashboard_v3": render_operator_dashboard_v3(route, trace, verifier, package if package.exists() else base, out / "operator_dashboard_v3.png", out / "operator_dashboard_v3.html"),
    }


def render_research_visuals(results_csv: str | Path, out_dir: str | Path) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    return {
        "gamma_delta_action_map": render_gamma_delta_action_map(results_csv, out / "gamma_delta_action_map.png", out / "gamma_delta_action_map.html"),
        "operator_trace_heatmap": render_operator_trace_heatmap(results_csv, out / "operator_trace_heatmap.png", out / "operator_trace_heatmap.html"),
        "representation_atlas": render_representation_atlas(results_csv, out / "representation_atlas.png", out / "representation_atlas.html"),
        "research_visual_dashboard": render_research_visual_dashboard(results_csv, out / "research_visual_dashboard.html", out / "research_visual_dashboard.png"),
    }


def write_visualization_report(out_path: str | Path, manifest: dict[str, object]) -> Path:
    out_path = ensure_parent(out_path)
    out_path.write_text(
        "# FuzzyXAI Visualization Report\n\n"
        "FuzzyXAI visual analytics shows the operator-risk-action trace rather than feature contribution alone.\n\n"
        "Canonical visualizations:\n\n"
        "1. Operator Route Sankey\n"
        "2. Gamma-Delta Action Map\n"
        "3. Risk Waterfall\n"
        "4. Operator Trace Heatmap\n"
        "5. Representation Class Atlas\n"
        "6. Explanation Coverage Curve\n"
        "7. Action Boundary Plot\n"
        "8. Proof Consistency Matrix\n\n"
        "All PNG and HTML artifacts are rendered from route, operator trace, proof/verifier data, or research CSV files.\n\n"
        "```json\n" + json.dumps(manifest, ensure_ascii=False, indent=2) + "\n```\n",
        encoding="utf-8",
    )
    return out_path
