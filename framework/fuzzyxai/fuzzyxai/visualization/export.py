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
        (axes[0, 0], sankey, "Operator Route Flow"),
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


def render_chapter_visuals(base_dir: str | Path, results_csv: str | Path, out_dir: str | Path) -> dict[str, Path]:
    base = Path(base_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    route = base / "route.json"
    trace = base / "operator_trace.json"
    package = base / "audit_package.zip"
    package_source = package if package.exists() else base
    return {
        "gamma_delta_action_map_chapter": render_gamma_delta_action_map(results_csv, out / "gamma_delta_action_map_chapter.png"),
        "risk_waterfall_chapter": render_risk_waterfall(trace, out / "risk_waterfall_chapter.png"),
        "action_boundary_chapter": render_action_boundary(route, out / "action_boundary_chapter.png"),
        "operator_trace_heatmap_compact": render_operator_trace_heatmap(results_csv, out / "operator_trace_heatmap_compact.png", compact=True),
        "representation_atlas_chapter": render_representation_atlas(results_csv, out / "representation_atlas_chapter.png"),
        "proof_consistency_matrix_chapter": render_proof_consistency_matrix(package_source, out / "proof_consistency_matrix_chapter.png"),
    }


def write_visualization_report(out_path: str | Path, manifest: dict[str, object]) -> Path:
    out_path = ensure_parent(out_path)
    out_path.write_text(
        "# FuzzyXAI Visualization Report\n\n"
        "## Goal\n\n"
        "The FuzzyXAI visual analytics layer is designed to make an operator route inspectable as a research object. "
        "Its central unit is not a feature contribution, but a trace of how trust in an explanation is transformed by "
        "operators. The visual layer follows the chain input artifact, explanation object, representation class, "
        "alignment gamma, reduction delta, risk rho, diagnostic state, action and proof. This is why the figures are "
        "built from route JSON, operator trace JSON, verifier reports, proof artifacts and research validation CSV files. "
        "The dashboard does not recompute gamma, delta or rho; it displays values already fixed by the framework trace.\n\n"
        "## Difference from SHAP-style figures\n\n"
        "SHAP-style visualizations answer a local prediction question: which features move a prediction upward or downward. "
        "FuzzyXAI answers a different question: why did the framework assign a specific trust regime to an explanation. "
        "A model can have useful feature attributions and still require lower confidence if the explanation is incomplete, "
        "the input quality is limited, or the route crosses an audit boundary. The visual layer therefore emphasizes "
        "operator state, risk components, decision boundaries, diagnostic semantics and proof consistency. The intent is "
        "not to replace feature attribution, but to show how feature attribution becomes part of a controlled explanatory "
        "route.\n\n"
        "## Semantic visual language\n\n"
        "All chapter-ready figures use the same semantic palette. Accept is green, lower_confidence is yellow, audit is "
        "orange, defer_to_human is purple and block or critical states are red. The same colors are reused in action maps, "
        "boundary plots, waterfall summaries and proof matrices. Gamma is treated as disagreement or uncertainty, delta "
        "as explanation reduction loss and rho as the final risk. Every chapter figure contains provenance markers: "
        "source_commit, route_id when available and verifier status. This makes the figures usable as audit artifacts, "
        "not just illustrations.\n\n"
        "## Canonical visualizations\n\n"
        "1. Operator Route Flow shows the ordered operator route. Nodes are operators, edges are transmitted values, and "
        "edge width follows numeric risk or diagnostic values when they are available. It is named as a route flow because "
        "the current implementation is a Sankey-compatible alluvial diagram rather than a mass-conserving Sankey model.\n\n"
        "2. Gamma-Delta Action Map shows the geometry of decision making. The x-axis is gamma, the y-axis is delta, and "
        "background regions encode action zones. Experiments become points in the operator risk space. This figure is the "
        "main visual argument for FuzzyXAI: the action is selected by location in a controlled risk space, not by a manual "
        "label.\n\n"
        "3. Risk Waterfall shows how uncertainty, reduction, quality, conflict and interval components contribute to the "
        "final rho value. It is the closest FuzzyXAI analogue to a waterfall plot, but its output is trust degradation and "
        "action selection rather than a raw prediction.\n\n"
        "4. Operator Trace Heatmap compares experiments across operator components. The full version keeps every experiment "
        "and component for appendices. The compact chapter version keeps gamma, delta, rho, uncertainty, reduction, quality "
        "and action severity so it can be read in a dissertation page.\n\n"
        "5. Representation Class Atlas shows where F0, F_int, NAS and F_ML are activated over task type and perturbation. "
        "It demonstrates that representation classes are not decorative labels: interval uncertainty activates F_int, "
        "source or quality conflicts activate NAS, and multilevel signal or image explanations activate F_ML.\n\n"
        "6. Explanation Coverage Curve explains delta. It plots accumulated top-k explanation coverage and the remaining "
        "loss delta equals one minus coverage. This prevents delta from appearing as an unexplained scalar.\n\n"
        "7. Action Boundary Plot shows the current rho value relative to accept, lower_confidence and audit boundaries. "
        "It explains how close a decision is to a neighboring trust regime and why a small change in risk may alter action.\n\n"
        "8. Proof Consistency Matrix checks whether route, operator trace, proof trace, dashboard data, verifier report and "
        "manifest agree on source commit, gamma, delta, rho, diagnostic, action and sha256 evidence. This figure supports "
        "the claim that dashboard artifacts are generated from the trace rather than manually drawn.\n\n"
        "## Data sources\n\n"
        "Single-case figures are rendered from route.json, operator_trace.json, verifier_report.json, dashboard_data.json "
        "and the audit package manifest. Research figures are rendered from research_validation_results.csv and related "
        "summary files. The chapter figures in this package are derived from the same artifacts and are copied into "
        "docs/chapter_4_framework/assets for dissertation use.\n\n"
        "The Operator Route Flow consumes route nodes and edges. Its node labels come from operator metadata and its edge "
        "labels come from passed_values. The Risk Waterfall consumes computed_result from the operator trace, using the "
        "same component names that the verifier checks. The Gamma-Delta Action Map consumes the research validation table, "
        "so every point corresponds to a completed experiment, not a manually placed visual sample. The Proof Consistency "
        "Matrix consumes the audit package directly and reads route, operator trace, proof trace, verifier report, "
        "dashboard data and manifest entries. These data-source choices are deliberate: each figure remains tied to a "
        "machine-checkable artifact.\n\n"
        "## Interpretation guide\n\n"
        "Read the figures as a sequence. First inspect the route flow to identify the operator path and final action. Then "
        "use the risk waterfall to find the dominant risk component. Use the action boundary plot to see whether the result "
        "is near accept or audit. For research comparison, use the gamma-delta map and compact heatmap to locate all "
        "experiments in the risk space. Finally, use the representation atlas and proof matrix to check whether the chosen "
        "representation and proof evidence are consistent with the route.\n\n"
        "When interpreting chapter figures, green regions should be read as full acceptance zones, yellow as limited trust, "
        "orange as audit pressure, purple as human deferral and red as critical blocking. A point in a yellow region does "
        "not mean that the model is wrong; it means that the explanation route contains enough uncertainty or reduction "
        "loss that automatic trust should be limited. Likewise, a high delta is not a model-quality measure by itself. It "
        "is a statement about the completeness of the explanation delivered to the route. This distinction is important "
        "because FuzzyXAI separates model prediction, explanation coverage, data quality and final governance action.\n\n"
        "For a single decision, the recommended reading order is route flow, waterfall, boundary and proof matrix. For "
        "research validation, the recommended order is gamma-delta map, compact heatmap, representation atlas and then "
        "diagnostic or action distributions. This keeps the reader from treating the figures as independent plots. They "
        "are a coordinated visual grammar for one operator framework.\n\n"
        "## Limitations\n\n"
        "The route flow is not yet a mathematically strict Sankey model because risk components are not conserved physical "
        "flows. It is an alluvial trace diagram with width proportional to operator values. The current figures are static "
        "PNG plus self-contained HTML wrappers; richer interaction can be added later. The visual layer validates the "
        "traceability and interpretability of operator routes, but it does not prove industrial or clinical suitability "
        "of any external model. Signal and image-like experiments remain methodological validation examples, not domain "
        "deployment claims.\n\n"
        "The current visual layer is intentionally conservative. It favors reproducible static PNG and simple standalone "
        "HTML over a heavier browser application. It does not attempt to provide interactive filtering, animated route "
        "transitions or statistical confidence bands on every visual component. Those features can be added later, but the "
        "release-candidate objective is auditability: every visual must be regenerated from saved framework artifacts and "
        "must expose source_commit and verifier state. This is why the package includes both broad research figures and "
        "chapter-ready compact figures.\n\n"
        "```json\n" + json.dumps(manifest, ensure_ascii=False, indent=2) + "\n```\n",
        encoding="utf-8",
    )
    return out_path
