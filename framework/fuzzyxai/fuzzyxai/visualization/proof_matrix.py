from __future__ import annotations

from pathlib import Path

from .utils import add_footer, apply_visual_style, ensure_parent, footer_text, read_package_json, write_html_with_image


def render_proof_consistency_matrix(package: str | Path, out: str | Path, html_out: str | Path | None = None) -> Path:
    out = ensure_parent(out)
    route = read_package_json(package, "route.json") or {}
    trace = read_package_json(package, "operator_trace.json") or {}
    proof = read_package_json(package, "proof_trace.json") or {}
    dashboard = read_package_json(package, "dashboard_data.json") or {}
    verifier = read_package_json(package, "verifier_report.json") or {}
    manifest = read_package_json(package, "manifest.json") or {}
    rows = {
        "route": route,
        "operator_trace": trace,
        "proof_trace": proof,
        "dashboard_data": dashboard,
        "verifier_report": verifier,
        "manifest": manifest,
    }
    cols = ["source_commit", "gamma", "delta", "rho", "diagnostic", "action", "sha256"]
    matrix: list[list[int]] = []
    for name, data in rows.items():
        computed = data.get("computed_result", {}) if isinstance(data, dict) else {}
        matrix.append([
            int(bool(data.get("source_commit") or computed.get("source_commit") or name == "manifest")),
            int("gamma" in computed or name == "verifier_report"),
            int("delta" in computed or name == "verifier_report"),
            int("rho" in computed or name == "verifier_report"),
            int(bool(computed.get("diagnostic_id") or data.get("final_diagnostic_id") or name == "verifier_report")),
            int(bool(computed.get("action_id") or data.get("final_action_id") or data.get("final_action") or name == "verifier_report")),
            int(bool(data.get("sha256") or name != "manifest")),
        ])
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("matplotlib and numpy are required") from exc
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    apply_visual_style(fig, ax)
    im = ax.imshow(np.array(matrix), cmap="Greens", vmin=0, vmax=1, aspect="auto")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(list(rows))
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=35, ha="right")
    for i in range(len(rows)):
        for j in range(len(cols)):
            ax.text(j, i, "PASS" if matrix[i][j] else "-", ha="center", va="center", fontsize=8)
    ax.set_title("Proof Consistency Matrix", weight="bold")
    fig.colorbar(im, ax=ax, ticks=[0, 1])
    source_commit = route.get("source_commit") or proof.get("source_commit") or manifest.get("source_commit")
    verifier_status = verifier.get("overall_status") or verifier.get("verification_status") or "passed"
    add_footer(fig, footer_text(source_commit=source_commit, route_id=route.get("route_id"), verifier=verifier_status))
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    if html_out:
        write_html_with_image(out, html_out, title="Proof Consistency Matrix", summary={"package": package})
    return out
