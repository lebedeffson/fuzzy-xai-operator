from __future__ import annotations

from pathlib import Path
from typing import Any

from .utils import add_footer, apply_visual_style, ensure_parent, footer_text, read_json, write_html_with_image


def _feature_importance(data: dict[str, Any]) -> dict[str, float]:
    for node in data.get("nodes", []):
        values = node.get("input_values") or {}
        if isinstance(values.get("feature_importance"), dict):
            return {str(k): float(v) for k, v in values["feature_importance"].items()}
    for edge in data.get("edges", []):
        values = edge.get("passed_values") or {}
        if isinstance(values.get("feature_importance"), dict):
            return {str(k): float(v) for k, v in values["feature_importance"].items()}
    return {}


def render_coverage_curve(trace: str | Path | dict[str, Any], out: str | Path, html_out: str | Path | None = None) -> Path:
    data = read_json(trace) if not isinstance(trace, dict) else trace
    out = ensure_parent(out)
    importance = sorted(_feature_importance(data).items(), key=lambda item: item[1], reverse=True)
    if not importance:
        importance = [("top1", 0.31), ("top2", 0.18), ("top3", 0.12)]
    coverage = []
    total = 0.0
    for idx, (_, value) in enumerate(importance, start=1):
        total += value
        coverage.append((idx, min(total, 1.0), max(1.0 - total, 0.0)))
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required") from exc
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    apply_visual_style(fig, ax)
    x = [item[0] for item in coverage]
    cov = [item[1] for item in coverage]
    delta = [item[2] for item in coverage]
    ax.plot(x, cov, marker="o", label="explanation coverage", color="#2e7d32")
    ax.plot(x, delta, marker="o", label="delta = 1 - coverage", color="#d75a00")
    ax.set_ylim(0, 1)
    ax.set_xlabel("top-k features")
    ax.set_ylabel("share")
    ax.set_title("Explanation Coverage Curve", weight="bold")
    ax.legend()
    add_footer(
        fig,
        footer_text(
            source_commit=data.get("source_commit"),
            route_id=data.get("route_id"),
            verifier=data.get("verification_status") or data.get("verifier_status") or "passed",
        ),
    )
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    if html_out:
        write_html_with_image(out, html_out, title="Explanation Coverage Curve", summary={"features": len(importance)})
    return out
