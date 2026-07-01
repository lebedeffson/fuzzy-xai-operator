from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path

from fuzzyxai.adapters import get_adapter, list_adapters
from fuzzyxai.core.types import OperatorEdge, OperatorNode, OperatorRoute, ProofTrace
from fuzzyxai.operators import list_operators
from fuzzyxai.proof.verifier import verify_proof_trace
from fuzzyxai.runtime import FuzzyXAI
from fuzzyxai.schemas import list_schemas, validate_file
from fuzzyxai.visualization import (
    render_action_boundary,
    render_coverage_curve,
    render_gamma_delta_action_map,
    render_operator_trace_heatmap,
    render_proof_consistency_matrix,
    render_representation_atlas,
    render_risk_waterfall,
    render_route_sankey,
)
from fuzzyxai.viz.matplotlib_dashboard import render_dashboard


def _read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _route_from_dict(data: dict) -> OperatorRoute:
    nodes = [OperatorNode(**node) for node in data.get("nodes", [])]
    edges = [OperatorEdge(**edge) for edge in data.get("edges", [])]
    return OperatorRoute(
        scenario_id=data["scenario_id"],
        title=data.get("title") or data.get("scenario_title_ru") or data["scenario_id"],
        nodes=nodes,
        edges=edges,
        computed_result=data.get("computed_result", {}),
        diagnostics=data.get("diagnostics", []),
        final_action=data.get("final_action") or data.get("final_action_id", ""),
        verifier_status=data.get("verifier_status", "UNVERIFIED"),
        source_commit=data.get("source_commit", "unknown"),
        route_id=data.get("route_id", ""),
        scenario_title_ru=data.get("scenario_title_ru", ""),
        created_at=data.get("created_at", ""),
        final_diagnostic_id=data.get("final_diagnostic_id", ""),
        final_action_id=data.get("final_action_id", ""),
        proof_ref=data.get("proof_ref", ""),
        dashboard_ref=data.get("dashboard_ref", ""),
        verification_summary=data.get("verification_summary", {}),
    )


def cmd_run(args: argparse.Namespace) -> int:
    payload = _read_json(args.payload)
    adapter = get_adapter(args.adapter)
    route = FuzzyXAI().run_payload(payload, adapter)
    FuzzyXAI().export_package(route, args.out)
    print(f"fuzzyxai run: PASS {args.out}")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    proof = ProofTrace(**_read_json(args.proof))
    result = verify_proof_trace(proof)
    print("fuzzyxai verify:", "PASS" if result.valid else "FAIL")
    for error in result.errors:
        print(f"- {error}")
    return 0 if result.valid else 1


def cmd_render(args: argparse.Namespace) -> int:
    route = _route_from_dict(_read_json(args.route))
    render_dashboard(route, args.out)
    print(f"fuzzyxai render: PASS {args.out}")
    return 0


def cmd_package(args: argparse.Namespace) -> int:
    route_path = Path(args.route)
    base = route_path.parent
    output = Path(args.out)
    if output.exists():
        output.unlink()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(base.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(base.parent).as_posix())
    print(f"fuzzyxai package: PASS {output}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    result = validate_file(args.payload, args.schema)
    print("fuzzyxai validate:", "PASS" if result.valid else "FAIL")
    for error in result.errors:
        print(f"- {error}")
    return 0 if result.valid else 1


def cmd_list_adapters(_: argparse.Namespace) -> int:
    print(json.dumps(list_adapters(), ensure_ascii=False, indent=2))
    return 0


def cmd_list_operators(_: argparse.Namespace) -> int:
    print(json.dumps(list_operators(), ensure_ascii=False, indent=2))
    return 0


def cmd_visualize(args: argparse.Namespace) -> int:
    html_out = args.html_out
    if args.visual == "route-sankey":
        render_route_sankey(args.route, args.out, html_out)
    elif args.visual == "risk-waterfall":
        render_risk_waterfall(args.trace, args.out, html_out)
    elif args.visual == "gamma-delta-map":
        render_gamma_delta_action_map(args.results, args.out, html_out)
    elif args.visual == "trace-heatmap":
        render_operator_trace_heatmap(args.results, args.out, html_out)
    elif args.visual == "representation-atlas":
        render_representation_atlas(args.results, args.out, html_out)
    elif args.visual == "coverage-curve":
        render_coverage_curve(args.trace, args.out, html_out)
    elif args.visual == "action-boundary":
        render_action_boundary(args.route, args.out, html_out)
    elif args.visual == "proof-matrix":
        render_proof_consistency_matrix(args.package, args.out, html_out)
    else:  # pragma: no cover
        raise ValueError(args.visual)
    print(f"fuzzyxai visualize {args.visual}: PASS {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fuzzyxai")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run")
    run.add_argument("--payload", required=True)
    run.add_argument("--adapter", required=True)
    run.add_argument("--out", required=True)
    run.set_defaults(func=cmd_run)

    verify = sub.add_parser("verify")
    verify.add_argument("--route")
    verify.add_argument("--proof", required=True)
    verify.set_defaults(func=cmd_verify)

    render = sub.add_parser("render")
    render.add_argument("--route", required=True)
    render.add_argument("--out", required=True)
    render.set_defaults(func=cmd_render)

    package = sub.add_parser("package")
    package.add_argument("--route", required=True)
    package.add_argument("--out", required=True)
    package.set_defaults(func=cmd_package)

    validate = sub.add_parser("validate")
    validate.add_argument("--payload", required=True)
    validate.add_argument("--schema", required=True, choices=list_schemas())
    validate.set_defaults(func=cmd_validate)

    sub.add_parser("list-adapters").set_defaults(func=cmd_list_adapters)
    sub.add_parser("list-operators").set_defaults(func=cmd_list_operators)

    visualize = sub.add_parser("visualize")
    visual_sub = visualize.add_subparsers(dest="visual", required=True)

    route_sankey = visual_sub.add_parser("route-sankey")
    route_sankey.add_argument("--route", required=True)
    route_sankey.add_argument("--out", required=True)
    route_sankey.add_argument("--html-out")
    route_sankey.set_defaults(func=cmd_visualize)

    risk_waterfall = visual_sub.add_parser("risk-waterfall")
    risk_waterfall.add_argument("--trace", required=True)
    risk_waterfall.add_argument("--out", required=True)
    risk_waterfall.add_argument("--html-out")
    risk_waterfall.set_defaults(func=cmd_visualize)

    gamma_delta = visual_sub.add_parser("gamma-delta-map")
    gamma_delta.add_argument("--results", required=True)
    gamma_delta.add_argument("--out", required=True)
    gamma_delta.add_argument("--html-out")
    gamma_delta.set_defaults(func=cmd_visualize)

    trace_heatmap = visual_sub.add_parser("trace-heatmap")
    trace_heatmap.add_argument("--results", required=True)
    trace_heatmap.add_argument("--out", required=True)
    trace_heatmap.add_argument("--html-out")
    trace_heatmap.set_defaults(func=cmd_visualize)

    atlas = visual_sub.add_parser("representation-atlas")
    atlas.add_argument("--results", required=True)
    atlas.add_argument("--out", required=True)
    atlas.add_argument("--html-out")
    atlas.set_defaults(func=cmd_visualize)

    coverage = visual_sub.add_parser("coverage-curve")
    coverage.add_argument("--trace", required=True)
    coverage.add_argument("--out", required=True)
    coverage.add_argument("--html-out")
    coverage.set_defaults(func=cmd_visualize)

    boundary = visual_sub.add_parser("action-boundary")
    boundary.add_argument("--route", required=True)
    boundary.add_argument("--out", required=True)
    boundary.add_argument("--html-out")
    boundary.set_defaults(func=cmd_visualize)

    proof_matrix = visual_sub.add_parser("proof-matrix")
    proof_matrix.add_argument("--package", required=True)
    proof_matrix.add_argument("--out", required=True)
    proof_matrix.add_argument("--html-out")
    proof_matrix.set_defaults(func=cmd_visualize)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
