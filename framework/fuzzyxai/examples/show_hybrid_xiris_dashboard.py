import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FRAMEWORK = ROOT / "framework" / "fuzzyxai"
if str(FRAMEWORK) not in sys.path:
    sys.path.insert(0, str(FRAMEWORK))

from fuzzyxai import build_proof_trace, build_route, render_dashboard, save_route_json, verify_proof_trace
from fuzzyxai.examples.hybrid_xiris import get_input
from fuzzyxai.viz import save_proof_trace_json


def main() -> None:
    route = build_route(get_input())
    trace = build_proof_trace(route)
    verification = verify_proof_trace(trace)
    if not verification.valid:
        raise SystemExit(f"proof trace failed: {verification.errors}")

    route_json = ROOT / "reports/framework/hybrid_xiris_route.json"
    proof_json = ROOT / "reports/framework/hybrid_xiris_proof_trace.json"
    figure = ROOT / "reports/figures/hybrid_xiris_operator_dashboard.png"
    save_route_json(route, route_json)
    save_proof_trace_json(trace, proof_json)
    render_dashboard(route, figure)
    print(figure)
    print(route_json)
    print(proof_json)


if __name__ == "__main__":
    main()
