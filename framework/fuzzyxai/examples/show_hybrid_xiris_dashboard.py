from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FRAMEWORK = ROOT / "framework" / "fuzzyxai"
if str(FRAMEWORK) not in sys.path:
    sys.path.insert(0, str(FRAMEWORK))

from fuzzyxai.viz import load_route_from_proof, render_operator_dashboard


def main() -> None:
    proof_path = ROOT / "applications/scenarios/hybrid_xiris/proof/hybrid_xiris_proof_package.json"
    route = load_route_from_proof(proof_path)
    route_json = ROOT / "reports/routes/hybrid_xiris_route.json"
    route.write_json(route_json)
    site_route = ROOT / "site/dubnaxai/public/routes/hybrid_xiris_route.json"
    site_route.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(route_json, site_route)
    figure = ROOT / "reports/figures/hybrid_xiris_operator_dashboard.png"
    render_operator_dashboard(route, figure)
    print(figure)
    print(route_json)


if __name__ == "__main__":
    main()

