from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

from fuzzyxai.adapters.base import BaseAdapter
from fuzzyxai.core.explain_plan import ExplainPlan
from fuzzyxai.core.types import AdaptedInput, OperatorRoute
from fuzzyxai.proof.trace import build_proof_trace
from fuzzyxai.proof.verifier import VerificationResult, verify_proof_trace
from fuzzyxai.viz import save_proof_trace_json, save_route_json, write_traceability_artifacts
from fuzzyxai.viz.matplotlib_dashboard import render_dashboard
from fuzzyxai.core.route import build_route


class FuzzyXAI:
    """Runtime facade for using FuzzyXAI as an installable framework."""

    def __init__(self, plan: ExplainPlan | None = None):
        self.plan = plan or ExplainPlan.default()

    def run(self, adapted_input: AdaptedInput) -> OperatorRoute:
        return build_route(adapted_input)

    def run_payload(self, payload: dict, adapter: BaseAdapter) -> OperatorRoute:
        validation = adapter.validate_payload(payload)
        if not validation.valid:
            raise ValueError("; ".join(validation.errors))
        return self.run(adapter.to_adapted_input(payload))

    def verify(self, route: OperatorRoute) -> VerificationResult:
        return verify_proof_trace(build_proof_trace(route))

    def export_package(self, route: OperatorRoute, output_dir: str | Path) -> dict[str, Path]:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        trace = build_proof_trace(route)
        verification = verify_proof_trace(trace)
        paths = {
            "route": save_route_json(route, output / "route.json"),
            "proof_trace": save_proof_trace_json(trace, output / "proof_trace.json"),
            "dashboard": render_dashboard(route, output / "operator_dashboard.png"),
        }
        paths.update(write_traceability_artifacts(route, trace, verification, output))
        (output / "summary.json").write_text(
            json.dumps(
                {
                    "scenario_id": route.scenario_id,
                    "action_id": route.final_action_id or route.final_action,
                    "diagnostic_id": route.final_diagnostic_id or route.computed_result.get("diagnostic_id"),
                    "verifier": "passed" if verification.valid else "failed",
                    "source_commit": route.source_commit,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return paths

    def export_zip(self, route: OperatorRoute, output_zip: str | Path) -> Path:
        output_zip = Path(output_zip)
        tmp = output_zip.with_suffix("")
        if tmp.exists():
            shutil.rmtree(tmp)
        self.export_package(route, tmp)
        with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(tmp.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(tmp.parent).as_posix())
        return output_zip
