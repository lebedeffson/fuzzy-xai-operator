#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from examples.integrations.mlflow_fuzzyxai_example import run_example


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/integrations/mlflow_demo"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(
            "reports/integrations/MLFLOW_INTEGRATION_REPORT.md"
        ),
    )
    args = parser.parse_args()
    result = run_example(args.output)
    status_path = (
        args.output / "MLFLOW_INTEGRATION_STATUS.json"
    )
    canonical_status = Path(
        "results/integrations/MLFLOW_INTEGRATION_STATUS.json"
    )
    canonical_status.parent.mkdir(parents=True, exist_ok=True)
    canonical_status.write_bytes(status_path.read_bytes())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        "\n".join(
            [
                "# MLflow Integration Report",
                "",
                f"- Status: `{result['status']}`",
                f"- MLflow: `{result['mlflow_version']}`",
                (
                    "- Storage: isolated local file-backed tracking "
                    "and model registry."
                ),
                (
                    f"- Registered model: "
                    f"`{result['registered_model_name']}` "
                    f"version `{result['model_version']}`"
                ),
                f"- Run ID: `{result['run_id']}`",
                f"- Route verification: `{result['route_verification']}`",
                "- External network during execution: `false`",
                (
                    "- Scope: reproducible engineering integration; "
                    "not a product-quality comparison or hypothesis test."
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
