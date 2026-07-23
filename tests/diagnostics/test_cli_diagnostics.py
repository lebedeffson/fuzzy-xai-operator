from __future__ import annotations

import json

from fuzzyxai.cli import main


def test_diagnose_cli_writes_json_and_html(tmp_path, invalid_route: dict) -> None:
    route = tmp_path / "route.json"
    output = tmp_path / "report.json"
    html = tmp_path / "report.html"
    route.write_text(json.dumps(invalid_route), encoding="utf-8")
    status = main(
        [
            "diagnose",
            "--route",
            str(route),
            "--output",
            str(output),
            "--repair-plan",
            "--html",
            str(html),
        ]
    )
    assert status == 2
    assert json.loads(output.read_text())["route_status"] == "invalid"
    assert "Минимальный разрез" in html.read_text()


def test_batch_cli_writes_machine_outputs(tmp_path, valid_route: dict, invalid_route: dict) -> None:
    routes = tmp_path / "routes.jsonl"
    routes.write_text(
        "\n".join(json.dumps(item) for item in (valid_route, invalid_route)) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "reports"
    assert main(["diagnose-batch", "--input", str(routes), "--output", str(output), "--repair-plan"]) == 0
    assert (output / "batch_summary.json").is_file()
    assert (output / "batch_reports.csv").is_file()
