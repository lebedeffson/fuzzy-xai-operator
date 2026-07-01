from __future__ import annotations

import base64
import csv
import html
import json
import zipfile
from pathlib import Path
from typing import Any


ACTION_COLORS = {
    "accept": "#2e7d32",
    "lower_confidence": "#f2a900",
    "audit": "#d75a00",
    "defer_to_human": "#8e24aa",
    "block": "#b71c1c",
    "audit_report": "#5d6d7e",
}


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8") as file:
        return list(csv.DictReader(file))


def ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def action_for_rho(rho: float) -> str:
    if rho < 0.35:
        return "accept"
    if rho < 0.60:
        return "lower_confidence"
    return "audit"


def status_color(action: str) -> str:
    return ACTION_COLORS.get(action, "#607d8b")


def write_html_with_image(
    image_path: str | Path,
    html_path: str | Path,
    *,
    title: str,
    summary: dict[str, Any] | None = None,
) -> Path:
    image_path = Path(image_path)
    html_path = ensure_parent(html_path)
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    rows = ""
    for key, value in (summary or {}).items():
        rows += f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>\n"
    body = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 28px; color: #16202a; }}
    img {{ max-width: 100%; border: 1px solid #d8dde3; }}
    table {{ border-collapse: collapse; margin: 18px 0; }}
    th, td {{ border: 1px solid #d8dde3; padding: 6px 9px; text-align: left; }}
    th {{ background: #f2f4f7; }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <table>{rows}</table>
  <img src="data:image/png;base64,{encoded}" alt="{html.escape(title)}">
</body>
</html>
"""
    html_path.write_text(body, encoding="utf-8")
    return html_path


def read_package_json(package: str | Path, suffix: str) -> dict[str, Any] | None:
    path = Path(package)
    if path.is_dir():
        matches = [item for item in path.rglob(suffix) if item.is_file()]
        if not matches:
            return None
        return read_json(matches[0])
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if name.endswith(suffix):
                return json.loads(archive.read(name).decode("utf-8"))
    return None
