from __future__ import annotations

import csv
import json
import re
from .common import ARTIFACT_ROOT, ROOT, sha256_file


LOCATOR = re.compile(r"row=(\d+),column=([A-Za-z0-9_]+)$")


def validate() -> dict[str, int | str]:
    payload = json.loads((ARTIFACT_ROOT / "closure" / "h10_final_gold_evidence_map.json").read_text())
    checked = 0
    for entry in payload["entries"]:
        path = ROOT / entry["source_file"]
        if not path.is_file():
            raise RuntimeError(f"missing evidence source: {path}")
        if sha256_file(path) != entry["sha256"]:
            raise RuntimeError(f"evidence hash mismatch: {path}")
        match = LOCATOR.fullmatch(entry["locator"])
        if not match:
            raise RuntimeError(f"unsupported locator: {entry['locator']}")
        row_number, column = int(match.group(1)), match.group(2)
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        source_value = float(rows[row_number - 2][column])
        if abs(source_value - float(entry["value"])) > 1e-12:
            raise RuntimeError(f"evidence value mismatch: {path}:{entry['locator']}")
        if entry["status"] != "exploratory":
            raise RuntimeError("preconfirmatory evidence map contains a non-exploratory result")
        checked += 1
    if checked < 188:
        raise RuntimeError(f"evidence map is unexpectedly sparse: {checked}")
    return {"status": "PASS", "entries": checked}


def main() -> None:
    print(json.dumps(validate(), sort_keys=True))


if __name__ == "__main__":
    main()
