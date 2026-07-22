from __future__ import annotations

import subprocess
import sys

from .common import ARTIFACT_ROOT, ROOT


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    python = sys.executable
    run([python, "-m", "pytest", "tests/h10", "-q"])
    completion = ARTIFACT_ROOT / "opening" / "completion_marker.json"
    if not completion.exists():
        raise SystemExit("H10 sealed scoring is not present. Prepare/freeze/score only through the registered staged workflow.")
    for module in (
        "experiments.h10.compute_statistics",
        "experiments.h10.run_replay",
        "experiments.h10.audit_methodology",
        "experiments.h10.build_tables",
        "experiments.h10.build_figures",
    ):
        run([python, "-m", module])
    run([python, "-m", "experiments.h10.validate_evidence", "validate"])
    run([python, "-m", "experiments.h10.package"])


if __name__ == "__main__":
    main()
