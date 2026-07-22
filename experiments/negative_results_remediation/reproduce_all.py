from __future__ import annotations

import argparse
import subprocess
import sys

from .common import ROOT, verify_protocol


def run(module: str, *args: str) -> None:
    subprocess.run([sys.executable, "-m", module, *args], cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    verify_protocol()
    objects = "1200" if args.smoke else "24000"
    events = "2000" if args.smoke else "500000"
    bootstrap = "200" if args.smoke else "5000"
    run("experiments.negative_results_remediation.prepare_data")
    for iteration in ("R1", "R2", "R3"):
        run("experiments.negative_results_remediation.fit_controller", "--iteration", iteration, "--objects", objects)
    run("experiments.negative_results_remediation.freeze")
    run("experiments.negative_results_remediation.run_h3_confirmatory")
    run("experiments.negative_results_remediation.run_h5")
    run("experiments.negative_results_remediation.run_h6", "envelope")
    run("experiments.negative_results_remediation.run_h6", "real")
    run("experiments.negative_results_remediation.run_replay", "--events", events, "--bootstrap", bootstrap)
    for stage in ("statistics", "claims", "evidence", "chapter", "check"):
        run("experiments.negative_results_remediation.closure", stage)
    if not args.smoke:
        run("experiments.negative_results_remediation.closure", "zip")


if __name__ == "__main__":
    main()
