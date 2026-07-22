from __future__ import annotations

import argparse
import subprocess
import sys

from .common import ROOT, verify_protocol_hash


def run_module(module: str, *arguments: str) -> None:
    subprocess.run([sys.executable, "-m", module, *arguments], cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    verify_protocol_hash()
    if args.smoke:
        run_module("experiments.chapter4_v13.smoke")
        return
    run_module("experiments.chapter4_v13.prepare_data")
    run_module("experiments.chapter4_v13.train_or_load_model")
    run_module("experiments.chapter4_v13.generate_explanations", "--split", "validation", "--objects", "2000")
    run_module("experiments.chapter4_v13.generate_explanations", "--split", "sealed_test", "--objects", "2000")
    run_module("experiments.chapter4_v13.run_policies", "--stage", "pre-score")
    run_module("experiments.chapter4_v13.run_policies", "--stage", "score")
    run_module("experiments.chapter4_v13.run_route_faults")
    run_module("experiments.chapter4_v13.benchmark_end_to_end")
    run_module("experiments.chapter4_v13.reproduce_case")
    run_module("experiments.chapter4_v13.build_tables")
    run_module("experiments.chapter4_v13.build_figures")
    run_module("experiments.chapter4_v13.validate_evidence")
    run_module("experiments.chapter4_v13.build_chapter")
    run_module("experiments.chapter4_v13.build_closure")
    run_module("experiments.chapter4_v13.validate_document")
    run_module("experiments.chapter4_v13.build_release")


if __name__ == "__main__":
    main()
