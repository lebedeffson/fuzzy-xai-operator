from __future__ import annotations

import argparse

from .audit import (
    build_methodology_audit,
    hash_manifest,
    run_independence_audit,
    run_leakage_audit,
)
from .reporting import (
    build_evidence_map,
    build_handoff_zip,
    build_tables,
    build_validation_report,
)
from .runner import (
    analyze,
    freeze,
    gate,
    generate,
    power,
    run,
    score_sealed,
    stability,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=(
            "generate-development",
            "run-development",
            "freeze",
            "generate-protocol-validation",
            "run-protocol-validation",
            "stability",
            "power",
            "audits",
            "gate",
            "reports",
            "package",
            "score-sealed",
        ),
    )
    args = parser.parse_args()
    if args.command == "generate-development":
        print(generate("development"))
    elif args.command == "run-development":
        print(run("development"))
        print(analyze("development"))
    elif args.command == "freeze":
        print(freeze())
    elif args.command == "generate-protocol-validation":
        print(generate("protocol_validation"))
    elif args.command == "run-protocol-validation":
        print(run("protocol_validation"))
        print(analyze("protocol_validation"))
    elif args.command == "stability":
        print(stability())
    elif args.command == "power":
        print(power())
    elif args.command == "audits":
        print(run_independence_audit())
        print(run_leakage_audit())
        print(build_methodology_audit())
    elif args.command == "gate":
        print(gate())
    elif args.command == "reports":
        print(*build_tables(), sep="\n")
        print(build_evidence_map())
        print(build_validation_report())
        print(hash_manifest())
    elif args.command == "package":
        print(build_handoff_zip())
    elif args.command == "score-sealed":
        score_sealed()


if __name__ == "__main__":
    main()

