from __future__ import annotations

import argparse
import os

from .runner import (
    freeze_method,
    gate,
    generate,
    generate_sealed,
    power,
    run,
    score_sealed,
    stability,
    template_audit,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=(
            "template-audit",
            "generate-development",
            "run-development",
            "freeze",
            "generate-protocol-validation",
            "run-protocol-validation",
            "stability",
            "power",
            "gate",
            "generate-sealed",
            "score-sealed",
        ),
    )
    args = parser.parse_args()
    if args.command == "template-audit":
        print(template_audit())
    elif args.command == "generate-development":
        print(generate("development"))
    elif args.command == "run-development":
        print(run("development"))
    elif args.command == "freeze":
        print(freeze_method())
    elif args.command == "generate-protocol-validation":
        print(generate("protocol_validation"))
    elif args.command == "run-protocol-validation":
        print(run("protocol_validation"))
    elif args.command == "stability":
        print(stability())
    elif args.command == "power":
        print(power())
    elif args.command == "gate":
        print(gate())
    elif args.command == "generate-sealed":
        print(generate_sealed())
    elif args.command == "score-sealed":
        score_sealed(os.environ.get("APPROVAL"))


if __name__ == "__main__":
    main()
