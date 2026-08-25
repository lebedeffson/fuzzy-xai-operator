"""``python -m fuzzyxai.verbalization doctor`` — read-only diagnostic.

Never installs, downloads, pulls a model, or starts a server. Only reports
what is already true and prints the exact command the user would need to run
themselves to fix a gap.
"""

from __future__ import annotations

import argparse
import sys

from fuzzyxai.verbalization.backends.ollama import OllamaBackend


def _run_doctor(model: str | None, host: str | None) -> int:
    print("FuzzyXAI verbalization doctor")
    print("==============================")
    print()
    print("[deterministic path] OK — result.summary() / result.verbalize() with no")
    print("  backend always works offline; no network call is ever made for this path.")
    print()

    backend = OllamaBackend(model=model, host=host)
    print(f"[ollama] host={backend.host}  model={backend.model}")
    health = backend.check()
    if not health.reachable:
        print(f"[ollama] UNREACHABLE — {health.detail}")
        print()
        print("Next step (run this yourself — the doctor command does not run it for you):")
        print("  ollama serve")
        return 0

    print("[ollama] reachable: OK")
    if health.model_present:
        print(f"[ollama] model '{backend.model}': present")
        print()
        print("Everything needed for result.verbalize_detailed(backend=OllamaBackend()) is ready.")
    else:
        print(f"[ollama] model '{backend.model}': MISSING")
        if health.models:
            print(f"[ollama] models currently available: {', '.join(health.models)}")
        print()
        print("Next step (run this yourself — the doctor command does not run it for you):")
        print(f"  ollama pull {backend.model}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m fuzzyxai.verbalization")
    subparsers = parser.add_subparsers(dest="command", required=True)
    doctor = subparsers.add_parser("doctor", help="check the deterministic and Ollama verbalization paths")
    doctor.add_argument("--model", default=None, help="override the model to check (default: FUZZYXAI_OLLAMA_MODEL or qwen3:1.7b)")
    doctor.add_argument("--host", default=None, help="override the Ollama host to check (default: FUZZYXAI_OLLAMA_HOST or http://localhost:11434)")
    args = parser.parse_args(argv)
    if args.command == "doctor":
        return _run_doctor(args.model, args.host)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
