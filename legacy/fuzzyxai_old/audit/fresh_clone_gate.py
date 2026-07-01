from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from .common import ROOT, current_commit


def run(cmd: list[str], cwd: Path) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-url", default=str(ROOT))
    parser.add_argument("--commit", default=current_commit())
    parser.add_argument("--workdir", default="/tmp/fuzzyxai-e2e")
    parser.add_argument("--skip-install", action="store_true")
    args = parser.parse_args()
    target = Path(args.workdir)
    if target.exists():
        shutil.rmtree(target)
    run(["git", "clone", args.repo_url, str(target)], ROOT)
    run(["git", "checkout", args.commit], target)
    run([sys.executable, "-m", "venv", ".venv"], target)
    py = target / ".venv/bin/python"
    if not args.skip_install:
        run([str(py), "-m", "pip", "install", "-U", "pip"], target)
        run([str(py), "-m", "pip", "install", "-e", ".[dev]"], target)
    run(["make", "doctorate-release-check", f"PYTHON={py}"], target)
    print("fresh-clone-gate: PASS")


if __name__ == "__main__":
    main()
