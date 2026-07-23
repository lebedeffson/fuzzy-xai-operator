from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PACKAGE_ROOT.parents[1]
CONFIG_DIR = PACKAGE_ROOT / "configs"
PROTOCOL_DIR = PACKAGE_ROOT / "protocol"
ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "h10_c2"
DELIVERABLE_ROOT = REPO_ROOT / "deliverables" / "h10-c2-preconfirmatory"

