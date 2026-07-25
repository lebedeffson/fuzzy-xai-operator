from __future__ import annotations

import hashlib
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RegisteredRepair:
    repair_id: str
    family: str
    target_file: str
    operation: Callable[[Path], None]
    expressible: bool = True


@dataclass(frozen=True)
class IncidentExecutionReport:
    statuses: tuple[str, ...]
    bug_reproduced: bool
    repair_executed: bool
    fail_to_pass_success: bool
    regression_success: bool
    recertification_success: bool
    before_sha256: str
    after_sha256: str
    stdout: str
    stderr: str


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if ".git" in path.parts or "__pycache__" in path.parts:
            continue
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


class IncidentSandboxExecutor:
    """Execute a registered repair in an isolated copy, never in the source tree."""

    def execute(
        self,
        source_root: Path,
        repair: RegisteredRepair,
        *,
        fail_to_pass_command: tuple[str, ...],
        regression_command: tuple[str, ...],
        recertify: Callable[[Path], bool],
        timeout_seconds: int = 300,
    ) -> IncidentExecutionReport:
        if not repair.expressible:
            return IncidentExecutionReport(
                ("REPAIR_NOT_EXPRESSIBLE",),
                False,
                False,
                False,
                False,
                False,
                "",
                "",
                "",
                "",
            )
        with tempfile.TemporaryDirectory(prefix="fuzzyxai-h10-c5b-") as temporary:
            sandbox = Path(temporary) / "repository"
            shutil.copytree(source_root, sandbox, ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache"))
            before = _tree_digest(sandbox)
            initial = subprocess.run(
                fail_to_pass_command,
                cwd=sandbox,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
            statuses = ["BUG_REPRODUCED" if initial.returncode else "BUG_NOT_REPRODUCED"]
            if initial.returncode == 0:
                return IncidentExecutionReport(
                    tuple(statuses),
                    False,
                    False,
                    False,
                    False,
                    False,
                    before,
                    before,
                    initial.stdout,
                    initial.stderr,
                )
            repair.operation(sandbox)
            statuses.append("REPAIR_EXECUTED")
            after_repair = subprocess.run(
                fail_to_pass_command,
                cwd=sandbox,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
            fail_to_pass = after_repair.returncode == 0
            statuses.append("FAIL_TO_PASS_PASS" if fail_to_pass else "FAIL_TO_PASS_FAIL")
            regression = subprocess.run(
                regression_command,
                cwd=sandbox,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
            regression_success = regression.returncode == 0
            statuses.append("REGRESSION_PASS" if regression_success else "REGRESSION_FAIL")
            recertification = fail_to_pass and regression_success and bool(recertify(sandbox))
            statuses.append("RECERTIFICATION_PASS" if recertification else "RECERTIFICATION_FAIL")
            return IncidentExecutionReport(
                tuple(statuses),
                True,
                True,
                fail_to_pass,
                regression_success,
                recertification,
                before,
                _tree_digest(sandbox),
                f"{initial.stdout}\n{after_repair.stdout}\n{regression.stdout}",
                f"{initial.stderr}\n{after_repair.stderr}\n{regression.stderr}",
            )
