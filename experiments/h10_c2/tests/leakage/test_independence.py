from __future__ import annotations

import os
import subprocess
import sys

from h10_c2.audit import audit_baselines, audit_oracle
from h10_c2.paths import PACKAGE_ROOT


def test_baselines_pass_ast_independence_audit() -> None:
    assert audit_baselines()["status"] == "PASS"


def test_oracle_passes_import_audit() -> None:
    assert audit_oracle()["status"] == "PASS"


def test_oracle_imports_without_fuzzyxai_on_pythonpath() -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PACKAGE_ROOT / "src")
    script = "from h10_c2.oracle.equivalent_cuts import enumerate_optimal_cuts; assert enumerate_optimal_cuts([['a']], {'a': 1})[1] == 1"
    result = subprocess.run([sys.executable, "-I", "-c", f"import sys; sys.path.insert(0, {str(PACKAGE_ROOT / 'src')!r}); {script}"], env=env, check=False)
    assert result.returncode == 0

