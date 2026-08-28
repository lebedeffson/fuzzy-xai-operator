import ast
from pathlib import Path


def test_public_runner_does_not_recompute_frozen_system_quantities():
    path = Path("chapter6_medical_validation/ophthalmology/scripts/run_fuzzyxai.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    forbidden = {"compute_gamma", "compute_delta", "compute_u_m", "compute_i_pre", "compute_strict_rho"}
    called = {node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert called.isdisjoint(forbidden)
