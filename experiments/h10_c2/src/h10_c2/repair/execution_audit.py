from __future__ import annotations


def validate_audit(trace: list[dict]) -> None:
    required = {"step", "target", "precondition_passed", "status", "before_sha256", "after_sha256"}
    if any(not required.issubset(item) for item in trace):
        raise ValueError("repair audit trace is incomplete")

