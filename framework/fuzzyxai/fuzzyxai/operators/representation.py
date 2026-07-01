from __future__ import annotations


def select_representation_class(diagnostics: list[dict]) -> str:
    return "FML-audit" if diagnostics else "F0"
