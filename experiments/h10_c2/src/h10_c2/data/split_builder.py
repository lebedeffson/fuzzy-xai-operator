from __future__ import annotations


def split_counts(total: int, fractions: dict[str, float]) -> dict[str, int]:
    if total <= 0 or abs(sum(fractions.values()) - 1.0) > 1e-9:
        raise ValueError("positive total and normalized fractions are required")
    names = list(fractions)
    counts = {name: int(total * fractions[name]) for name in names}
    counts[names[-1]] += total - sum(counts.values())
    return counts

