def categorical_agreement(left: list[str], right: list[str]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("paired non-empty ratings are required")
    return sum(a == b for a, b in zip(left, right)) / len(left)

