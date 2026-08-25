"""Reference corpus + similar-case evidence (P1 / P1.1).

    python examples/02_tabular_similarity.py

Registering ``reference_data``/``reference_labels``/``reference_ids`` once on
``wrap()`` makes ``result.similar_cases`` populate automatically on every
``explain_one()`` call — no per-call flag needed, and no similarity evidence
at all when no reference corpus is registered (never fabricated).
"""

from __future__ import annotations

from fuzzyxai import FuzzyXAI
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def main() -> None:
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
    model = LogisticRegression(max_iter=2000).fit(X_train, y_train)
    train_ids = [f"train_{i}" for i in range(len(X_train))]

    fx = FuzzyXAI.wrap(
        model,
        reference_data=X_train,
        reference_labels=y_train,
        reference_ids=train_ids,
    )
    result = fx.explain_one(X_test[0])

    print(f"{len(result.similar_cases)} similar reference case(s) found:\n")
    for case in result.similar_cases:
        print(
            f"  {case['reference_object_id']} — score {case['score']:.3f}, "
            f"rank {case['reference_rank']} of {case['reference_count']}, label {case['reference_label']}"
        )
        print(f"    matched: {list(case['matched_features'])[:3]}")
        print(f"    different: {list(case['different_features'])[:3]}")

    # The compact "## Похожие примеры" digest is folded into summary().
    print()
    print(result.summary())


if __name__ == "__main__":
    main()
