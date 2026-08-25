"""The minimal end-to-end example: an already-trained sklearn model, explained.

    python examples/01_tabular_sklearn.py

No reference corpus, no raw object, no verbalizer — just prediction plus
claim-grounded evidence. See 02_tabular_similarity.py for reference cases,
03/04 for text/image raw objects, 05 for the optional SLM verbalizer.
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

    # The one line that wraps any already-trained model for explanation.
    fx = FuzzyXAI.wrap(model, task="classification")
    result = fx.explain_one(X_test[0])

    print("Native model prediction:", model.predict(X_test[:1]))
    print("FuzzyXAI prediction:    ", result.prediction.predictions)
    print()
    print(result.summary())  # deterministic, works fully offline

    # Structured, claim-grounded evidence, not just prose:
    print(f"{len(result.claims)} typed claims, action = {result.action!r}")

    # Three JSON projections of the *same* canonical result (P2):
    result.export_json("/tmp/fuzzyxai_example_01_compact.json", detail="compact")
    result.export_json("/tmp/fuzzyxai_example_01_audit.json", detail="audit")

    # A rendered figure of the object with evidence overlaid.
    result.visualize(output="/tmp/fuzzyxai_example_01.png")
    print("\nWrote /tmp/fuzzyxai_example_01_compact.json, _audit.json, and .png")


if __name__ == "__main__":
    main()
